"""Agente RAG DIEM principale.

Implementa il facade RAGAgent e la factory RAGAgentFactory per
l'assemblaggio e l'interazione con l'agente conversazionale.
"""

import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from langchain_core.messages import AIMessage
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import AppSettings, load_settings
from agent.callbacks import (
    RAGObservabilityHandler,
    create_observability_handler,
    InteractionLogHandler,
    create_interaction_log_handler,
)
from agent.memory import SmartConversationMemory, create_conversation_memory, _is_meta_query
from agent.prompts import get_agent_system_prompt
from agent.guardrails import GuardrailsChecker, build_guardrails_checker
from agent.tools import set_retrieval_engine, set_chat_history, get_all_tools, get_last_search_meta
from agent.llm_providers import create_chat_model
from retrieval.engine import RetrievalEngine, QueryOptimizer

logger = logging.getLogger(__name__)


def _get_temporal_system_message() -> dict:
    """Costruisce un dizionario messaggio di sistema con il contesto temporale.

    Utilizza Negative Prompting per evitare l'uso forzato dell'anno
    quando l'utente non lo specifica esplicitamente.

    Returns:
        Dizionario con chiavi 'role' e 'content' per l'iniezione nel contesto.
    """
    now = datetime.now()
    giorni = ["lunedi", "martedi", "mercoledi", "giovedi", "venerdi", "sabato", "domenica"]
    mesi = [
        "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
        "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre",
    ]

    anno_solare = now.year
    anno_accademico = anno_solare if now.month >= 10 else anno_solare - 1

    content = (
        f"[CONTESTO TEMPORALE - Oggi: {giorni[now.weekday()]} {now.day} {mesi[now.month - 1]} {anno_solare}. "
        f"Anno Accademico corrente: {anno_accademico}/{anno_accademico + 1}.]\n"
        f"REGOLA CRITICA PER I TOOL: Devi valorizzare il parametro 'anno' SOLO E SOLTANTO SE "
        f"l'utente fa esplicito riferimento al tempo (es. 'quest'anno', 'nel 2024', 'l'anno scorso'). "
        f"Se la domanda NON contiene riferimenti temporali espliciti, lascia SEMPRE il parametro 'anno' VUOTO/NULL.]"
    )

    return {"role": "system", "content": content}


class RAGAgent:
    """Facade principale per l'interazione con l'agente RAG DIEM.

    I guardrails sono integrati direttamente in chat():
    check_input() sulla query originale prima del rewriting,
    check_output() sulla risposta finale dopo l'agente.
    Le interazioni bloccate dai guardrails non vengono salvate in memoria.
    """

    def __init__(
        self,
        agent_graph,
        memory: SmartConversationMemory,
        settings: AppSettings,
        interaction_logger: InteractionLogHandler,
        query_optimizer: Optional[QueryOptimizer] = None,
        guardrails_checker: Optional[GuardrailsChecker] = None,
    ):
        """Inizializza l'agente RAG.

        Args:
            agent_graph: Grafo dell'agente LangChain compilato.
            memory: Memoria conversazionale.
            settings: Configurazione dell'applicazione.
            interaction_logger: Handler per il logging delle interazioni.
            query_optimizer: Ottimizzatore per il rewriting delle query.
            guardrails_checker: Checker per i guardrails input/output.
        """
        self._agent = agent_graph
        self._memory = memory
        self._settings = settings
        self._interaction_logger = interaction_logger
        self._query_optimizer = query_optimizer
        self._guardrails_checker = guardrails_checker
        self._traces: List[Dict[str, Any]] = []

    def chat(self, user_query: str) -> dict:
        """Punto di ingresso principale per l'interazione con l'agente.

        Flusso lineare con guardrails:
        1. Input guardrail check (query originale, prima del rewriting)
        2. Memory: costruzione messaggi con storico
        3. Query rewriting: risoluzione coreferenze
        4. Cache: corto circuito se match esatto
        5. Memory: salvataggio query
        6. Iniezione contesto temporale
        7. Chat history injection per tool
        8. Agent: singola invocazione
        9. Output guardrail check (risposta finale)
        10. Logging e memory update (solo se non bloccato)

        Le interazioni bloccate dai guardrails (input o output) non
        vengono salvate in memoria per evitare inquinamento.

        Args:
            user_query: Domanda dell'utente.

        Returns:
            Dizionario con response, blocked, block_reason, trace e turn.
        """
        logger.info("USER QUERY (pulita): %s", user_query)

        if self._guardrails_checker:
            input_allowed, block_message = self._guardrails_checker.check_input(user_query)
            if not input_allowed:
                logger.warning("INPUT BLOCCATO dal guardrail: '%s...'", user_query[:80])

                return {
                    "response": block_message,
                    "blocked": True,
                    "block_reason": "input_guardrail",
                    "trace": {
                        "tool_name": "(Bloccato da Input Guardrail)",
                        "tools_invoked": [],
                        "rewritten_query": "",
                        "multi_queries": [],
                        "collection": "",
                        "metadata_filter": None,
                        "top_links": [],
                    },
                    "turn": self._memory.turn_count,
                }

        messages = self._memory.get_messages_for_agent(user_query)

        rewritten_query = user_query
        if self._query_optimizer and not _is_meta_query(user_query):
            rewritten_query = self._rewrite_query(user_query)
            if rewritten_query != user_query:
                logger.info("Query rewritten: '%s' -> '%s'", user_query, rewritten_query)
                self._replace_query_in_messages(messages, user_query, rewritten_query)

        cached_response = self._memory.find_exact_match(rewritten_query)

        if cached_response:
            logger.info("RISPOSTA RECUPERATA DA CACHE PER: %s", rewritten_query)

            return {
                "response": cached_response,
                "blocked": False,
                "block_reason": None,
                "trace": {
                    "tool_name": "(Risposta da Cache)",
                    "tools_invoked": [],
                    "rewritten_query": "",
                    "multi_queries": [],
                    "collection": "",
                    "metadata_filter": None,
                    "top_links": [],
                },
                "turn": self._memory.turn_count,
            }

        turn_number = self._memory.add_user_message(rewritten_query)

        temporal_msg = _get_temporal_system_message()
        if messages and messages[-1]["role"] == "user":
            messages.insert(-1, temporal_msg)
        else:
            messages.append(temporal_msg)

        logger.debug("MESSAGGI INVIATI ALL'AGENTE: %s", messages)

        set_chat_history(self._memory.get_langchain_history())

        obs_handler = create_observability_handler(
            self._settings.observability,
            conversation_turn=turn_number,
        )

        try:
            result = self._agent.invoke(
                {"messages": messages},
                config={
                    "callbacks": [obs_handler],
                    "recursion_limit": self._settings.guardrails.max_agent_iterations,
                },
            )
            response_text = self._extract_final_response(result)
            logger.info("TESTO DI RISPOSTA AGENTE AL TURNO %d: %s", turn_number, response_text)

        except Exception as e:
            error_str = str(e).lower()
            error_type = type(e).__name__

            if ("recursion" in error_str or "recursion" in error_type.lower()
                    or "iteration" in error_str):
                logger.error(
                    "LOOP RILEVATO - Agente terminato forzatamente. Query: '%s'",
                    user_query[:80],
                )
                response_text = (
                    "Mi scuso, ho riscontrato difficolta nell'elaborare la tua "
                    "domanda. Prova a riformularla in modo piu specifico."
                )
            else:
                logger.error("Errore agente: %s", e, exc_info=True)
                response_text = (
                    "Mi scuso, si e' verificato un errore. "
                    "Riprova tra qualche istante."
                )

        if self._guardrails_checker and response_text:
            output_allowed, block_message = self._guardrails_checker.check_output(response_text)
            if not output_allowed:
                logger.warning("OUTPUT BLOCCATO dal guardrail al turno #%d", turn_number)

                self._memory.rollback_last_turn()

                obs_handler.set_final_output(block_message)
                trace_dict = obs_handler.get_trace_dict()
                self._traces.append(trace_dict)

                return {
                    "response": block_message,
                    "blocked": True,
                    "block_reason": "output_guardrail",
                    "trace": trace_dict,
                    "turn": turn_number,
                }

        obs_handler.set_final_output(response_text)
        self._memory.add_assistant_message(response_text)

        trace_dict = obs_handler.get_trace_dict()
        self._traces.append(trace_dict)

        self._save_interaction_log(
            turn_number=turn_number,
            user_query=user_query,
            obs_handler=obs_handler,
            final_response=response_text,
            rewritten_query=rewritten_query,
        )

        if self._settings.observability.enable_verbose_callbacks:
            obs_handler.print_summary()

        return {
            "response": response_text,
            "blocked": False,
            "block_reason": None,
            "trace": trace_dict,
            "turn": turn_number,
        }

    def _rewrite_query(self, user_query: str) -> str:
        """Riscrive la query risolvendo coreferenze con il turno precedente.

        Args:
            user_query: Query originale dell'utente.

        Returns:
            Query riscritta, oppure quella originale se il rewriting non e' possibile.
        """
        if not self._query_optimizer:
            return user_query

        last_user, last_assistant = self._memory.get_last_completed_turn()

        if not last_user:
            logger.debug("Nessun turno precedente, skip rewriting.")
            return user_query

        try:
            rewritten = self._query_optimizer.rewrite(
                question=user_query,
                last_user_query=last_user,
                last_assistant_answer=last_assistant,
            )
            return rewritten
        except Exception as e:
            logger.warning("Errore durante il rewriting in chat(): %s", e)
            return user_query

    @staticmethod
    def _replace_query_in_messages(
        messages: list, original_query: str, rewritten_query: str
    ) -> None:
        """Sostituisce la query originale con quella riscritta nei messaggi.

        Args:
            messages: Lista di messaggi da aggiornare.
            original_query: Query originale da sostituire.
            rewritten_query: Query riscritta sostitutiva.
        """
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.get("role") == "user" and original_query in msg["content"]:
                messages[i]["content"] = msg["content"].replace(
                    original_query, rewritten_query, 1
                )
                logger.debug(
                    "Query sostituita nel messaggio #%d: '%s' -> '%s'",
                    i,
                    original_query[:50],
                    rewritten_query[:50],
                )
                break

    def _save_interaction_log(
        self,
        turn_number: int,
        user_query: str,
        obs_handler: RAGObservabilityHandler,
        final_response: str,
        rewritten_query: str = "",
    ) -> None:
        """Salva il log dell'interazione corrente su file.

        Args:
            turn_number: Numero del turno corrente.
            user_query: Query originale dell'utente.
            obs_handler: Handler di osservabilita con i dati della pipeline.
            final_response: Risposta finale dell'agente.
            rewritten_query: Query riscritta, se diversa dall'originale.
        """
        try:
            system_prompt = obs_handler.get_system_prompt()
            history_str = self._memory.get_history_summary()
            search_meta = get_last_search_meta()

            effective_rewritten = rewritten_query if rewritten_query != user_query else ""
            meta_rewritten = search_meta.get("rewritten_query", "")
            final_rewritten = effective_rewritten or meta_rewritten

            multi_queries = search_meta.get("multi_queries", [])
            tool_name = search_meta.get("tool_name", "(nessun tool invocato)")
            collection = search_meta.get("collection", "")
            metadata_filter = search_meta.get("metadata_filter", None)
            top_links = search_meta.get("top_links", [])

            if metadata_filter:
                meta_str = f"{collection} (filtro: {metadata_filter})"
            else:
                meta_str = collection if collection else "(nessun metadata)"

            self._interaction_logger.save_interaction(
                turn_number=turn_number,
                system_prompt=system_prompt,
                history=history_str,
                user_query=user_query,
                rewritten_query=final_rewritten,
                multi_queries=multi_queries,
                tool_name=tool_name,
                metadata_info=meta_str,
                top_links=top_links,
                final_response=final_response,
            )
        except Exception as e:
            logger.error("Errore salvataggio log interazione: %s", e)

    def get_all_traces(self) -> List[Dict[str, Any]]:
        """Restituisce tutte le trace accumulate nella sessione."""
        return self._traces

    def reset_memory(self) -> None:
        """Resetta la memoria e le trace della sessione."""
        self._memory.clear()
        self._traces.clear()
        logger.info("Sessione agente resettata")

    @property
    def memory(self) -> SmartConversationMemory:
        """Restituisce l'istanza di memoria conversazionale."""
        return self._memory

    @staticmethod
    def _extract_final_response(result: Dict[str, Any]) -> str:
        """Estrae la risposta finale dai messaggi prodotti dall'agente.

        Args:
            result: Dizionario risultato dell'invocazione dell'agente.

        Returns:
            Testo della risposta finale, oppure stringa vuota.
        """
        messages = result.get("messages", [])

        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                return msg.content
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                if not msg.get("tool_calls"):
                    return msg.get("content", "")

        for msg in reversed(messages):
            content = (
                getattr(msg, 'content', None)
                or (msg.get('content', '') if isinstance(msg, dict) else '')
            )
            if content:
                return content

        return ""


class RAGAgentFactory:
    """Factory per l'assemblaggio completo dell'agente RAG DIEM."""

    @staticmethod
    def create(
        retrieval_engine: RetrievalEngine,
        settings: Optional[AppSettings] = None,
        enable_scope_guardrail: bool = True,
        max_memory_turns: int = 10,
        log_output_dir: str = "logs/interactions",
        embedding_model: Optional[HuggingFaceEmbeddings] = None,
    ) -> "RAGAgent":
        """Assembla e restituisce un agente RAG completo.

        Args:
            retrieval_engine: Motore di retrieval inizializzato.
            settings: Configurazione dell'applicazione. Se None, viene caricata automaticamente.
            enable_scope_guardrail: Se True, abilita il guardrail di pertinenza tematica.
            max_memory_turns: Numero massimo di turni in memoria.
            log_output_dir: Directory di output per i log delle interazioni.
            embedding_model: Modello di embedding. Se None, viene usato quello del retrieval engine.

        Returns:
            Istanza di RAGAgent pronta all'uso.
        """
        settings = settings or load_settings()

        logger.info("=" * 60)
        logger.info("  RAGAgentFactory - Assemblaggio agente in corso...")
        logger.info("=" * 60)

        chat_model = create_chat_model(settings.llm)
        logger.info("    ChatModel: %s", settings.llm.model_name)

        set_retrieval_engine(retrieval_engine)
        tools = get_all_tools()
        logger.info("    Tools registrati: %s", [t.name for t in tools])

        for t in tools:
            if hasattr(t, 'args_schema') and t.args_schema is not None:
                schema_fields = list(t.args_schema.model_fields.keys())
                logger.info("      %s args_schema: %s", t.name, schema_fields)

        system_prompt = get_agent_system_prompt()
        logger.info("    System prompt caricato")

        memory = create_conversation_memory(
            max_turns=max_memory_turns,
            llm_for_summary=chat_model,
            embedding_model=embedding_model,
        )
        logger.info("    SmartConversationMemory: max_turns=%d", max_memory_turns)

        interaction_logger = create_interaction_log_handler(log_output_dir)
        logger.info("    InteractionLogHandler: %s", log_output_dir)

        query_optimizer = getattr(retrieval_engine, '_optimizer', None)
        if query_optimizer:
            logger.info("    QueryOptimizer estratto per rewriting a livello agente")
        else:
            logger.warning("    QueryOptimizer non disponibile - rewriting disabilitato")

        logger.info("    Configurazione GuardrailsChecker...")
        guardrails_checker = build_guardrails_checker(
            enable_pii=settings.guardrails.enable_pii_filter,
            enable_topical=enable_scope_guardrail,
            enable_injection=True,
            enable_toxicity=True,
            enable_hallucination=True,
            enable_code_guard=True,
        )

        if guardrails_checker:
            logger.info("    GuardrailsChecker configurato e attivo")
        else:
            logger.warning("    GuardrailsChecker non disponibile - guardrails disabilitati")

        from langchain.agents import create_agent

        agent_graph = create_agent(
            model=chat_model,
            tools=tools,
            system_prompt=system_prompt,
        )
        logger.info("    create_agent() - grafo agente compilato")

        agent = RAGAgent(
            agent_graph=agent_graph,
            memory=memory,
            settings=settings,
            interaction_logger=interaction_logger,
            query_optimizer=query_optimizer,
            guardrails_checker=guardrails_checker,
        )

        now = datetime.now()
        anno_acc = now.year if now.month >= 10 else now.year - 1
        logger.info("=" * 60)
        logger.info(" Agente RAG DIEM assemblato e pronto!")
        logger.info("    Temporale: iniezione SEMPRE attiva (A.A. default: %d/%d)", anno_acc, anno_acc + 1)
        logger.info("    Memoria: SmartConversationMemory")
        logger.info("    Tools: %d tool con Pydantic args_schema", len(tools))
        logger.info("    Rewriting: %s", "ATTIVO a livello agente" if query_optimizer else "DISABILITATO")
        logger.info("    Guardrails: CHECKER INTEGRATO IN chat()")
        if guardrails_checker:
            logger.info("      - Input check: PRIMA del rewriting (query originale)")
            logger.info("      - Output check: DOPO la risposta finale")
            logger.info("      - LLM check: Groq Llama 3.3 70B")
        else:
            logger.info("      - DISABILITATI (nessuna API key)")
        logger.info("    Fallback: search_all INTERNO ai tool")
        logger.info("    Chat loop: LINEARE (no retry, no reinvocazione)")
        logger.info("=" * 60)

        return agent


def bootstrap_agent(
    retrieval_engine: RetrievalEngine,
    settings: Optional[AppSettings] = None,
    embedding_model: Optional[HuggingFaceEmbeddings] = None,
    **kwargs,
) -> RAGAgent:
    """Shortcut per creare un agente RAG completo.

    Args:
        retrieval_engine: Motore di retrieval inizializzato.
        settings: Configurazione dell'applicazione.
        embedding_model: Modello di embedding opzionale.
        **kwargs: Argomenti aggiuntivi passati a RAGAgentFactory.create().

    Returns:
        Istanza di RAGAgent pronta all'uso.
    """
    return RAGAgentFactory.create(
        retrieval_engine=retrieval_engine,
        settings=settings,
        embedding_model=embedding_model,
        **kwargs,
    )