"""Agente RAG DIEM principale.

Implementa il facade RAGAgent e la factory RAGAgentFactory per
l'assemblaggio e l'interazione con l'agente conversazionale.
"""

import re
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import AppSettings, load_settings
from agent.callbacks import (
    RAGObservabilityHandler,
    create_observability_handler,
    InteractionLogHandler,
    create_interaction_log_handler,
)
from agent.memory import SmartConversationMemory, create_conversation_memory
from agent.prompts import get_agent_system_prompt, get_meta_system_prompt
from agent.guardrails import GuardrailsChecker, build_guardrails_checker
from agent.tools import (
    set_retrieval_engine,
    set_chat_history,
    get_all_tools,
    get_last_search_meta,
)
from agent.llm_providers import create_chat_model
from retrieval.engine import RetrievalEngine, QueryOptimizer

logger = logging.getLogger(__name__)

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think_tags(text: str) -> str:
    """Rimuove i tag <think>...</think> dal testo e restituisce il contenuto pulito."""
    if not text:
        return ""
    cleaned = _THINK_TAG_RE.sub("", text).strip()
    return cleaned


def _get_temporal_system_message() -> dict:
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
    - check_meta() sulla query originale per identificare domande meta
      (saluti, ringraziamenti, ecc.) che non richiedono retrieval
      e non vengono salvate in memoria.
    - check_input() sulla query originale prima del rewriting.
    - check_output() sulla risposta finale dopo l'agente.
    Le interazioni bloccate dai guardrails non vengono salvate in memoria.
    """

    def __init__(
        self,
        agent_graph,
        chat_model,
        memory: SmartConversationMemory,
        settings: AppSettings,
        interaction_logger: InteractionLogHandler,
        query_optimizer: Optional[QueryOptimizer] = None,
        guardrails_checker: Optional[GuardrailsChecker] = None,
    ):
        self._agent = agent_graph
        self._memory = memory
        self._settings = settings
        self._interaction_logger = interaction_logger
        self._query_optimizer = query_optimizer
        self._guardrails_checker = guardrails_checker
        self._chat_model = chat_model
        self._traces: List[Dict[str, Any]] = []

    def chat(self, user_query: str) -> dict:
        logger.info("USER QUERY (pulita): %s", user_query)

        if self._guardrails_checker:
            input_allowed, block_message = self._guardrails_checker.check_input(user_query)
            if not input_allowed:
                logger.warning("INPUT BLOCCATO dal guardrail: '%s...'", user_query[:80])

                return {
                    "response": block_message,
                    "blocked": True,
                    "block_reason": "input_guardrail",
                    "is_meta": False,
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

        is_meta = False
        if self._guardrails_checker:
            is_meta = self._guardrails_checker.check_meta(user_query)

        if is_meta:
            return self._handle_meta_query(user_query)

        messages = self._memory.get_messages_for_agent(user_query)

        rewritten_query = user_query
        if self._query_optimizer:
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
                "is_meta": False,
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

            if not response_text:
                logger.warning(
                    "Content vuoto dal grafo agente al turno #%d. "
                    "Tentativo retry con LLM diretto...",
                    turn_number,
                )
                response_text = self._retry_with_direct_llm(result, user_query)

                if not response_text:
                    response_text = (
                        "Mi scuso, ho riscontrato un problema nell'elaborazione "
                        "della risposta. Per favore, riprova o riformula la domanda."
                    )
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
            elif "toolcalllimit" in error_str or "tool call limit" in error_str:
                # Con exit_behavior="continue" questa eccezione non dovrebbe
                # verificarsi (il middleware blocca le chiamate eccedenti con
                # un messaggio di errore e l'LLM continua a rispondere).
                # Mantenuto come safety net nel caso di cambi futuri.
                logger.warning(
                    "TOOL CALL LIMIT raggiunto (middleware). Query: '%s'",
                    user_query[:80],
                )
                response_text = (
                    "Mi scuso, non sono riuscito a trovare informazioni sufficienti "
                    "per rispondere alla tua domanda. Ti consiglio di consultare "
                    "il sito web del DIEM o contattare la segreteria."
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
                    "is_meta": False,
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
            "is_meta": False,
            "trace": trace_dict,
            "turn": turn_number,
        }

    def _handle_meta_query(self, user_query: str) -> dict:
        """Gestisce le domande meta SENZA il grafo agente.

        Le domande meta (saluti, ringraziamenti, domande sull'identita, ecc.)
        vengono processate con una chiamata LLM diretta, SENZA passare dal
        grafo agente e SENZA tool di ricerca. Il contatore dei turni non
        viene incrementato.

        Args:
            user_query: Query meta dell'utente.

        Returns:
            Dizionario con la risposta e i metadati dell'interazione meta.
        """
        logger.info("GESTIONE META QUERY (LLM diretto, no retrieval): '%s'", user_query)

        obs_handler = create_observability_handler(
            self._settings.observability,
            conversation_turn=self._memory.turn_count,
        )

        try:
            meta_system_prompt = get_meta_system_prompt()
            messages = [
                meta_system_prompt,
                HumanMessage(content=user_query),
            ]

            response = self._chat_model.invoke(messages)
            response_text = _strip_think_tags(response.content)

            if not response_text:
                response_text = (
                    "Ciao! Sono l'assistente virtuale del Dipartimento DIEM "
                    "dell'Universita degli Studi di Salerno. "
                    "Posso aiutarti con informazioni su corsi, docenti, esami, "
                    "regolamenti, laboratori e servizi universitari. "
                    "Come posso esserti utile?"
                )

            logger.info("RISPOSTA META (LLM diretto): %s", response_text)

        except Exception as e:
            logger.error("Errore LLM su meta query: %s", e, exc_info=True)
            response_text = (
                "Ciao! Sono l'assistente virtuale del Dipartimento DIEM "
                "dell'Universita degli Studi di Salerno. "
                "Posso aiutarti con informazioni su corsi, docenti, esami, "
                "regolamenti, laboratori e servizi universitari. "
                "Come posso esserti utile?"
            )

        obs_handler.set_final_output(response_text)

        trace_dict = obs_handler.get_trace_dict()
        trace_dict["tool_name"] = "(Meta Query - No Retrieval)"
        self._traces.append(trace_dict)

        logger.info(
            "META QUERY completata (turno NON salvato in memoria): '%s...'",
            user_query[:80],
        )

        return {
            "response": response_text,
            "blocked": False,
            "block_reason": None,
            "is_meta": True,
            "trace": trace_dict,
            "turn": self._memory.turn_count,
        }

    def _rewrite_query(self, user_query: str) -> str:
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
        return self._traces

    def reset_memory(self) -> None:
        self._memory.clear()
        self._traces.clear()
        logger.info("Sessione agente resettata")

    @property
    def memory(self) -> SmartConversationMemory:
        return self._memory

    @staticmethod
    def _extract_final_response(result: Dict[str, Any]) -> str:
        """Estrae la risposta finale dai messaggi prodotti dall'agente.

        Gestisce modelli thinking (Nemotron, Qwen, DeepSeek-R1) che possono:
        - Mettere la risposta in content con <think>...</think> tags
        - Mettere la risposta in additional_kwargs["reasoning_content"]
          lasciando content vuoto
        - Comportarsi normalmente con content pieno
        """
        messages = result.get("messages", [])

        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                content = _strip_think_tags(msg.content)
                if content:
                    return content

                reasoning = (
                    getattr(msg, "additional_kwargs", {})
                    .get("reasoning_content", "")
                )
                if reasoning:
                    logger.warning(
                        "Risposta trovata in reasoning_content (content era vuoto). "
                        "Considera di impostare reasoning=False in ChatOllama."
                    )
                    return _strip_think_tags(reasoning)

            if isinstance(msg, dict) and msg.get("role") == "assistant":
                if not msg.get("tool_calls"):
                    content = _strip_think_tags(msg.get("content", ""))
                    if content:
                        return content

        for msg in reversed(messages):
            content = ""
            if isinstance(msg, AIMessage):
                content = _strip_think_tags(msg.content)
                if not content:
                    content = _strip_think_tags(
                        getattr(msg, "additional_kwargs", {})
                        .get("reasoning_content", "")
                    )
            elif isinstance(msg, dict):
                content = _strip_think_tags(msg.get("content", ""))

            if content:
                return content

        logger.error(
            "Impossibile estrarre risposta dai messaggi dell'agente. "
            "Tutti i content sono vuoti."
        )
        return ""
    
    def _retry_with_direct_llm(self, result, user_query: str) -> str:
        """Tentativo di recovery quando _extract_final_response restituisce vuoto.

        Estrae i documenti dai ToolMessage nel result del grafo e chiede
        all'LLM una sintesi diretta, senza passare dal grafo agente.

        Args:
            result: Il risultato completo dell'invoke del grafo agente.
            user_query: La query originale dell'utente.

        Returns:
            Testo della risposta sintetizzata, oppure stringa vuota se il retry fallisce.
        """
        from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage

        messages = result.get("messages", [])

        tool_contents = []
        for msg in messages:
            if isinstance(msg, ToolMessage) and msg.content:
                content = _strip_think_tags(msg.content) if hasattr(msg, 'content') else ""
                if content and "Errore" not in content:
                    tool_contents.append(content)

        if not tool_contents:
            logger.warning("Retry LLM: nessun ToolMessage con contenuto trovato.")
            return ""

        combined_docs = "\n\n---\n\n".join(tool_contents)

        synthesis_prompt = (
            "Sei l'assistente virtuale del DIEM, Universita degli Studi di Salerno. "
            "Ti vengono forniti dei documenti recuperati dalla knowledge base. "
            "Rispondi alla domanda dell'utente basandoti ESCLUSIVAMENTE su questi documenti. "
            "Se i documenti non contengono l'informazione richiesta, dillo chiaramente. "
            "Rispondi nella stessa lingua usata dall'utente."
        )

        try:
            llm_messages = [
                SystemMessage(content=synthesis_prompt),
                HumanMessage(content=(
                    f"DOCUMENTI RECUPERATI:\n{combined_docs}\n\n"
                    f"DOMANDA DELL'UTENTE: {user_query}"
                )),
            ]

            response = self._chat_model.invoke(llm_messages)
            response_text = _strip_think_tags(response.content)

            if response_text:
                logger.info(
                    "RETRY LLM DIRETTO RIUSCITO: risposta di %d caratteri generata.",
                    len(response_text),
                )
            else:
                logger.warning("RETRY LLM DIRETTO: anche il retry ha prodotto content vuoto.")

            return response_text

        except Exception as e:
            logger.error("Errore nel retry LLM diretto: %s", e, exc_info=True)
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
        settings = settings or load_settings()

        logger.info("=" * 60)
        logger.info("  RAGAgentFactory - Assemblaggio agente in corso...")
        logger.info("=" * 60)

        chat_model = create_chat_model(settings.llm)

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
            enable_meta=True,
        )

        # --- ToolCallLimitMiddleware ---
        # Limita il numero di tool call per run a livello di grafo.
        # Con exit_behavior="continue", le chiamate tool eccedenti vengono
        # bloccate con un messaggio di errore, ma l'LLM continua a eseguire
        # e puo' sintetizzare una risposta dai documenti gia' recuperati
        # nelle chiamate precedenti.
        from langchain.agents import create_agent
        from langchain.agents.middleware import ToolCallLimitMiddleware

        tool_call_limit = settings.guardrails.max_tool_calls

        logger.info(
            "    ToolCallLimitMiddleware: run_limit=%d, exit_behavior='continue'",
            tool_call_limit,
        )

        agent_graph = create_agent(
            model=chat_model,
            tools=tools,
            system_prompt=system_prompt,
            middleware=[
                ToolCallLimitMiddleware(
                    run_limit=tool_call_limit,
                    exit_behavior="continue",
                ),
            ],
        )
        logger.info("    create_agent() - grafo agente compilato con middleware")

        agent = RAGAgent(
            agent_graph=agent_graph,
            chat_model=chat_model,
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
        logger.info("=" * 60)
        logger.info("    Temporale: iniezione SEMPRE attiva (A.A. default: %d/%d)", anno_acc, anno_acc + 1)
        logger.info("    Memoria: SmartConversationMemory (max_turns=%d)", max_memory_turns)
        logger.info("    Tools: %d tool con Pydantic args_schema", len(tools))
        logger.info(
            "    Anti-loop: ToolCallLimitMiddleware (run_limit=%d, exit='continue')",
            tool_call_limit,
        )

        if query_optimizer:
            logger.info("    Rewriting: ATTIVO")
        else:
            logger.info("    Rewriting: DISABILITATO (QueryOptimizer non disponibile)")

        if guardrails_checker:
            logger.info("    Guardrails: ATTIVI (Groq Llama 3.3 70B)")
            logger.info("      - Input check: PRIMA del rewriting (query originale)")
            logger.info("      - Meta check: DOPO input check, PRIMA del rewriting")
            logger.info("      - Output check: DOPO la risposta finale")
        else:
            logger.info("    Guardrails: DISABILITATI")

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
    return RAGAgentFactory.create(
        retrieval_engine=retrieval_engine,
        settings=settings,
        embedding_model=embedding_model,
        **kwargs,
    )