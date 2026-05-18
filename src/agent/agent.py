"""
Agente RAG DIEM — Orchestratore principale.

REFACTORING v3.4 — CHAT LOOP OTTIMIZZATO:

  CAMBIAMENTI RISPETTO A v3.3:

    1. RIMOSSO IL RETRY "NO TOOL":
       Il vecchio meccanismo _MAX_RETRY_NO_TOOL è stato eliminato.
       Se il modello risponde direttamente (dalla memoria, dallo storico,
       o da conoscenza contestuale) senza invocare tool, la risposta
       viene accettata così com'è. Nessuna reinvocazione forzata.

    2. RIMOSSO IL FALLBACK search_all A LIVELLO AGENTE:
       Il fallback su search_all è stato spostato DENTRO _search_collection()
       nel file tools/__init__.py. Quando un tool specifico non trova
       risultati, il fallback trasversale avviene IMMEDIATAMENTE a livello
       tool, PRIMA che il modello possa decidere di invocare altri tool.

       PRIMA (v3.3):
         search_persone → 0 docs → fallback signal al modello
         → modello chiama search_offerta_formativa → 0 docs
         → modello chiama search_dipartimento → 0 docs
         → modello risponde "non trovato"
         → agent.py intercetta → reinvoca con search_all
         TOTALE: 4 ricerche + reinvocazione agente

       ADESSO (v3.4):
         search_persone → 0 docs → fallback interno retrieve_from_all()
         → risultati (o "non trovato" definitivo) → modello risponde → FINE
         TOTALE: 1 ricerca + 1 fallback interno

    3. CHAT LOOP LINEARE:
       Il metodo chat() è ora completamente lineare:
       query → rewriting → cache check → agent invoke → risposta → FINE
       Nessun loop, nessun retry, nessuna reinvocazione condizionale.

  FLUSSO:
    User Query → Costruzione messaggi (con storico)
               → Query Rewriting (risoluzione coreferenze)
               → Corto circuito cache (se match esatto)
               → Salvataggio in memoria
               → Iniezione contesto temporale
               → Agent Graph (SINGOLA invocazione)
               → Response

  INVARIATI:
    - Contesto temporale iniettato come messaggio di sistema effimero
    - Guardrails via middleware LangChain
    - Pydantic tools con args_schema
    - SmartConversationMemory a due stadi
    - Query Rewriting a livello agente (v3.3)
"""

import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import AppSettings, load_settings
from agent.callbacks import (
    RAGObservabilityHandler,
    create_observability_handler,
    InteractionLogHandler,
    create_interaction_log_handler,
)
from agent.memory import SmartConversationMemory, create_conversation_memory
from agent.prompts import get_agent_system_prompt
from agent.guardrails import build_guardrail_middleware
from agent.tools import set_retrieval_engine, set_chat_history, get_all_tools, get_last_search_meta
from agent.llm_providers import create_chat_model
from retrieval.engine import RetrievalEngine, QueryOptimizer

logger = logging.getLogger(__name__)

# ============================================================
# MIDDLEWARE: Contesto Temporale (Isolato e Deterministico)
# ============================================================

def _get_temporal_system_message() -> dict:
    """
    Costruisce un dizionario (messaggio di sistema) con il contesto temporale.
    La direttiva usa il "Negative Prompting" per evitare l'uso forzato dell'anno.
    """
    now = datetime.now()
    giorni = ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"]
    mesi = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
            "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"]

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


# ============================================================
# FACADE: RAGAgent
# ============================================================

class RAGAgent:
    """
    Facade principale per l'interazione con l'agente RAG DIEM.

    NOTA v3.4: Il chat loop è ora completamente lineare.
      - Nessun retry "no tool" (risposta diretta = risposta valida)
      - Nessun fallback a livello agente (gestito internamente dai tool)
      - Una singola invocazione dell'agente per turno
    """

    def __init__(self, agent_graph, memory, settings,
                 interaction_logger: InteractionLogHandler,
                 query_optimizer: Optional[QueryOptimizer] = None):
        self._agent = agent_graph
        self._memory = memory
        self._settings = settings
        self._interaction_logger = interaction_logger
        self._query_optimizer = query_optimizer
        self._traces = []

    def chat(self, user_query: str) -> dict:
        """
        Punto di ingresso principale per l'interazione con l'agente.

        Flusso v3.4 (LINEARE — nessun loop/retry):
          1. Memory: costruzione messaggi con storico
          2. Query Rewriting: risoluzione coreferenze
          3. Cache: corto circuito se match esatto
          4. Memory: salvataggio query
          5. Iniezione contesto temporale (effimero)
          6. Chat history injection per tool
          7. Agent: SINGOLA invocazione
             - Se il modello risponde direttamente (senza tool) → OK
             - Se il modello invoca un tool e il tool non trova nulla →
               il fallback su search_all avviene DENTRO il tool, in modo
               trasparente. Il modello riceve i risultati e risponde.
          8. Logging + Memory update
        """

        print(f"USER QUERY (pulita): {user_query}")

        # --- STEP 1: Recupera lo storico usando la query PULITA ---
        messages = self._memory.get_messages_for_agent(user_query)

        # --- STEP 2: QUERY REWRITING (risoluzione coreferenze) ---
        rewritten_query = user_query
        if self._query_optimizer and not self._is_meta_query(user_query):
            rewritten_query = self._rewrite_query(user_query)
            if rewritten_query != user_query:
                logger.info(
                    f"Query rewritten in chat(): '{user_query}' → '{rewritten_query}'"
                )
                print(f"QUERY RISCRITTA: {user_query} → {rewritten_query}")
                # Sostituisci la query nel messaggio utente (ultimo messaggio)
                # preservando eventuali suffix (es. RETRIEVAL_REMINDER)
                self._replace_query_in_messages(messages, user_query, rewritten_query)

        # --- STEP 3: Corto Circuito via Cache (Match Esatto/Praticamente Uguale) ---
        cached_response = self._memory.find_exact_match(rewritten_query)

        if cached_response:
            print(f"⚡ RISPOSTA RECUPERATA IMMEDIATAMENTE DA CACHE PER: {rewritten_query}")

            dummy_trace = {
                "tool_name": "(Risposta da Cache)",
                "tools_invoked": [],
                "rewritten_query": "",
                "multi_queries": [],
                "collection": "",
                "metadata_filter": None,
                "top_links": []
            }

            return {
                "response": cached_response,
                "blocked": False,
                "block_reason": None,
                "trace": dummy_trace,
                "turn": self._memory.turn_count,
            }

        # --- STEP 4: Salva in memoria la query ORIGINALE e PULITA ---
        turn_number = self._memory.add_user_message(rewritten_query)

        # --- STEP 5: Iniezione Effimera del Contesto Temporale ---
        temporal_msg = _get_temporal_system_message()
        if messages and messages[-1]["role"] == "user":
            messages.insert(-1, temporal_msg)
        else:
            messages.append(temporal_msg)

        print(f"MESSAGGI INVIATI ALL'AGENTE: {messages}")

        # --- STEP 6: Inietta la chat history nei tool ---
        set_chat_history(self._memory.get_langchain_history())

        obs_handler = create_observability_handler(
            self._settings.observability,
            conversation_turn=turn_number,
        )

        try:
            # =============================================================
            # STEP 7: SINGOLA INVOCAZIONE DELL'AGENTE
            #
            # v3.4: Nessun retry, nessun fallback a questo livello.
            #
            # - Se il modello risponde senza invocare tool (es. dalla
            #   history, dalla memoria, o per una meta-query): la risposta
            #   è valida e viene restituita direttamente.
            #
            # - Se il modello invoca un tool specifico e quel tool non
            #   trova risultati: il fallback su search_all avviene
            #   INTERNAMENTE al tool (in _search_collection), in modo
            #   completamente trasparente per l'agente. Il modello riceve
            #   i risultati del fallback (o il messaggio definitivo di
            #   "non trovato") e genera la risposta finale.
            #
            # In entrambi i casi: UNA SOLA invocazione, nessun loop.
            # =============================================================
            result = self._agent.invoke(
                {"messages": messages},
                config={
                    "callbacks": [obs_handler],
                    "recursion_limit": self._settings.guardrails.max_agent_iterations,
                },
            )
            response_text = self._extract_final_response(result)

            print(f"TESTO DI RISPOSTA AGENTE AL TURNO {turn_number}: {response_text}")

        except Exception as e:
            error_str = str(e).lower()
            error_type = type(e).__name__

            if ("recursion" in error_str or "recursion" in error_type.lower()
                    or "iteration" in error_str):
                logger.error(
                    f"🔄 LOOP RILEVATO — Agente terminato forzatamente. "
                    f"Query: '{user_query[:80]}'"
                )
                response_text = (
                    "Mi scuso, ho riscontrato difficoltà nell'elaborare la tua "
                    "domanda. Prova a riformularla in modo più specifico."
                )
            else:
                logger.error(f"Errore agente: {e}", exc_info=True)
                response_text = (
                    "Mi scuso, si è verificato un errore. "
                    "Riprova tra qualche istante."
                )

        # --- STEP 8: Aggiorna handler e memoria ---
        obs_handler.set_final_output(response_text)
        self._memory.add_assistant_message(response_text)

        trace_dict = obs_handler.get_trace_dict()
        self._traces.append(trace_dict)

        # --- STEP 9: Salva il log dell'interazione ---
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

    # ==============================
    # QUERY REWRITING (v3.3)
    # ==============================

    def _rewrite_query(self, user_query: str) -> str:
        """
        Riscrive la query risolvendo coreferenze dall'ultimo turno.

        Estrae l'ultimo turno completato dalla memoria e lo passa
        al QueryOptimizer per risolvere pronomi e riferimenti impliciti.

        Returns:
            La query riscritta, oppure la query originale se:
            - Non c'è storico (primo turno)
            - Il QueryOptimizer non è disponibile
            - Si verifica un errore
        """
        if not self._query_optimizer:
            return user_query

        # Estrai l'ultimo turno completato dalla memoria
        last_user, last_assistant = self._memory.get_last_completed_turn()

        if not last_user:
            # Primo turno: nessuna coreferenza possibile
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
            logger.warning(f"Errore durante il rewriting in chat(): {e}")
            return user_query

    @staticmethod
    def _replace_query_in_messages(
        messages: list, original_query: str, rewritten_query: str
    ) -> None:
        """
        Sostituisce la query originale con quella riscritta nell'ultimo
        messaggio utente della lista messaggi.

        Preserva eventuali suffissi aggiunti da get_messages_for_agent()
        (es. RETRIEVAL_REMINDER).
        """
        # Cerca l'ultimo messaggio "user" e sostituisci la query
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.get("role") == "user" and original_query in msg["content"]:
                messages[i]["content"] = msg["content"].replace(
                    original_query, rewritten_query, 1
                )
                logger.debug(
                    f"Query sostituita nel messaggio #{i}: "
                    f"'{original_query[:50]}' → '{rewritten_query[:50]}'"
                )
                break

    # ==============================
    # LOGGING & HELPERS
    # ==============================

    def _save_interaction_log(
        self,
        turn_number: int,
        user_query: str,
        obs_handler: RAGObservabilityHandler,
        final_response: str,
        rewritten_query: str = "",
    ) -> None:
        """Raccoglie i dati e salva un unico file di log per l'interazione."""
        try:
            system_prompt = obs_handler.get_system_prompt()
            history_str = self._memory.get_history_summary()
            search_meta = get_last_search_meta()

            # v3.3: se il rewriting è avvenuto a livello agente, usiamo quello
            effective_rewritten = rewritten_query if rewritten_query != user_query else ""
            meta_rewritten = search_meta.get("rewritten_query", "")
            # Priorità: rewriting dell'agente > rewriting del tool (che ora non c'è più)
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
            logger.error(f"Errore salvataggio log interazione: {e}")

    def get_all_traces(self) -> List[Dict[str, Any]]:
        """Restituisce tutte le trace della sessione."""
        return self._traces

    def reset_memory(self) -> None:
        """Resetta la memoria conversazionale."""
        self._memory.clear()
        self._traces.clear()
        logger.info("Sessione agente resettata")

    @property
    def memory(self) -> SmartConversationMemory:
        """Restituisce l'istanza di SmartConversationMemory."""
        return self._memory

    # ==============================
    # PRIVATE HELPERS
    # ==============================

    @staticmethod
    def _extract_final_response(result: Dict[str, Any]) -> str:
        messages = result.get("messages", [])

        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                return msg.content
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                if not msg.get("tool_calls"):
                    return msg.get("content", "")

        for msg in reversed(messages):
            content = getattr(msg, 'content', None) or msg.get('content', '') if isinstance(msg, dict) else ''
            if content:
                return content

        return ""

    @staticmethod
    def _is_meta_query(query: str) -> bool:
        from agent.memory import _is_meta_query
        return _is_meta_query(query)


# ============================================================
# FACTORY: RAGAgentFactory
# ============================================================

class RAGAgentFactory:
    """
    Factory per la costruzione dell'agente RAG completo.

    REFACTORING v3.4:
      - Chat loop lineare (nessun retry, nessun fallback a livello agente)
      - Fallback search_all gestito internamente dai tool
    """

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
        logger.info("🏗️  RAGAgentFactory — Assemblaggio agente in corso...")
        logger.info("=" * 60)

        # --- 1. Chat Model ---
        chat_model = create_chat_model(settings.llm)
        logger.info(f"   ✅ ChatModel: {settings.llm.model_name}")

        # --- 2. Inietta il RetrievalEngine nei tools ---
        set_retrieval_engine(retrieval_engine)
        tools = get_all_tools()
        logger.info(f"   ✅ Tools registrati: {[t.name for t in tools]}")

        for t in tools:
            if hasattr(t, 'args_schema') and t.args_schema is not None:
                schema_fields = list(t.args_schema.model_fields.keys())
                logger.info(f"      📐 {t.name} args_schema: {schema_fields}")

        # --- 3. System Prompt ---
        system_prompt = get_agent_system_prompt()
        logger.info("   ✅ System prompt caricato (v2, ottimizzato 7B)")

        # --- 4. SmartConversationMemory ---
        memory = create_conversation_memory(
            max_turns=max_memory_turns,
            llm_for_summary=chat_model,
            embedding_model=embedding_model,
        )
        logger.info(f"   ✅ SmartConversationMemory: max_turns={max_memory_turns}")

        # --- 5. Interaction Logger ---
        interaction_logger = create_interaction_log_handler(log_output_dir)
        logger.info(f"   ✅ InteractionLogHandler: {log_output_dir}")

        # --- 5.5. QueryOptimizer per rewriting a livello agente ---
        query_optimizer = getattr(retrieval_engine, '_optimizer', None)
        if query_optimizer:
            logger.info("   ✅ QueryOptimizer estratto per rewriting a livello agente")
        else:
            logger.warning("   ⚠️ QueryOptimizer non disponibile — rewriting disabilitato")

        # --- 6. GUARDRAILS VIA MIDDLEWARE ---
        logger.info("   📋 Assemblaggio middleware guardrails...")
        guardrail_middleware = build_guardrail_middleware(
            classifier_llm=chat_model if enable_scope_guardrail else None,
            enable_pii=settings.guardrails.enable_pii_filter,
            enable_topical=enable_scope_guardrail,
            enable_injection=True,
            enable_toxicity=True,
            enable_hallucination=True,
            enable_code_guard=True,
            max_model_calls_per_run=settings.guardrails.max_agent_iterations,
            max_tool_calls_per_run=12,
        )

        # --- 7. Assembla l'agente CON middleware ---
        from langchain.agents import create_agent

        agent_graph = create_agent(
            model=chat_model,
            tools=tools,
            system_prompt=system_prompt,
            middleware=guardrail_middleware,
        )
        logger.info("   ✅ create_agent() — grafo agente compilato CON middleware")

        # --- 8. Wrappa nel Facade ---
        agent = RAGAgent(
            agent_graph=agent_graph,
            memory=memory,
            settings=settings,
            interaction_logger=interaction_logger,
            query_optimizer=query_optimizer,
        )

        # --- Log di riepilogo ---
        now = datetime.now()
        anno_acc = now.year if now.month >= 10 else now.year - 1
        logger.info("=" * 60)
        logger.info("🚀 Agente RAG DIEM assemblato e pronto!")
        logger.info("   📋 Prompt: v2 (ottimizzato 7B, no data statica)")
        logger.info(f"   ⏰ Temporale: iniezione SEMPRE attiva (A.A. default: {anno_acc}/{anno_acc+1})")
        logger.info("   🧠 Memoria: SmartConversationMemory")
        logger.info("   🔧 Tools: 4 tool con Pydantic args_schema")
        logger.info(f"   ✏️ Rewriting: {'ATTIVO a livello agente' if query_optimizer else 'DISABILITATO'}")
        logger.info("   🛡️ Guardrails: MIDDLEWARE-BASED")
        logger.info(f"      - {len(guardrail_middleware)} middleware attivi")
        logger.info("      - Injection Guard (before_model)")
        logger.info("      - Toxicity Filter (before_model)")
        logger.info("      - Topical Guard (before_model, dual-layer)")
        logger.info("      - Hallucination Guard (after_model)")
        logger.info("      - Code Generation Guard (after_model)")
        logger.info("      - PII Guard (codice fiscale, NO email/tel)")
        logger.info("      - Model Call Limit (anti-loop)")
        logger.info("      - Tool Call Limit")
        logger.info("   🔄 Fallback: search_all INTERNO ai tool (v3.4)")
        logger.info("   ⚡ Chat loop: LINEARE (no retry, no reinvocazione)")
        logger.info("=" * 60)

        return agent


# ============================================================
# CONVENIENCE
# ============================================================

def bootstrap_agent(
    retrieval_engine: RetrievalEngine,
    settings: Optional[AppSettings] = None,
    embedding_model: Optional[HuggingFaceEmbeddings] = None,
    **kwargs,
) -> RAGAgent:
    """Shortcut per creare un agente RAG completo."""
    return RAGAgentFactory.create(
        retrieval_engine=retrieval_engine,
        settings=settings,
        embedding_model=embedding_model,
        **kwargs,
    )