"""
Agente RAG DIEM — Orchestratore principale.

REFACTORING per 3 Vector Store (audit_fattibilita_metadati.md):
  - Import corretto: create_agent da langchain.agents (LangChain v1)
    (DIVIETO: create_tool_calling_agent, AgentExecutor, langgraph diretto)
  - Tool aggiornati: search_persone, search_offerta_formativa,
    search_dipartimento, search_all
  - EasyCourse ESCLUSO

AGGIORNAMENTO — Integrazione SmartConversationMemory e Prompt XML:
  - SmartConversationMemory sostituisce ConversationMemory.
    La nuova memoria richiede un LLM per la summarization (Stadio 2)
    e un HuggingFaceEmbeddings per il filtraggio per similarità coseno
    (Stadio 1). Entrambi vengono passati dalla Factory.
  - Il system prompt in formato XML (con direttive anti-compressione,
    regole di routing e fallback strategy) viene caricato da prompts.py
    e passato a create_agent come SystemMessage.
  - I tool con descrizioni XML e fallback espliciti vengono bindati
    all'agente tramite create_agent(tools=...).

Architettura:
  create_agent(model, tools, system_prompt=system_prompt)
      ↓
  Loop ReAct: LLM → decide tool → esegue tool → LLM → ... → risposta
      ↓
  Esterno: SmartConversationMemory gestisce lo storico multi-turno
           (filtro similarità coseno + summarization con token budget)
           RAGObservabilityHandler traccia ogni fase
           InteractionLogHandler salva 1 file per interazione
"""

import logging
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
# ── ALLINEAMENTO: import aggiornato a SmartConversationMemory ──
# La vecchia ConversationMemory è stata sostituita dalla
# SmartConversationMemory a due stadi (filtro similarità + summarization).
from agent.memory import SmartConversationMemory, create_conversation_memory
from agent.prompts import get_agent_system_prompt
from agent.guardrails import (
    ScopeGuardrail,
    InputSanitizer,
    OutputValidator,
    ScopeViolationError,
    InputInjectionError,
)
from agent.tools import set_retrieval_engine, set_chat_history, get_all_tools, get_last_search_meta
from agent.llm_providers import create_chat_model
from retrieval.engine import RetrievalEngine

logger = logging.getLogger(__name__)


# ============================================================
# FACADE: RAGAgent
# ============================================================

class RAGAgent:

    _MAX_RETRY_NO_TOOL = 1

    def __init__(self, agent_graph, memory, scope_guardrail,
                 input_sanitizer, output_validator, settings,
                 interaction_logger: InteractionLogHandler):
        self._agent = agent_graph
        self._memory = memory
        self._scope_guardrail = scope_guardrail
        self._input_sanitizer = input_sanitizer
        self._output_validator = output_validator
        self._settings = settings
        self._interaction_logger = interaction_logger
        self._traces = []
        self._retry_count = 0

    def chat(self, user_query: str) -> dict:
        """
        Punto di ingresso principale per l'interazione con l'agente.

        Flusso:
          1. Pre-processing: guardrails (scope + sanitization)
          2. Memory: costruzione messaggi con storico
             (SmartConversationMemory applica filtro similarità + summarization)
          3. Chat history injection per tool (query rewriting contestuale)
          4. Agent: invocazione con callback di osservabilità
             (la query utente viene passata INTEGRA — direttiva anti-compressione)
          5. Post-processing: output validation
          6. Logging: salvataggio interazione su file unico
          7. Memory: aggiornamento storico (con embedding del turno)
        """
        # --- STEP 1: Pre-processing Guardrails ---

        # 1a. Input Sanitization
        sanitizer_ctx: Dict[str, Any] = {}

        print(f"USER QUERY: {user_query}")

        passed, sanitized_query = self._input_sanitizer.check(user_query, sanitizer_ctx)

        print(f"QUERY SANITIZZATA: {sanitized_query}")

        if not passed:
            logger.warning(f"Input bloccato da InputSanitizer: {user_query[:80]}")
            return {
                "response": sanitized_query,
                "blocked": True,
                "block_reason": "prompt_injection",
                "trace": {},
                "turn": self._memory.turn_count + 1,
            }

        # 1b. Scope Check
        if self._scope_guardrail:
            passed, scope_result = self._scope_guardrail.check(sanitized_query)
            if not passed:
                logger.info(f"Query OOD bloccata: {user_query[:80]}")
                return {
                    "response": scope_result,
                    "blocked": True,
                    "block_reason": "out_of_scope",
                    "trace": {},
                    "turn": self._memory.turn_count + 1,
                }

        # --- STEP 2: Registra il turno e costruisci i messaggi ---
        # SmartConversationMemory.add_user_message registra la query e avanza il contatore.
        turn_number = self._memory.add_user_message(sanitized_query)

        # SmartConversationMemory.get_messages_for_agent applica i due stadi:
        #   Stadio 1: filtraggio per similarità coseno (scarta turni irrilevanti)
        #   Stadio 2: summarization con token budget (riassume turni vecchi se necessario)
        # La query utente viene passata INTEGRA (direttiva anti-compressione
        # rispettata: il RETRIEVAL_REMINDER viene aggiunto IN CODA, senza
        # modificare la query originale).
        messages = self._memory.get_messages_for_agent(sanitized_query)

        print(f"MESSAGGIO DALLA MEMORIA: {messages}")

        # Inietta la chat history nei tool per il query rewriting
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

            print(f"TESTO DI RISPOSTA AGENTE AL TURNO {turn_number}: {response_text}")

            # Retry se nessun tool invocato (invariato)
            trace = obs_handler.get_trace()
            if (len(trace.tools_invoked) == 0
                    and not self._is_meta_query(sanitized_query)
                    and self._retry_count < self._MAX_RETRY_NO_TOOL):

                self._retry_count += 1
                forced_messages = messages.copy()
                forced_messages[-1]["content"] = (
                    f"[Invoca un tool di ricerca per rispondere.] "
                    f"{sanitized_query}"
                )
                obs_handler_retry = create_observability_handler(
                    self._settings.observability,
                    conversation_turn=turn_number,
                )
                result = self._agent.invoke(
                    {"messages": forced_messages},
                    config={
                        "callbacks": [obs_handler_retry],
                        "recursion_limit": self._settings.guardrails.max_agent_iterations,
                    },
                )
                response_text = self._extract_final_response(result)
                obs_handler = obs_handler_retry

            self._retry_count = 0

        except Exception as e:
            error_str = str(e).lower()
            error_type = type(e).__name__

            if ("recursion" in error_str or "recursion" in error_type.lower()
                    or "iteration" in error_str):
                logger.error(
                    f"🔄 LOOP RILEVATO — Agente terminato forzatamente. "
                    f"Query: '{sanitized_query[:80]}'"
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

        # --- STEP 5: Post-processing (Output Validation) ---
        passed, validated_response = self._output_validator.check(response_text)
        if not passed:
            validated_response = (
                "Mi dispiace, non sono riuscito a generare una risposta valida. "
                "Riprova con una domanda più specifica."
            )

        # --- STEP 6: Aggiorna handler e memoria ---
        obs_handler.set_final_output(validated_response)

        # SmartConversationMemory.add_assistant_message completa il turno e
        # calcola l'embedding del turno (user + assistant) per il filtraggio
        # per similarità coseno dei turni futuri.
        self._memory.add_assistant_message(validated_response)

        trace_dict = obs_handler.get_trace_dict()
        self._traces.append(trace_dict)

        # --- STEP 7: Salva il log dell'interazione (1 file unico) ---
        self._save_interaction_log(
            turn_number=turn_number,
            user_query=sanitized_query,
            obs_handler=obs_handler,
            final_response=validated_response,
        )

        if self._settings.observability.enable_verbose_callbacks:
            obs_handler.print_summary()

        return {
            "response": validated_response,
            "blocked": False,
            "block_reason": None,
            "trace": trace_dict,
            "turn": turn_number,
        }

    def _save_interaction_log(
        self,
        turn_number: int,
        user_query: str,
        obs_handler: RAGObservabilityHandler,
        final_response: str,
    ) -> None:
        """
        Raccoglie i dati e salva un unico file di log per l'interazione.
        """
        try:
            system_prompt = obs_handler.get_system_prompt()
            history_str = self._memory.get_history_summary()
            search_meta = get_last_search_meta()

            rewritten_query = search_meta.get("rewritten_query", "")
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
                rewritten_query=rewritten_query,
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
        """Resetta la memoria conversazionale (SmartConversationMemory)."""
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

    REFACTORING:
      - Import CORRETTO: create_agent da langchain.agents (LangChain v1)
      - DIVIETO rispettato: NO create_tool_calling_agent, NO AgentExecutor,
        NO langgraph (import diretto), NO create_react_agent (deprecato)

    AGGIORNAMENTO — SmartConversationMemory:
      La Factory ora istanzia la SmartConversationMemory passandole:
        - llm_for_summary: il ChatModel usato anche dall'agente (o un modello
          dedicato), necessario per la summarization dei turni (Stadio 2).
        - embedding_model: HuggingFaceEmbeddings per il calcolo della
          similarità coseno tra la query corrente e i turni memorizzati
          (Stadio 1 — filtraggio per similarità).
      Questi vengono passati a create_conversation_memory().
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
        # I tool aggiornati hanno descrizioni XML con direttiva anti-compressione
        # e segnali di fallback espliciti per guidare l'agente verso tool alternativi.
        set_retrieval_engine(retrieval_engine)
        tools = get_all_tools()
        logger.info(f"   ✅ Tools registrati (con descrizioni XML e fallback): {[t.name for t in tools]}")

        # --- 3. System Prompt XML ---
        # Il system prompt è in formato XML con sezioni:
        #   <query_passthrough> — direttiva anti-compressione
        #   <tool_usage>        — regole di routing esplicite
        #   <fallback_strategy> — strategia di fallback obbligatoria
        #   <lingua>            — enforcement lingua italiana
        system_prompt = get_agent_system_prompt()
        logger.info("   ✅ System prompt XML caricato (anti-compressione, routing, fallback)")

        # --- 4. Guardrails ---
        scope_guardrail = None
        if enable_scope_guardrail:
            scope_guardrail = ScopeGuardrail(chat_model)
            logger.info("   ✅ ScopeGuardrail attivato")

        input_sanitizer = InputSanitizer()
        logger.info("   ✅ InputSanitizer attivato")

        output_validator = OutputValidator(
            enable_pii_filter=settings.guardrails.enable_pii_filter
        )
        logger.info("   ✅ OutputValidator attivato")

        # --- 5. SmartConversationMemory (ALLINEAMENTO) ---
        # Inizializzazione della nuova SmartConversationMemory a due stadi:
        #   Stadio 1 — Filtraggio per similarità coseno:
        #     Usa l'embedding_model (HuggingFaceEmbeddings) per calcolare
        #     la similarità tra la query corrente e i turni memorizzati.
        #     I turni sotto soglia (default 0.3) vengono scartati.
        #   Stadio 2 — Summarization con token budget:
        #     Usa il chat_model come LLM per la ConversationSummaryBufferMemory
        #     di LangChain, che riassume i turni più vecchi quando il totale
        #     supera max_token_limit.
        #
        # Se embedding_model non è fornito esternamente, create_conversation_memory
        # ne crea uno internamente dalle settings (modello di embedding configurato).
        memory = create_conversation_memory(
            max_turns=max_memory_turns,
            llm_for_summary=chat_model,
            embedding_model=embedding_model,
        )
        logger.info(
            f"   ✅ SmartConversationMemory: max_turns={max_memory_turns}, "
            f"llm_for_summary={settings.llm.model_name}, "
            f"embedding_model={'fornito esternamente' if embedding_model else 'creato da settings'}"
        )

        # --- 6. Interaction Logger ---
        interaction_logger = create_interaction_log_handler(log_output_dir)
        logger.info(f"   ✅ InteractionLogHandler: {log_output_dir}")

        # --- 7. Assembla l'agente con create_agent (LangChain v1) ---
        # VINCOLO ASSOLUTO: usare ESCLUSIVAMENTE create_agent
        # da langchain.agents (LangChain v1).
        # DIVIETO: create_tool_calling_agent, AgentExecutor, langgraph
        # diretto, create_react_agent (deprecato in LangChain v1).
        #
        # create_agent restituisce un CompiledStateGraph che gestisce
        # il loop ReAct internamente:
        #   input → LLM → tool call → osservazione → LLM → ... → risposta
        #
        # Accetta {"messages": [...]} e restituisce {"messages": [...]}.
        # Il system_prompt (SystemMessage XML con anti-compressione, routing
        # e fallback) viene iniettato automaticamente come SystemMessage
        # all'inizio della lista messaggi ad ogni chiamata al modello.
        #
        # I tool bindati hanno descrizioni XML con:
        #   - Direttiva anti-compressione: "Passa la query INTEGRA"
        #   - Segnali di fallback: indicano quale tool alternativo provare
        from langchain.agents import create_agent

        agent_graph = create_agent(
            model=chat_model,
            tools=tools,
            system_prompt=system_prompt,
        )
        logger.info("   ✅ create_agent() — grafo agente compilato con prompt XML e tool aggiornati")

        # --- 8. Wrappa nel Facade ---
        agent = RAGAgent(
            agent_graph=agent_graph,
            memory=memory,
            scope_guardrail=scope_guardrail,
            input_sanitizer=input_sanitizer,
            output_validator=output_validator,
            settings=settings,
            interaction_logger=interaction_logger,
        )

        logger.info("=" * 60)
        logger.info("🚀 Agente RAG DIEM assemblato e pronto!")
        logger.info("   📋 Prompt: XML (anti-compressione + routing + fallback)")
        logger.info("   🧠 Memoria: SmartConversationMemory (similarità + summarization)")
        logger.info("   🔧 Tools: 4 tool con descrizioni XML e fallback espliciti")
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