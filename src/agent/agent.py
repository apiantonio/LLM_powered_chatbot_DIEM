"""
Agente RAG DIEM — Orchestratore principale.

REFACTORING APPLICATO:
  1. ELIMINATO PromptDumpHandler (generava decine di file).
  2. NUOVO InteractionLogHandler: genera 1 unico file per interazione.
  3. Raccolta dati da _last_search_meta (tool) per popolare il log:
     rewritten_query, multi_queries, tool_name, metadata, top_links.
  4. Il system prompt viene catturato dall'obs_handler per il log.

Architettura:
  create_agent(model, tools, system_prompt)
      ↓
  Loop ReAct: LLM → decide tool → esegue tool → LLM → ... → risposta
      ↓
  Esterno: ConversationMemory gestisce lo storico multi-turno
           RAGObservabilityHandler traccia ogni fase
           InteractionLogHandler salva 1 file per interazione
"""

import logging
from typing import Optional, List, Dict, Any

from langchain_core.messages import AIMessage, BaseMessage

from config.settings import AppSettings, load_settings
from agent.callbacks import (
    RAGObservabilityHandler,
    create_observability_handler,
    InteractionLogHandler,
    create_interaction_log_handler,
)
from agent.memory import ConversationMemory, create_conversation_memory
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
          3. Chat history injection per tool
          4. Agent: invocazione con callback di osservabilità
          5. Post-processing: output validation
          6. Logging: salvataggio interazione su file unico
          7. Memory: aggiornamento storico
        """
        # --- STEP 1: Pre-processing Guardrails ---

        # 1a. Input Sanitization
        sanitizer_ctx: Dict[str, Any] = {}
        passed, sanitized_query = self._input_sanitizer.check(user_query, sanitizer_ctx)
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
        turn_number = self._memory.add_user_message(sanitized_query)
        messages = self._memory.get_messages_for_agent(sanitized_query)

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
            # System prompt catturato dall'obs_handler
            system_prompt = obs_handler.get_system_prompt()

            # History formattata
            history_str = self._memory.get_history_summary()

            # Dati dal tool (rewritten_query, multiqueries, tool, links)
            search_meta = get_last_search_meta()

            rewritten_query = search_meta.get("rewritten_query", "")
            multi_queries = search_meta.get("multi_queries", [])
            tool_name = search_meta.get("tool_name", "(nessun tool invocato)")
            collection = search_meta.get("collection", "")
            metadata_filter = search_meta.get("metadata_filter", None)
            top_links = search_meta.get("top_links", [])

            # Formatta metadata info
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
        """Resetta la memoria conversazionale."""
        self._memory.clear()
        self._traces.clear()
        logger.info("Sessione agente resettata")

    @property
    def memory(self) -> ConversationMemory:
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
    Ora include anche InteractionLogHandler.
    """

    @staticmethod
    def create(
        retrieval_engine: RetrievalEngine,
        settings: Optional[AppSettings] = None,
        enable_scope_guardrail: bool = True,
        max_memory_turns: int = 10,
        log_output_dir: str = "logs/interactions",
    ) -> RAGAgent:
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

        # --- 3. System Prompt ---
        system_prompt = get_agent_system_prompt()
        logger.info("   ✅ System prompt caricato (con enforcement lingua italiana)")

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

        # --- 5. Memory ---
        memory = create_conversation_memory(max_turns=max_memory_turns)
        logger.info(f"   ✅ ConversationMemory: max_turns={max_memory_turns}")

        # --- 6. Interaction Logger (NUOVO) ---
        interaction_logger = create_interaction_log_handler(log_output_dir)
        logger.info(f"   ✅ InteractionLogHandler: {log_output_dir}")

        # --- 7. Assembla l'agente ---
        from langchain.agents import create_agent

        agent_graph = create_agent(
            model=chat_model,
            tools=tools,
            system_prompt=system_prompt,
            name="diem_rag_agent",
        )
        logger.info("   ✅ create_agent() — grafo agente compilato")

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
        logger.info("=" * 60)

        return agent


# ============================================================
# CONVENIENCE
# ============================================================

def bootstrap_agent(
    retrieval_engine: RetrievalEngine,
    settings: Optional[AppSettings] = None,
    **kwargs,
) -> RAGAgent:
    """Shortcut per creare un agente RAG completo."""
    return RAGAgentFactory.create(
        retrieval_engine=retrieval_engine,
        settings=settings,
        **kwargs,
    )