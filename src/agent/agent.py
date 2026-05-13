"""
Agente RAG DIEM — Orchestratore principale.

REFACTORING v2:
  - Middleware iniezione temporale: inietta data/ora SOLO se la query
    contiene riferimenti temporali impliciti o relativi.
  - System prompt snellito (vedi prompts.py v2)
  - Memory invariata (SmartConversationMemory)
  - Rimossa data/ora statica dal system prompt
"""

import re
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
# MIDDLEWARE: Iniezione Temporale Dinamica (Punto 5)
# ============================================================

# Pattern che indicano riferimenti temporali nella query
_TEMPORAL_PATTERNS = [
    r"\b(oggi|domani|ieri|dopodomani)\b",
    r"\b(luned[iì]|marted[iì]|mercoled[iì]|gioved[iì]|venerd[iì]|sabato|domenica)\s+(prossim[oa]|scors[oa])\b",
    r"\b(questa|prossima|scorsa)\s+(settimana|lezione)\b",
    r"\b(questo|prossimo|scorso)\s+(mese|semestre|anno|trimestre)\b",
    r"\b(lunedì|martedì|mercoledì|giovedì|venerdì|sabato|domenica)\b",
    r"\b(ora|adesso|attualmente|corrente|in corso)\b",
    r"\b(quando|a che ora|orario)\b",
    r"\b(prossim[oa]|scors[oa])\b",
]

_TEMPORAL_REGEX = [re.compile(p, re.IGNORECASE) for p in _TEMPORAL_PATTERNS]


def _has_temporal_reference(query: str) -> bool:
    """Controlla se la query contiene riferimenti temporali impliciti."""
    return any(p.search(query) for p in _TEMPORAL_REGEX)


def _get_temporal_context() -> str:
    """Genera la stringa di contesto temporale corrente."""
    now = datetime.now()
    giorni = ["lunedì", "martedì", "mercoledì", "giovedì",
              "venerdì", "sabato", "domenica"]
    mesi = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
            "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"]
    return (
        f"[Contesto temporale: {giorni[now.weekday()]} "
        f"{now.day} {mesi[now.month - 1]} {now.year}, "
        f"ore {now.strftime('%H:%M')}]"
    )


def inject_temporal_context(query: str) -> str:
    """
    Middleware pre-agente: inietta data/ora SOLO se la query
    contiene riferimenti temporali impliciti o relativi.
    """
    if _has_temporal_reference(query):
        temporal = _get_temporal_context()
        logger.info(f"Iniezione temporale: {temporal}")
        return f"{query}\n{temporal}"
    return query


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
          2. Middleware temporale: inietta data/ora se necessario
          3. Memory: costruzione messaggi con storico
          4. Chat history injection per tool (query rewriting contestuale)
          5. Agent: invocazione con callback di osservabilità
          6. Post-processing: output validation
          7. Logging + Memory update
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

        # --- STEP 2: Middleware Iniezione Temporale (Punto 5) ---
        # Inietta data/ora SOLO se la query ha riferimenti temporali
        query_for_agent = inject_temporal_context(sanitized_query)

        # --- STEP 3: Registra il turno e costruisci i messaggi ---
        turn_number = self._memory.add_user_message(sanitized_query)

        messages = self._memory.get_messages_for_agent(query_for_agent)

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

            # Retry se nessun tool invocato
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

        # La memoria salva la query ORIGINALE (senza contesto temporale)
        # per non inquinare lo storico con metadati di sistema
        self._memory.add_assistant_message(validated_response)

        trace_dict = obs_handler.get_trace_dict()
        self._traces.append(trace_dict)

        # --- STEP 7: Salva il log dell'interazione ---
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
        """Raccoglie i dati e salva un unico file di log per l'interazione."""
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
    """Factory per la costruzione dell'agente RAG completo."""

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

        # --- 3. System Prompt (snellito per 7B, senza data/ora) ---
        system_prompt = get_agent_system_prompt()
        logger.info("   ✅ System prompt caricato (v2, ottimizzato 7B)")

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

        # --- 5. SmartConversationMemory ---
        memory = create_conversation_memory(
            max_turns=max_memory_turns,
            llm_for_summary=chat_model,
            embedding_model=embedding_model,
        )
        logger.info(f"   ✅ SmartConversationMemory: max_turns={max_memory_turns}")

        # --- 6. Interaction Logger ---
        interaction_logger = create_interaction_log_handler(log_output_dir)
        logger.info(f"   ✅ InteractionLogHandler: {log_output_dir}")

        # --- 7. Assembla l'agente ---
        from langchain.agents import create_agent

        agent_graph = create_agent(
            model=chat_model,
            tools=tools,
            system_prompt=system_prompt,
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
        logger.info("   📋 Prompt: v2 (ottimizzato 7B, no data statica)")
        logger.info("   ⏰ Temporale: middleware dinamico")
        logger.info("   🧠 Memoria: SmartConversationMemory")
        logger.info("   🔧 Tools: 4 tool con routing syllabus fix")
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