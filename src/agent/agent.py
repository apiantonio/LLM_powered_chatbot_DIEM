"""
Agente RAG DIEM — Orchestratore principale.

REFACTORING v3 — GUARDRAILS VIA MIDDLEWARE:
  I vecchi guardrails (ScopeGuardrail, InputSanitizer, OutputValidator)
  sono stati COMPLETAMENTE ELIMINATI e sostituiti con middleware LangChain
  inseriti direttamente nel grafo dell'agente tramite create_agent(middleware=[...]).

  Middleware attivi (in ordine):
    Prebuilt:
      - PIIMiddleware (codice fiscale, NO email/telefoni)
      - ModelCallLimitMiddleware (anti-loop)
      - ToolCallLimitMiddleware (limiti tool)
    Custom (before_model):
      - InjectionGuardMiddleware (prompt injection)
      - ToxicityFilterMiddleware (profanità/minacce)
      - TopicalGuardrailMiddleware (fuori contesto)
    Custom (after_model):
      - HallucinationGuardMiddleware (confabulazione)
      - CodeGenerationGuardMiddleware (blocco generazione codice)
      - OutputPIIGuardMiddleware (PII nell'output)

  Il flusso dell'agente è ora:
    User Query → Temporal Middleware (pre-agent)
               → Memory → Agent Graph (con middleware integrati)
               → Response

  I middleware gestiscono internamente:
    - Blocco injection PRIMA del model call
    - Blocco tossicità PRIMA del model call
    - Blocco off-topic PRIMA del model call
    - Validazione output DOPO il model call
    - Limiti su model/tool calls
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
from agent.guardrails import build_guardrail_middleware
from agent.tools import set_retrieval_engine, set_chat_history, get_all_tools, get_last_search_meta
from agent.llm_providers import create_chat_model
from retrieval.engine import RetrievalEngine

logger = logging.getLogger(__name__)


# ============================================================
# MIDDLEWARE: Iniezione Temporale Dinamica
# ============================================================

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
    """
    Facade principale per l'interazione con l'agente RAG DIEM.

    NOTA: I guardrails sono ora gestiti internamente dal grafo dell'agente
    tramite middleware LangChain. Questo Facade si occupa solo di:
      1. Pre-processing temporale
      2. Gestione memoria conversazionale
      3. Invocazione dell'agente
      4. Logging e osservabilità
    """

    _MAX_RETRY_NO_TOOL = 1

    def __init__(self, agent_graph, memory, settings,
                 interaction_logger: InteractionLogHandler):
        self._agent = agent_graph
        self._memory = memory
        self._settings = settings
        self._interaction_logger = interaction_logger
        self._traces = []
        self._retry_count = 0

    def chat(self, user_query: str) -> dict:
        """
        Punto di ingresso principale per l'interazione con l'agente.

        Flusso semplificato (i guardrails sono nei middleware):
          1. Middleware temporale: inietta data/ora se necessario
          2. Memory: costruzione messaggi con storico
          3. Chat history injection per tool
          4. Agent: invocazione (i middleware gestiscono guardrails)
          5. Logging + Memory update
        """
        print(f"USER QUERY: {user_query}")

        # --- STEP 1: Middleware Iniezione Temporale ---
        query_for_agent = inject_temporal_context(user_query)

        # --- STEP 2: Registra il turno e costruisci i messaggi ---
        turn_number = self._memory.add_user_message(user_query)
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

            # Retry se nessun tool invocato (e non è una meta-query)
            trace = obs_handler.get_trace()
            if (len(trace.tools_invoked) == 0
                    and not self._is_meta_query(user_query)
                    and self._retry_count < self._MAX_RETRY_NO_TOOL):

                self._retry_count += 1
                forced_messages = messages.copy()
                forced_messages[-1]["content"] = (
                    f"[Invoca un tool di ricerca per rispondere.] "
                    f"{user_query}"
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

        # --- STEP 3: Aggiorna handler e memoria ---
        obs_handler.set_final_output(response_text)
        self._memory.add_assistant_message(response_text)

        trace_dict = obs_handler.get_trace_dict()
        self._traces.append(trace_dict)

        # --- STEP 4: Salva il log dell'interazione ---
        self._save_interaction_log(
            turn_number=turn_number,
            user_query=user_query,
            obs_handler=obs_handler,
            final_response=response_text,
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
    """
    Factory per la costruzione dell'agente RAG completo.

    REFACTORING v3:
      I guardrails sono ora middleware LangChain passati a create_agent().
      Non esistono più oggetti ScopeGuardrail, InputSanitizer, OutputValidator
      separati — tutto è integrato nel grafo dell'agente.
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
        )

        logger.info("=" * 60)
        logger.info("🚀 Agente RAG DIEM assemblato e pronto!")
        logger.info("   📋 Prompt: v2 (ottimizzato 7B, no data statica)")
        logger.info("   ⏰ Temporale: middleware dinamico")
        logger.info("   🧠 Memoria: SmartConversationMemory")
        logger.info("   🔧 Tools: 4 tool con routing syllabus fix")
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