"""
Agente RAG DIEM — Orchestratore principale.

REFACTORING v3.1 — RISOLUZIONE TEMPORALE + PYDANTIC TOOLS:

  RISOLUZIONE TEMPORALE (v3.1):
    Il middleware temporale è stato potenziato:
    - Rileva riferimenti temporali relativi ("quest'anno", "anno scorso",
      "due anni fa", "prossimo semestre") e assoluti impliciti ("oggi",
      "domani", "lunedì prossimo")
    - Inietta un BLOCCO DI CONTESTO TEMPORALE strutturato nella query,
      contenente data corrente, anno corrente, anno precedente, anno
      successivo — così il LLM può risolvere autonomamente le espressioni
      temporali senza bisogno di parser regex complessi
    - Il blocco viene iniettato PRIMA dell'invio all'agente, in modo
      trasparente per il resto della pipeline

  GUARDRAILS VIA MIDDLEWARE (invariato da v3):
    I guardrails sono middleware LangChain integrati nel grafo dell'agente.

  PYDANTIC TOOLS (v3.1):
    I tool usano ora args_schema con BaseModel + Field per compatibilità
    ottimale con Qwen2.5 7B/14B. L'anno è Optional[int] e viene risolto
    dal LLM grazie al contesto temporale iniettato.

  Il flusso dell'agente è ora:
    User Query → Temporal Middleware (pre-agent, arricchito)
               → Memory → Agent Graph (con middleware integrati)
               → Response
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
# MIDDLEWARE: Risoluzione Temporale Dinamica (v3.1 — Potenziato)
# ============================================================

# --- Pattern per riferimenti temporali RELATIVI ---
# Questi richiedono iniezione del contesto data/anno per il LLM
_TEMPORAL_RELATIVE_PATTERNS = [
    # Riferimenti relativi all'anno
    r"\b(quest['']?\s*anno|anno\s+corrente|anno\s+in\s+corso)\b",
    r"\b(anno\s+scorso|l['']?\s*anno\s+scorso|anno\s+passato|anno\s+precedente)\b",
    r"\b(anno\s+prossimo|l['']?\s*anno\s+prossimo|prossimo\s+anno)\b",
    r"\b(\d+)\s+ann[io]\s+fa\b",
    # Riferimenti relativi a semestre/trimestre
    r"\b(questo|prossimo|scorso|corrente)\s+(semestre|trimestre)\b",
    # Riferimenti relativi a settimana/mese
    r"\b(questa|prossima|scorsa)\s+(settimana|lezione)\b",
    r"\b(questo|prossimo|scorso)\s+(mese)\b",
]

# --- Pattern per riferimenti temporali ASSOLUTI IMPLICITI ---
# Questi richiedono la data esatta per la risoluzione
_TEMPORAL_ABSOLUTE_PATTERNS = [
    r"\b(oggi|domani|ieri|dopodomani|l['']?\s*altro\s+ieri)\b",
    r"\b(luned[iì]|marted[iì]|mercoled[iì]|gioved[iì]|venerd[iì]|sabato|domenica)\s+(prossim[oa]|scors[oa])\b",
    r"\b(luned[iì]|marted[iì]|mercoled[iì]|gioved[iì]|venerd[iì]|sabato|domenica)\b",
    r"\b(ora|adesso|attualmente|corrente|in\s+corso)\b",
    r"\b(quando|a\s+che\s+ora|orario)\b",
    r"\b(prossim[oa]|scors[oa])\b",
]

_TEMPORAL_RELATIVE_REGEX = [
    re.compile(p, re.IGNORECASE) for p in _TEMPORAL_RELATIVE_PATTERNS
]
_TEMPORAL_ABSOLUTE_REGEX = [
    re.compile(p, re.IGNORECASE) for p in _TEMPORAL_ABSOLUTE_PATTERNS
]

# Pattern per verificare se l'utente ha già specificato un anno esatto
_EXPLICIT_YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")


def _has_temporal_reference(query: str) -> bool:
    """Controlla se la query contiene QUALSIASI riferimento temporale."""
    return (
        any(p.search(query) for p in _TEMPORAL_RELATIVE_REGEX)
        or any(p.search(query) for p in _TEMPORAL_ABSOLUTE_REGEX)
    )


def _has_relative_temporal_reference(query: str) -> bool:
    """Controlla se la query contiene riferimenti temporali RELATIVI (che richiedono l'anno)."""
    return any(p.search(query) for p in _TEMPORAL_RELATIVE_REGEX)


def _has_explicit_year(query: str) -> bool:
    """Controlla se la query contiene già un anno esplicito (es. 2024, 2025)."""
    return bool(_EXPLICIT_YEAR_PATTERN.search(query))


def _get_temporal_context_block() -> str:
    """
    Genera il blocco di contesto temporale strutturato per il LLM.

    Questo blocco contiene tutte le informazioni necessarie affinché
    il LLM possa risolvere autonomamente espressioni temporali come
    "quest'anno", "anno scorso", "due anni fa", ecc.

    Il formato è pensato per essere compatto ma non ambiguo per Qwen2.5 7B.
    """
    now = datetime.now()
    giorni = ["lunedì", "martedì", "mercoledì", "giovedì",
              "venerdì", "sabato", "domenica"]
    mesi = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
            "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"]

    anno_corrente = now.year
    anno_precedente = anno_corrente - 1
    anno_successivo = anno_corrente + 1

    return (
        f"[Informazione di sistema: La data odierna è "
        f"{giorni[now.weekday()]} {now.day} {mesi[now.month - 1]} {anno_corrente}, "
        f"ore {now.strftime('%H:%M')}. "
        f"Anno corrente: {anno_corrente}. "
        f"Anno scorso: {anno_precedente}. "
        f"Anno prossimo: {anno_successivo}. "
        f"Usa queste informazioni per risolvere riferimenti temporali relativi "
        f"come 'quest\\'anno' → {anno_corrente}, "
        f"'anno scorso' → {anno_precedente}, "
        f"'due anni fa' → {anno_corrente - 2}. "
        f"Se l'utente menziona un anno relativo, converti nel parametro 'anno' (intero) del tool.]"
    )


def _get_simple_temporal_context() -> str:
    """
    Genera una stringa di contesto temporale semplice (solo data/ora).
    Usata quando ci sono riferimenti temporali assoluti ma non relativi.
    """
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
    Middleware pre-agente: Risoluzione Temporale Dinamica (v3.1).

    Strategia a 3 livelli:
      1. Se l'utente ha già un anno esplicito (es. "bandi 2024") → nessuna iniezione
      2. Se ci sono riferimenti RELATIVI ("quest'anno", "anno scorso") →
         inietta il BLOCCO COMPLETO con anno corrente/precedente/successivo
         per permettere al LLM di risolvere e passare `anno` come int al tool
      3. Se ci sono solo riferimenti ASSOLUTI ("oggi", "domani", "lunedì") →
         inietta solo data/ora corrente (contesto semplice)
    """
    # Livello 1: anno esplicito presente → nessuna iniezione necessaria
    if _has_explicit_year(query):
        logger.debug(f"Anno esplicito rilevato in query, skip iniezione temporale")
        return query

    # Livello 2: riferimenti relativi → blocco completo con anni
    if _has_relative_temporal_reference(query):
        temporal_block = _get_temporal_context_block()
        logger.info(f"Iniezione temporale COMPLETA (riferimento relativo rilevato)")
        return f"{query}\n{temporal_block}"

    # Livello 3: riferimenti assoluti → contesto semplice
    if _has_temporal_reference(query):
        temporal = _get_simple_temporal_context()
        logger.info(f"Iniezione temporale semplice: {temporal}")
        return f"{query}\n{temporal}"

    # Nessun riferimento temporale → query invariata
    return query


# ============================================================
# FACADE: RAGAgent
# ============================================================

class RAGAgent:
    """
    Facade principale per l'interazione con l'agente RAG DIEM.

    NOTA: I guardrails sono ora gestiti internamente dal grafo dell'agente
    tramite middleware LangChain. Questo Facade si occupa solo di:
      1. Pre-processing temporale (v3.1 — risoluzione anno relativo)
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
          1. Middleware temporale: inietta data/anno se necessario (v3.1)
          2. Memory: costruzione messaggi con storico
          3. Chat history injection per tool
          4. Agent: invocazione (i middleware gestiscono guardrails)
          5. Logging + Memory update
        """
        print(f"USER QUERY: {user_query}")

        # --- STEP 1: Middleware Risoluzione Temporale (v3.1) ---
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

    REFACTORING v3.1:
      - Tool con Pydantic args_schema per compatibilità Qwen2.5 7B/14B
      - Risoluzione temporale potenziata nel middleware pre-agente
      - Guardrails middleware LangChain (invariato da v3)
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

        # Log degli schema Pydantic per debug
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
        logger.info("   ⏰ Temporale: middleware dinamico v3.1 (risoluzione anno)")
        logger.info("   🧠 Memoria: SmartConversationMemory")
        logger.info("   🔧 Tools: 4 tool con Pydantic args_schema")
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