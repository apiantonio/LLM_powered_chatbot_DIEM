"""
Agente RAG DIEM — Orchestratore principale.

Assembla i componenti del sistema RAG in un agente conversazionale
basato su `create_agent` di LangChain (NO LangGraph diretto).

Architettura:
  create_agent(model, tools, system_prompt)
      ↓
  Loop ReAct:  LLM → decide tool → esegue tool → LLM → ... → risposta
      ↓
  Esterno: ConversationMemory gestisce lo storico multi-turno
           RAGObservabilityHandler traccia ogni fase

Pattern GoF applicati:
  - Factory Method: RAGAgentFactory costruisce l'agente con tutte le dipendenze.
  - Strategy: ConversationMemory è intercambiabile (in-memory / persistent).
  - Observer: RAGObservabilityHandler osserva il ciclo senza modificarlo.
  - Facade: RAGAgent espone un'API semplice (.chat()) che orchestra tutto.
  - Builder: PipelineTrace viene costruita incrementalmente dai callback.

Vincoli rispettati:
  ✅ Nessun import di langgraph (solo create_agent da langchain.agents)
  ✅ Memoria conversazionale con sliding window
  ✅ Osservabilità totale tramite callbacks
  ✅ Integrazione con RetrievalEngine esistente
  ✅ Coerenza con settings.py e SoC del progetto

KPI Impact:
  - Scope Awareness: system prompt + ScopeGuardrail.
  - Faithfulness: grounding al contesto via search_knowledge_base tool.
  - Correctness: fonti citate, EasyCourse per orari real-time.
  - Robustness: InputSanitizer + OutputValidator.
"""

import logging
from typing import Optional, List, Dict, Any

from langchain_core.messages import AIMessage, BaseMessage

from config.settings import AppSettings, load_settings
from agent.callbacks import RAGObservabilityHandler, create_observability_handler
from agent.memory import ConversationMemory, create_conversation_memory
from agent.prompts import get_agent_system_prompt
from agent.guardrails import (
    ScopeGuardrail,
    InputSanitizer,
    OutputValidator,
    ScopeViolationError,
    InputInjectionError,
)
from agent.tools import set_retrieval_engine, get_all_tools
from agent.llm_providers import create_chat_model
from retrieval.engine import RetrievalEngine

logger = logging.getLogger(__name__)


# ============================================================
# FACADE: RAGAgent
# ============================================================

class RAGAgent:
    """
    Facade (GoF) per l'agente RAG conversazionale.
    
    Espone un'unica interfaccia semplice — .chat(query) — che orchestra
    internamente: guardrails, memory, create_agent, callbacks e output validation.
    
    Responsabilità:
      1. Riceve la query utente.
      2. Applica guardrails pre-agent (scope check, input sanitization).
      3. Costruisce la lista messaggi con storico conversazionale.
      4. Invoca l'agente create_agent con callbacks di osservabilità.
      5. Applica guardrails post-agent (output validation).
      6. Aggiorna la memoria conversazionale.
      7. Restituisce la risposta + trace di osservabilità.
    
    NON contiene logica di business — la delega ai componenti specializzati.
    """
    
    def __init__(
        self,
        agent_graph,
        memory: ConversationMemory,
        scope_guardrail: Optional[ScopeGuardrail],
        input_sanitizer: InputSanitizer,
        output_validator: OutputValidator,
        settings: AppSettings,
    ):
        """
        Args:
            agent_graph: Il grafo compilato restituito da create_agent().
            memory: Strategia di memoria conversazionale.
            scope_guardrail: Guardrail pre-agent per scope check (opzionale).
            input_sanitizer: Guardrail per sanitizzazione input.
            output_validator: Guardrail per validazione output.
            settings: Configurazione centralizzata dell'applicazione.
        """
        self._agent = agent_graph
        self._memory = memory
        self._scope_guardrail = scope_guardrail
        self._input_sanitizer = input_sanitizer
        self._output_validator = output_validator
        self._settings = settings
        self._traces: List[Dict[str, Any]] = []
    
    _MAX_RETRY_NO_TOOL = 1  # Max 1 retry con prompt forzato

    def chat(self, user_query: str) -> Dict[str, Any]:
        """
        Punto di ingresso principale per l'interazione con l'agente.
        
        Flusso completo:
          1. Pre-processing: guardrails (scope + sanitization)
          2. Memory: costruzione messaggi con storico
          3. Agent: invocazione create_agent con callbacks
          4. Post-processing: output validation
          5. Memory: aggiornamento storico
        
        Args:
            user_query: La domanda dell'utente.
        
        Returns:
            Dict con:
              - "response": str — la risposta dell'agente
              - "blocked": bool — True se bloccato da guardrail
              - "block_reason": str — motivazione del blocco (se bloccato)
              - "trace": dict — trace completo di osservabilità
              - "turn": int — numero del turno
        """
        # --- STEP 1: Pre-processing Guardrails ---
        
        # 1a. Input Sanitization (deterministico, sempre attivo)
        sanitizer_ctx: Dict[str, Any] = {}
        passed, sanitized_query = self._input_sanitizer.check(user_query, sanitizer_ctx)
        if not passed:
            logger.warning(f"Input bloccato da InputSanitizer: {user_query[:80]}")
            return {
                "response": sanitized_query,  # Contiene il messaggio di rifiuto
                "blocked": True,
                "block_reason": "prompt_injection",
                "trace": {},
                "turn": self._memory.turn_count + 1,
            }
        
        # 1b. Scope Check (LLM-based, opzionale)
        if self._scope_guardrail:
            passed, scope_result = self._scope_guardrail.check(sanitized_query)
            if not passed:
                logger.info(f"Query OOD bloccata: {user_query[:80]}")
                return {
                    "response": scope_result,  # Contiene il messaggio di rifiuto
                    "blocked": True,
                    "block_reason": "out_of_scope",
                    "trace": {},
                    "turn": self._memory.turn_count + 1,
                }
        
        # --- STEP 2: Registra il turno e costruisci i messaggi ---
        turn_number = self._memory.add_user_message(sanitized_query)
        messages = self._memory.get_messages_for_agent(sanitized_query)
        
        obs_handler = create_observability_handler(
            self._settings.observability,
            conversation_turn=turn_number,
        )
        
        try:
            result = self._agent.invoke(
                {"messages": messages},
                config={"callbacks": [obs_handler]},
            )
            response_text = self._extract_final_response(result)
            
            # --- STEP 4b: POST-HOC VALIDATION (NUOVO) ---
            trace = obs_handler.get_trace()
            if (len(trace.tools_invoked) == 0
                    and not self._is_meta_query(sanitized_query)
                    and self._retry_count < self._MAX_RETRY_NO_TOOL):
                
                logger.warning(
                    f"⚠️ Nessun tool invocato per query non-meta. "
                    f"Retry con prompt forzato."
                )
                self._retry_count += 1
                
                # Modifica l'ultimo messaggio con istruzione di forza
                forced_messages = messages.copy()
                forced_messages[-1]["content"] = (
                    f"[OBBLIGATORIO: Invoca un tool di ricerca ADESSO.] "
                    f"{sanitized_query}"
                )
                
                obs_handler_retry = create_observability_handler(
                    self._settings.observability,
                    conversation_turn=turn_number,
                )
                result = self._agent.invoke(
                    {"messages": forced_messages},
                    config={"callbacks": [obs_handler_retry]},
                )
                response_text = self._extract_final_response(result)
                obs_handler = obs_handler_retry  # Usa il nuovo handler
            
            self._retry_count = 0  # Reset dopo successo
            
        except Exception as e:
            logger.error(f"Errore agente: {e}", exc_info=True)
            response_text = (
                "Mi scuso, si è verificato un errore. Riprova tra qualche istante."
            )
        
        # --- STEP 5: Post-processing (Output Validation) ---
        passed, validated_response = self._output_validator.check(response_text)
        if not passed:
            validated_response = (
                "Mi dispiace, non sono riuscito a generare una risposta valida. "
                "Riprova con una domanda più specifica."
            )
        
        # --- STEP 6: Aggiorna memoria e trace ---
        obs_handler.set_final_output(validated_response)
        self._memory.add_assistant_message(validated_response)
        
        trace_dict = obs_handler.get_trace_dict()
        self._traces.append(trace_dict)
        
        # Stampa riepilogo se verbose
        if self._settings.observability.enable_verbose_callbacks:
            obs_handler.print_summary()
        
        return {
            "response": validated_response,
            "blocked": False,
            "block_reason": None,
            "trace": trace_dict,
            "turn": turn_number,
        }
    
    def get_all_traces(self) -> List[Dict[str, Any]]:
        """Restituisce tutte le trace della sessione (per RAGAS batch evaluation)."""
        return self._traces
    
    def reset_memory(self) -> None:
        """Resetta la memoria conversazionale (nuova sessione)."""
        self._memory.clear()
        self._traces.clear()
        logger.info("Sessione agente resettata")
    
    @property
    def memory(self) -> ConversationMemory:
        """Accesso alla memoria per ispezione esterna."""
        return self._memory
    
    # ==============================
    # PRIVATE HELPERS
    # ==============================
    
    @staticmethod
    def _extract_final_response(result: Dict[str, Any]) -> str:
        """
        Estrai la risposta finale dal risultato di create_agent.invoke().
        
        Il risultato contiene {"messages": [...]}, l'ultimo AIMessage
        senza tool_calls è la risposta finale.
        """
        messages = result.get("messages", [])
        
        # Cerca l'ultimo AIMessage senza tool_calls
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                return msg.content
            # Supporto per dict-format messages
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                if not msg.get("tool_calls"):
                    return msg.get("content", "")
        
        # Fallback: ultimo messaggio qualsiasi con contenuto
        for msg in reversed(messages):
            content = getattr(msg, 'content', None) or msg.get('content', '') if isinstance(msg, dict) else ''
            if content:
                return content
        
        return ""

    @staticmethod
    def _is_meta_query(query: str) -> bool:
        """Rileva saluti e meta-domande."""
        from agent.memory import _is_meta_query
        return _is_meta_query(query)

# ============================================================
# FACTORY: RAGAgentFactory
# ============================================================

class RAGAgentFactory:
    """
    Factory (GoF) per la costruzione dell'agente RAG completo.
    
    Centralizza l'assemblaggio di tutte le dipendenze:
      - ChatModel (da LLMConfig via create_chat_model)
      - Tools (search_knowledge_base, get_course_schedule)
      - System Prompt (da prompts.py)
      - Guardrails (scope, sanitizer, validator)
      - Memory (ConversationMemory)
      - create_agent (LangChain)
    
    Separation of Concerns:
      Il factory conosce COME assemblare i pezzi.
      L'agente conosce COME orchestrare l'interazione.
      I singoli componenti conoscono COME fare il loro lavoro.
    """
    
    @staticmethod
    def create(
        retrieval_engine: RetrievalEngine,
        settings: Optional[AppSettings] = None,
        enable_scope_guardrail: bool = True,
        max_memory_turns: int = 10,
    ) -> RAGAgent:
        """
        Assembla e restituisce un RAGAgent completamente configurato.
        
        Args:
            retrieval_engine: Il RetrievalEngine già inizializzato (Sprint 2).
            settings: Configurazione centralizzata. Se None, carica da env.
            enable_scope_guardrail: Se True, attiva il classificatore OOD.
            max_memory_turns: Numero massimo di turni in memoria.
        
        Returns:
            RAGAgent pronto per l'uso (.chat()).
        
        Raises:
            ValueError: Se il ChatModel non può essere inizializzato.
        """
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
        logger.info("   ✅ System prompt caricato (framework CO-STAR)")
        
        # --- 4. Guardrails ---
        scope_guardrail = None
        if enable_scope_guardrail:
            scope_guardrail = ScopeGuardrail(chat_model)
            logger.info("   ✅ ScopeGuardrail attivato (LLM-based OOD detection)")
        
        input_sanitizer = InputSanitizer()
        logger.info("   ✅ InputSanitizer attivato (regex-based injection detection)")
        
        output_validator = OutputValidator(
            enable_pii_filter=settings.guardrails.enable_pii_filter
        )
        logger.info("   ✅ OutputValidator attivato (PII filter)")
        
        # --- 5. Memory ---
        memory = create_conversation_memory(max_turns=max_memory_turns)
        logger.info(f"   ✅ ConversationMemory: max_turns={max_memory_turns}")
        
        # --- 6. Assembla l'agente con create_agent ---
        from langchain.agents import create_agent
        
        agent_graph = create_agent(
            model=chat_model,
            tools=tools,
            system_prompt=system_prompt,
            name="diem_rag_agent",
        )
        logger.info("   ✅ create_agent() — grafo agente compilato")
        
        # --- 7. Wrappa nel Facade ---
        agent = RAGAgent(
            agent_graph=agent_graph,
            memory=memory,
            scope_guardrail=scope_guardrail,
            input_sanitizer=input_sanitizer,
            output_validator=output_validator,
            settings=settings,
        )
        
        logger.info("=" * 60)
        logger.info("🚀 Agente RAG DIEM assemblato e pronto!")
        logger.info("=" * 60)
        
        return agent


# ============================================================
# CONVENIENCE: funzione top-level per bootstrap rapido
# ============================================================

def bootstrap_agent(
    retrieval_engine: RetrievalEngine,
    settings: Optional[AppSettings] = None,
    **kwargs,
) -> RAGAgent:
    """
    Shortcut per creare un agente RAG completo in una riga.
    
    Uso:
        from retrieval.engine import RetrievalEngine
        from agent.agent import bootstrap_agent
        
        engine = ...  # già inizializzato
        agent = bootstrap_agent(engine)
        result = agent.chat("Quali sono i corsi di laurea del DIEM?")
        print(result["response"])
    """
    return RAGAgentFactory.create(
        retrieval_engine=retrieval_engine,
        settings=settings,
        **kwargs,
    )