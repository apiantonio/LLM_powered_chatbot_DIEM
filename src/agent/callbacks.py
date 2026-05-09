"""
RAG Observability Handler — Callback LangChain per tracciamento completo della pipeline.

Questo modulo implementa un BaseCallbackHandler che si aggancia al grafo
dell'agente LangChain per fornire visibilità totale su:
  1. Input ricevuto dall'utente.
  2. Query Optimization (riscrittura contestualizzata).
  3. Tool invocati dall'agente (nome, argomenti, output).
  4. Documenti recuperati dal retriever (chunk, metadata, source_url, doc_type).
  5. Prompt Augmentato (contesto + query) inviato al LLM.
  6. Risposta finale generata.

Pattern: Observer (GoF) — il callback osserva il grafo senza modificarne il comportamento.
         Il handler è un Subject-less Observer: si registra nel runtime LangChain
         tramite il meccanismo config={"callbacks": [handler]}, riceve notifiche
         su ogni evento del ciclo di vita senza accoppiamento diretto con l'agente.

Pattern: Builder (GoF) — PipelineTrace viene costruito incrementalmente dai callback,
         ogni evento aggiunge un pezzo alla trace completa. Il get_trace() restituisce
         l'oggetto fully-built solo al termine dell'interazione.

Uso:
    handler = RAGObservabilityHandler()
    agent.invoke({"messages": [...]}, config={"callbacks": [handler]})
    
    # Dopo l'invocazione, i dati catturati sono disponibili per RAGAS:
    handler.get_trace()  →  PipelineTrace(...)
    handler.get_trace_dict()  →  {"input": ..., "tools": [...], ...}

KPI Impact:
  - Debugging: visione cristallina del flusso runtime.
  - Sprint 6 (RAGAS): i dati catturati alimentano direttamente l'evaluation pipeline.
"""

import logging
import json
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum, auto

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


# ============================================================
# ENUMS & DATA CLASSES (Builder pattern - componenti della trace)
# ============================================================

class PipelinePhase(Enum):
    """Fasi osservabili della pipeline RAG."""
    USER_INPUT = auto()
    QUERY_OPTIMIZATION = auto()
    TOOL_INVOCATION = auto()
    RETRIEVAL_HTML = auto()
    RETRIEVAL_PDF = auto()
    RERANKING = auto()
    AUGMENTED_PROMPT = auto()
    LLM_GENERATION = auto()
    FINAL_OUTPUT = auto()


@dataclass
class ToolInvocation:
    """Record di una singola invocazione di tool."""
    tool_name: str
    tool_input: str
    tool_output: str = ""
    documents_retrieved: List[Dict[str, Any]] = field(default_factory=list)
    duration_ms: float = 0.0


@dataclass
class PipelineTrace:
    """
    Trace completo di una singola interazione utente-agente.
    
    Builder Pattern: costruito incrementalmente dai callback.
    Ogni campo viene popolato da un evento diverso nel lifecycle.
    """
    user_input: str = ""
    query_rewritten: str = ""
    tools_invoked: List[ToolInvocation] = field(default_factory=list)
    augmented_prompt_preview: str = ""
    final_output: str = ""
    total_candidates: int = 0
    total_after_rerank: int = 0
    llm_calls_count: int = 0
    total_duration_ms: float = 0.0
    conversation_turn: int = 0


# ============================================================
# OBSERVER: RAGObservabilityHandler
# ============================================================

class RAGObservabilityHandler(BaseCallbackHandler):
    """
    Callback handler per tracciamento glass-box del ciclo ReAct dell'agente.
    
    Observer Pattern (GoF):
      - Si registra come listener nel runtime LangChain via config.
      - Riceve notifiche passive su ogni evento (tool, llm, chain).
      - NON modifica il comportamento dell'agente — pura osservazione.
    
    Builder Pattern (GoF):
      - Costruisce PipelineTrace incrementalmente: ogni on_* aggiunge un pezzo.
      - get_trace() restituisce la trace completa solo a fine interazione.
    
    Si aggancia ai seguenti eventi LangChain:
      - on_chat_model_start → cattura input utente e prompt augmentato
      - on_tool_start       → cattura nome tool e input (query ottimizzata)
      - on_tool_end         → cattura output e documenti recuperati
      - on_tool_error       → cattura errori nei tool
    
    Thread-safe: ogni istanza traccia una singola invocazione.
    Per sessioni multiple, creare un handler per ogni invoke().
    """
    
    name = "RAGObservabilityHandler"
    
    def __init__(
        self,
        verbose: bool = True,
        max_preview_chars: int = 200,
        conversation_turn: int = 1,
    ):
        super().__init__()
        self._verbose = verbose
        self._max_preview = max_preview_chars
        self._trace = PipelineTrace(conversation_turn=conversation_turn)
        self._current_tool: Optional[ToolInvocation] = None
        self._tool_start_time: float = 0.0
        self._pipeline_start_time: float = time.time()
        self._llm_call_index: int = 0
    
    # ==============================
    # CHAT MODEL TRACKING (input + augmented prompt)
    # ==============================
    
    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List, **kwargs: Any
    ) -> None:
        """
        Cattura i messaggi inviati al LLM — glass-box completo.
        
        Visualizza la struttura completa del prompt:
        - System prompt (troncato)
        - Storico conversazione (human/assistant)
        - Tool results (contesto recuperato)
        - Query utente corrente
        """
        self._llm_call_index += 1
        self._trace.llm_calls_count = self._llm_call_index
        
        model_name = "unknown"
        if serialized.get("id"):
            model_name = serialized["id"][-1] if isinstance(serialized["id"], list) else str(serialized["id"])
        
        if self._verbose:
            logger.info(f"\n🧠 LLM CALL #{self._llm_call_index} → modello: {model_name}")
        
        if not messages:
            return
        
        # Flatten messaggi per ispezione
        flat_messages = []
        for batch in messages:
            if isinstance(batch, list):
                flat_messages.extend(batch)
            else:
                flat_messages.append(batch)
        
        # --- Prima chiamata: cattura input utente (ULTIMO HumanMessage) ---
        if not self._trace.user_input:
            for msg in reversed(flat_messages):
                if hasattr(msg, 'type') and msg.type == 'human':
                    self._trace.user_input = msg.content
                    if self._verbose:
                        logger.info(
                            f"\n{'#'*60}\n"
                            f"👤 TURNO #{self._trace.conversation_turn} — INPUT UTENTE:\n"
                            f"   \"{msg.content}\"\n"
                            f"{'#'*60}"
                        )
                    break
        
        # --- Log struttura completa del prompt inviato al LLM ---
        if self._verbose:
            self._log_full_prompt_structure(flat_messages)
    
    def _log_full_prompt_structure(self, flat_messages: List) -> None:
        """
        Logga la struttura completa dei messaggi inviati al LLM.
        
        Mostra ogni messaggio con il suo ruolo, evidenziando:
        - SYSTEM: il system prompt (troncato)
        - HUMAN: domande utente (storico + corrente)
        - AI: risposte precedenti dell'agente
        - AI (tool_calls): decisioni di invocazione tool
        - TOOL: contesto recuperato dal retrieval
        """
        logger.info(f"\n{'─'*60}")
        logger.info(f"📋 STRUTTURA PROMPT → LLM (totale: {len(flat_messages)} messaggi)")
        logger.info(f"{'─'*60}")
        
        tool_context_parts = []
        
        for i, msg in enumerate(flat_messages):
            msg_type = getattr(msg, 'type', 'unknown')
            content = getattr(msg, 'content', '')
            content_str = str(content) if content else ''
            
            if msg_type == 'system':
                # System prompt — mostra solo le prime righe
                preview = content_str[:150].replace('\n', ' ')
                logger.info(f"   [{i}] 🔒 SYSTEM ({len(content_str)} chars): {preview}...")
            
            elif msg_type == 'human':
                logger.info(f"   [{i}] 👤 HUMAN: \"{content_str[:120]}\"")
            
            elif msg_type == 'ai':
                tool_calls = getattr(msg, 'tool_calls', None)
                if tool_calls:
                    tool_names = [tc.get('name', '?') for tc in tool_calls]
                    logger.info(f"   [{i}] 🤖 AI → tool_calls: {tool_names}")
                else:
                    preview = content_str[:120] if content_str else "(vuoto)"
                    logger.info(f"   [{i}] 🤖 AI: \"{preview}...\"")
            
            elif msg_type == 'tool':
                tool_name = getattr(msg, 'name', 'unknown_tool')
                # Questo è il contesto recuperato — salvalo per la trace
                tool_context_parts.append(content_str)
                logger.info(
                    f"   [{i}] 🔧 TOOL [{tool_name}] ({len(content_str)} chars): "
                    f"\"{content_str[:100]}...\""
                )
            
            else:
                logger.info(f"   [{i}] ❓ {msg_type}: {content_str[:80]}")
        
        logger.info(f"{'─'*60}")
        
        # --- Se ci sono tool results, questa è la query augmentata ---
        if tool_context_parts:
            combined = "\n---\n".join(tool_context_parts)
            self._trace.augmented_prompt_preview = combined[:2000]
            
            # Trova l'ultimo HumanMessage — è la domanda dell'utente
            last_human = ""
            for msg in reversed(flat_messages):
                if getattr(msg, 'type', None) == 'human':
                    last_human = str(getattr(msg, 'content', ''))
                    break
            
            logger.info(
                f"\n{'='*60}\n"
                f"📝 QUERY AUGMENTATA FINALE → LLM:\n"
                f"   Domanda utente: \"{last_human}\"\n"
                f"   Contesto RAG iniettato: {len(tool_context_parts)} blocchi, "
                f"{len(combined)} chars totali\n"
                f"   Preview contesto:\n"
                f"   {combined[:self._max_preview * 2]}...\n"
                f"{'='*60}"
            )
    
    # ==============================
    # TOOL TRACKING
    # ==============================
    
    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """
        Cattura l'inizio di un'invocazione tool — registra nome e input.
        
        NOTA: create_agent passa input_str come rappresentazione stringificata
        del dict degli argomenti, es. "{'query': 'Mario Vento'}".
        Estraiamo il valore della query per display e confronto puliti.
        """
        tool_name = serialized.get("name", "unknown_tool")
        self._tool_start_time = time.time()
        
        # Estrai la query pulita dall'input (che può essere un dict stringificato)
        clean_input = self._extract_tool_query(input_str)
        
        self._current_tool = ToolInvocation(
            tool_name=tool_name,
            tool_input=clean_input,
        )
        
        if self._verbose:
            phase_label = "🔍 RETRIEVAL" if tool_name == "search_knowledge_base" else "📅 TOOL"
            logger.info(
                f"\n{'='*60}\n"
                f"{phase_label}: {tool_name}\n"
                f"   Query inviata al tool: \"{clean_input[:self._max_preview]}\"\n"
                f"{'='*60}"
            )
            
            # Evidenzia la query optimization se la query è diversa dall'input originale
            if (tool_name == "search_knowledge_base" 
                    and self._trace.user_input 
                    and clean_input.strip().lower() != self._trace.user_input.strip().lower()):
                self._trace.query_rewritten = clean_input
                logger.info(
                    f"   ✨ QUERY OPTIMIZATION RILEVATA:\n"
                    f"      Originale: \"{self._trace.user_input}\"\n"
                    f"      Ottimizzata: \"{clean_input}\""
                )
    
    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        """
        Cattura la fine del tool — output, documenti recuperati, durata.
        
        NOTA: create_agent passa un ToolMessage (non str) a questo callback.
        Normalizziamo a stringa estraendo .content se necessario.
        """
        if self._current_tool is None:
            return
        
        # --- Normalizzazione output: ToolMessage → str ---
        if isinstance(output, str):
            output_str = output
        elif hasattr(output, 'content'):
            # ToolMessage, AIMessage, o qualsiasi BaseMessage
            output_str = str(output.content)
        else:
            output_str = str(output)
        
        duration = (time.time() - self._tool_start_time) * 1000
        self._current_tool.tool_output = output_str
        self._current_tool.duration_ms = round(duration, 1)
        
        # Se il tool è search_knowledge_base, parsa i documenti recuperati
        if self._current_tool.tool_name == "search_knowledge_base":
            self._current_tool.documents_retrieved = self._parse_retrieved_docs(output_str)
            
            if self._verbose:
                self._log_retrieved_documents(self._current_tool.documents_retrieved)
        
        if self._verbose:
            preview = output_str[:self._max_preview]
            logger.info(
                f"   ✅ Tool completato in {self._current_tool.duration_ms}ms\n"
                f"   Output ({len(output_str)} chars): {preview}..."
            )
        
        self._trace.tools_invoked.append(self._current_tool)
        self._current_tool = None
    
    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        """Cattura errori nei tool."""
        if self._current_tool:
            self._current_tool.tool_output = f"ERRORE: {error}"
            self._trace.tools_invoked.append(self._current_tool)
            self._current_tool = None
        
        logger.error(f"   ❌ TOOL ERROR: {error}")
    
    # ==============================
    # LLM TRACKING (legacy — per modelli non-chat)
    # ==============================
    
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Fallback per modelli LLM non-chat (non ChatModel)."""
        if self._verbose:
            model_id = serialized.get("id", ["unknown"])
            model_name = model_id[-1] if isinstance(model_id, list) else str(model_id)
            logger.info(f"🧠 LLM (non-chat) chiamato: {model_name}")
    
    # ==============================
    # PUBLIC API (Builder output)
    # ==============================
    
    def set_final_output(self, output: str) -> None:
        """
        Imposta la risposta finale dell'agente.
        
        Chiamato esplicitamente dall'orchestratore dopo invoke(),
        poiché on_chain_end non sempre cattura il messaggio finale
        in modo affidabile con create_agent.
        """
        self._trace.final_output = output
        self._trace.total_duration_ms = round(
            (time.time() - self._pipeline_start_time) * 1000, 1
        )
        
        if self._verbose:
            logger.info(
                f"\n{'#'*60}\n"
                f"💬 RISPOSTA FINALE ({len(output)} chars):\n"
                f"   {output[:self._max_preview * 2]}...\n"
                f"   ⏱️  Durata totale turno: {self._trace.total_duration_ms}ms\n"
                f"{'#'*60}"
            )
    
    def get_trace(self) -> PipelineTrace:
        """Restituisce il trace completo dell'interazione corrente (Builder output)."""
        return self._trace
    
    def get_trace_dict(self) -> Dict[str, Any]:
        """
        Restituisce il trace come dizionario — formato diretto per RAGAS evaluation.
        
        Struttura:
        {
            "user_input": "...",
            "query_rewritten": "...",
            "tools": [{"name": ..., "input": ..., "output_preview": ..., "docs": [...]}],
            "retrieved_contexts": ["chunk1", "chunk2", ...],
            "source_urls": ["https://...", ...],
            "augmented_prompt_preview": "...",
            "final_output": "...",
            "conversation_turn": N,
            "llm_calls": N,
            "total_duration_ms": N
        }
        """
        all_contexts = []
        all_sources = []
        
        for tool_inv in self._trace.tools_invoked:
            for doc in tool_inv.documents_retrieved:
                all_contexts.append(doc.get("content", ""))
                source = doc.get("source_url", "")
                if source and source not in all_sources:
                    all_sources.append(source)
        
        return {
            "user_input": self._trace.user_input,
            "query_rewritten": self._trace.query_rewritten,
            "tools": [
                {
                    "name": t.tool_name,
                    "input": t.tool_input,
                    "output_preview": t.tool_output[:500],
                    "docs_count": len(t.documents_retrieved),
                    "duration_ms": t.duration_ms,
                }
                for t in self._trace.tools_invoked
            ],
            "retrieved_contexts": all_contexts,
            "source_urls": all_sources,
            "augmented_prompt_preview": self._trace.augmented_prompt_preview[:500],
            "final_output": self._trace.final_output,
            "conversation_turn": self._trace.conversation_turn,
            "llm_calls": self._trace.llm_calls_count,
            "total_duration_ms": self._trace.total_duration_ms,
        }
    
    def print_summary(self) -> None:
        """Stampa un riepilogo leggibile dell'intera interazione."""
        trace = self._trace
        
        print(f"\n{'='*70}")
        print(f"📊 RIEPILOGO INTERAZIONE RAG — Turno #{trace.conversation_turn}")
        print(f"{'='*70}")
        print(f"👤 Input utente: {trace.user_input}")
        
        if trace.query_rewritten:
            print(f"✨ Query ottimizzata: {trace.query_rewritten}")
        
        print(f"🔧 Tool invocati: {len(trace.tools_invoked)}")
        
        for i, tool in enumerate(trace.tools_invoked, 1):
            print(f"\n  [{i}] {tool.tool_name} ({tool.duration_ms}ms)")
            print(f"      Query: {tool.tool_input}")
            if tool.documents_retrieved:
                print(f"      Documenti recuperati: {len(tool.documents_retrieved)}")
                for j, doc in enumerate(tool.documents_retrieved, 1):
                    source = doc.get("source_url", "N/D")
                    doc_type = doc.get("doc_type", "N/D")
                    score = doc.get("relevance_score", "N/D")
                    print(f"        📄 [{j}] {doc_type} — score: {score}")
                    print(f"           Fonte: {source}")
        
        if trace.augmented_prompt_preview:
            print(f"\n📝 Prompt augmentato (preview): {trace.augmented_prompt_preview[:200]}...")
        
        print(f"\n🧠 Chiamate LLM: {trace.llm_calls_count}")
        print(f"💬 Risposta finale: {trace.final_output[:300]}...")
        print(f"⏱️  Durata totale: {trace.total_duration_ms}ms")
        print(f"{'='*70}\n")
    
    # ==============================
    # PRIVATE HELPERS
    # ==============================
    
    @staticmethod
    def _extract_tool_query(input_str: str) -> str:
        """
        Estrai la query pulita dall'input del tool.
        
        create_agent passa input_str come repr del dict degli argomenti,
        es. "{'query': 'Mario Vento'}" oppure '{"query": "Mario Vento"}'.
        Questo metodo prova a parsare il dict ed estrarre il primo valore;
        se fallisce, restituisce l'input originale.
        """
        if not input_str:
            return input_str
        
        # Prova parsing come JSON
        try:
            import ast
            parsed = ast.literal_eval(input_str)
            if isinstance(parsed, dict):
                # Restituisci il primo valore (tipicamente 'query')
                return str(next(iter(parsed.values())))
        except (ValueError, SyntaxError, StopIteration):
            pass
        
        try:
            parsed = json.loads(input_str)
            if isinstance(parsed, dict):
                return str(next(iter(parsed.values())))
        except (json.JSONDecodeError, StopIteration):
            pass
        
        return input_str
    
    @staticmethod
    def _parse_retrieved_docs(tool_output: str) -> List[Dict[str, Any]]:
        """
        Parsa l'output del tool search_knowledge_base per estrarre i documenti.
        
        Formato atteso (aggiornato con score):
        [Documento N — tipo — source_url — score: 0.8765]
        contenuto chunk
        
        ---
        
        [Documento N+1 — ...]
        """
        docs = []
        if not tool_output or "Errore" in tool_output or "Non ho trovato" in tool_output:
            return docs
        
        blocks = tool_output.split("\n\n---\n\n")
        for block in blocks:
            doc_info: Dict[str, Any] = {}
            lines = block.strip().split("\n", 1)
            
            if lines and lines[0].startswith("[Documento"):
                header = lines[0].strip("[]")
                parts = [p.strip() for p in header.split("—")]
                
                # Parsing dei campi header:
                # parts[0] = "Documento N"
                # parts[1] = tipo (html/pdf)
                # parts[2] = source_url
                # parts[3] = "score: 0.8765" (opzionale)
                if len(parts) >= 2:
                    doc_info["doc_type"] = parts[1].strip()
                if len(parts) >= 3:
                    doc_info["source_url"] = parts[2].strip()
                if len(parts) >= 4:
                    # Estrai lo score numerico da "score: 0.8765"
                    score_part = parts[3].strip()
                    if score_part.startswith("score:"):
                        try:
                            doc_info["relevance_score"] = float(score_part.split(":")[1].strip())
                        except (ValueError, IndexError):
                            pass
                
                if len(lines) > 1:
                    doc_info["content"] = lines[1].strip()
                else:
                    doc_info["content"] = ""
            else:
                doc_info["content"] = block.strip()
            
            if doc_info.get("content"):
                docs.append(doc_info)
        
        return docs
    
    def _log_retrieved_documents(self, docs: List[Dict[str, Any]]) -> None:
        """Log dettagliato dei documenti recuperati dal retriever."""
        if not docs:
            logger.info("   📭 Nessun documento recuperato")
            return
        
        logger.info(f"\n   📚 DOCUMENTI RECUPERATI: {len(docs)}")
        logger.info(f"   {'─'*50}")
        
        for i, doc in enumerate(docs, 1):
            source = doc.get("source_url", "fonte non disponibile")
            doc_type = doc.get("doc_type", "sconosciuto")
            content_preview = doc.get("content", "")[:self._max_preview]
            score = doc.get("relevance_score", "N/D")
            
            logger.info(
                f"   📄 [{i}] Tipo: {doc_type} | Score: {score}\n"
                f"       Fonte: {source}\n"
                f"       Preview: {content_preview}..."
            )
        
        logger.info(f"   {'─'*50}")


# ============================================================
# FACTORY (GoF) — crea handler pre-configurati da settings
# ============================================================

def create_observability_handler(
    settings: "ObservabilityConfig",
    conversation_turn: int = 1,
) -> RAGObservabilityHandler:
    """
    Factory Method (GoF): crea un RAGObservabilityHandler configurato
    a partire dall'ObservabilityConfig centralizzato in settings.py.
    
    Ogni invoke() dell'agente deve creare un nuovo handler (thread-safe).
    
    Args:
        settings: ObservabilityConfig da AppSettings.
        conversation_turn: Numero del turno corrente nella conversazione.
    
    Returns:
        RAGObservabilityHandler pronto per l'iniezione in config["callbacks"].
    """
    return RAGObservabilityHandler(
        verbose=settings.enable_verbose_callbacks,
        max_preview_chars=settings.max_chunk_preview_chars,
        conversation_turn=conversation_turn,
    )