"""
RAG Observability Handler — Callback LangChain per tracciamento COMPLETO della pipeline.

REDESIGN COMPLETO DELL'OUTPUT:
  L'output precedente era illeggibile:
  - Stringhe troncate con "..." ovunque
  - Nessuna query riscritta visibile
  - Sezioni duplicate (dettaglio + riepilogo)
  - Preview inutili del prompt augmentato
  - Struttura prompt LLM loggata ad ogni iterazione (rumore)

NUOVO FORMATO:
  Output lineare a step numerati, SENZA troncamenti.
  Ogni turno segue un flusso chiaro:

    ══ TURNO #N ═══════════════════════════════════════
    STEP 1 │ INPUT UTENTE
    STEP 2 │ QUERY RISCRITTA (solo se diversa)
    STEP 3 │ ROUTING → tool [collection] (sezione)
    STEP 4 │ DOCUMENTI RECUPERATI (fonte + score, NO preview contenuto)
    STEP 5 │ RISPOSTA FINALE (testo COMPLETO)
    ══ FINE TURNO #N ══ 2 LLM calls ══ 1245ms ════════

Pattern: Observer (GoF), Builder (GoF).
"""

import logging
import json
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum, auto

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)


# ============================================================
# ENUMS & DATA CLASSES
# ============================================================

class PipelinePhase(Enum):
    USER_INPUT = auto()
    QUERY_OPTIMIZATION = auto()
    TOOL_INVOCATION = auto()
    RETRIEVAL = auto()
    RERANKING = auto()
    AUGMENTED_PROMPT = auto()
    LLM_GENERATION = auto()
    FINAL_OUTPUT = auto()


_SEARCH_TOOL_PREFIX = "search_"

_TOOL_COLLECTION_MAP = {
    "search_docenti": "docenti_e_didattica",
    "search_offerta_formativa": "offerta_formativa_e_corsi",
    "search_bandi": "bandi_e_amministrazione",
    "search_dipartimento": "dipartimento_e_ricerca",
    "search_strutture_fisiche": "dipartimento_e_ricerca",
    "search_all": "ALL",
    "get_course_schedule": "easycourse",
}


@dataclass
class ToolInvocation:
    tool_name: str
    tool_input: str
    tool_params: Dict[str, Any] = field(default_factory=dict)
    tool_output: str = ""
    documents_retrieved: List[Dict[str, Any]] = field(default_factory=list)
    duration_ms: float = 0.0
    collection_queried: str = ""
    react_iteration: int = 0


@dataclass
class PipelineTrace:
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
# OBSERVER: RAGObservabilityHandler (REDESIGN)
# ============================================================

class RAGObservabilityHandler(BaseCallbackHandler):
    """
    Callback handler con output LINEARE e COMPLETO.
    
    Nessun troncamento. Nessuna preview. Nessun dump della struttura prompt.
    Solo i passi della pipeline, in ordine, con tutti i dati visibili.
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
        self._max_preview = max_preview_chars  # Mantenuto per compatibilità trace dict
        self._trace = PipelineTrace(conversation_turn=conversation_turn)
        self._current_tool: Optional[ToolInvocation] = None
        self._tool_start_time: float = 0.0
        self._pipeline_start_time: float = time.time()
        self._llm_call_index: int = 0
        self._react_iteration: int = 0
        self._header_printed: bool = False
    
    # ==============================
    # CHAT MODEL TRACKING
    # ==============================
    
    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List, **kwargs: Any
    ) -> None:
        self._llm_call_index += 1
        self._trace.llm_calls_count = self._llm_call_index
        
        if not messages:
            return
        
        # Estrai l'input utente (solo la prima volta)
        if not self._trace.user_input:
            flat_messages = []
            for batch in messages:
                if isinstance(batch, list):
                    flat_messages.extend(batch)
                else:
                    flat_messages.append(batch)
            
            for msg in reversed(flat_messages):
                if hasattr(msg, 'type') and msg.type == 'human':
                    raw_content = msg.content
                    # Rimuovi il reminder di sistema dall'input visualizzato
                    clean_input = raw_content.split("\n\n[SISTEMA:")[0].strip()
                    clean_input = clean_input.split("\n\n[Invoca un tool")[0].strip()
                    self._trace.user_input = clean_input
                    break
            
            # STEP 1: Stampa l'header del turno e l'input
            if self._verbose and self._trace.user_input:
                self._header_printed = True
                turn = self._trace.conversation_turn
                print(f"\n{'═' * 70}")
                print(f"  TURNO #{turn}")
                print(f"{'═' * 70}")
                print(f"  STEP 1 │ INPUT UTENTE")
                print(f"         │ \"{self._trace.user_input}\"")
    
    # ==============================
    # TOOL TRACKING
    # ==============================
    
    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        tool_name = serialized.get("name", "unknown_tool")
        self._tool_start_time = time.time()
        self._react_iteration += 1
        
        tool_params = self._extract_tool_input(input_str)
        clean_query = tool_params.get(
            "query",
            tool_params.get(
                "course_or_professor",
                str(next(iter(tool_params.values()), ""))
            )
        )
        
        collection_queried = _TOOL_COLLECTION_MAP.get(tool_name, "")
        sezione = tool_params.get("sezione", None)
        
        self._current_tool = ToolInvocation(
            tool_name=tool_name,
            tool_input=clean_query,
            tool_params=tool_params,
            collection_queried=collection_queried,
            react_iteration=self._react_iteration,
        )
        
        if not self._verbose:
            return
        
        # STEP 2: Query riscritta (solo se diversa dall'input originale)
        if (self._trace.user_input 
                and clean_query.strip().lower() != self._trace.user_input.strip().lower()
                and not self._trace.query_rewritten):
            self._trace.query_rewritten = clean_query
            print(f"  STEP 2 │ QUERY RISCRITTA")
            print(f"         │ originale:  \"{self._trace.user_input}\"")
            print(f"         │ riscritta:  \"{clean_query}\"")
        
        # STEP 3: Routing
        step_num = 3 if self._trace.query_rewritten else 2
        sezione_str = f" → sezione=\"{sezione}\"" if sezione else ""
        iter_str = f" (iter #{self._react_iteration})" if self._react_iteration > 1 else ""
        
        print(f"  STEP {step_num} │ ROUTING{iter_str}")
        print(f"         │ tool:       {tool_name}")
        print(f"         │ collection: {collection_queried}{sezione_str}")
        print(f"         │ query:      \"{clean_query}\"")
    
    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        if self._current_tool is None:
            return
        
        if isinstance(output, str):
            output_str = output
        elif hasattr(output, 'content'):
            output_str = str(output.content)
        else:
            output_str = str(output)
        
        duration = (time.time() - self._tool_start_time) * 1000
        self._current_tool.tool_output = output_str
        self._current_tool.duration_ms = round(duration, 1)
        
        # Parsa i documenti
        if self._current_tool.tool_name.startswith(_SEARCH_TOOL_PREFIX):
            self._current_tool.documents_retrieved = self._parse_retrieved_docs(output_str)
        
        if self._verbose:
            docs = self._current_tool.documents_retrieved
            has_rewrite = bool(self._trace.query_rewritten)
            step_num = 4 if has_rewrite else 3
            
            # STEP 4: Documenti recuperati
            if docs:
                print(f"  STEP {step_num} │ DOCUMENTI RECUPERATI: {len(docs)} ({self._current_tool.duration_ms}ms)")
                for i, doc in enumerate(docs, 1):
                    source = doc.get("source_url", "N/D")
                    doc_type = doc.get("doc_type", "?")
                    category = doc.get("doc_category", "?")
                    score = doc.get("relevance_score", None)
                    score_str = f"{score:.4f}" if score is not None else "N/D"
                    print(f"         │   [{i}] score={score_str}  tipo={doc_type}  cat={category}")
                    print(f"         │       fonte: {source}")
            elif "Errore" in output_str:
                print(f"  STEP {step_num} │ ERRORE TOOL ({self._current_tool.duration_ms}ms)")
                # Mostra l'errore COMPLETO, senza troncamento
                print(f"         │ {output_str}")
            else:
                print(f"  STEP {step_num} │ NESSUN DOCUMENTO TROVATO ({self._current_tool.duration_ms}ms)")
        
        self._trace.tools_invoked.append(self._current_tool)
        self._current_tool = None
    
    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        if self._current_tool:
            self._current_tool.tool_output = f"ERRORE: {error}"
            self._trace.tools_invoked.append(self._current_tool)
            self._current_tool = None
        if self._verbose:
            print(f"         │ ❌ ERRORE TOOL: {error}")
    
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        pass  # Non loggare nulla per LLM non-chat
    
    # ==============================
    # PUBLIC API
    # ==============================
    
    def set_final_output(self, output: str) -> None:
        self._trace.final_output = output
        self._trace.total_duration_ms = round(
            (time.time() - self._pipeline_start_time) * 1000, 1
        )
        
        if self._verbose:
            has_rewrite = bool(self._trace.query_rewritten)
            step_num = 5 if has_rewrite else 4
            
            # STEP 5: Risposta finale — COMPLETA, senza troncamento
            print(f"  STEP {step_num} │ RISPOSTA FINALE")
            # Stampa ogni riga con indentazione
            for line in output.split("\n"):
                print(f"         │ {line}")
            
            # Footer del turno
            turn = self._trace.conversation_turn
            llm_calls = self._trace.llm_calls_count
            duration = self._trace.total_duration_ms
            tools_count = len(self._trace.tools_invoked)
            print(f"{'═' * 70}")
            print(
                f"  FINE TURNO #{turn}  │  "
                f"{tools_count} tool  │  "
                f"{llm_calls} LLM calls  │  "
                f"{duration:.0f}ms"
            )
            print(f"{'═' * 70}")
    
    def get_trace(self) -> PipelineTrace:
        return self._trace
    
    def get_trace_dict(self) -> Dict[str, Any]:
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
                    "params": t.tool_params,
                    "output_preview": t.tool_output[:500],
                    "docs_count": len(t.documents_retrieved),
                    "duration_ms": t.duration_ms,
                    "collection_queried": t.collection_queried,
                    "react_iteration": t.react_iteration,
                }
                for t in self._trace.tools_invoked
            ],
            "retrieved_contexts": all_contexts,
            "source_urls": all_sources,
            "augmented_prompt_preview": self._trace.augmented_prompt_preview[:500],
            "final_output": self._trace.final_output,
            "conversation_turn": self._trace.conversation_turn,
            "llm_calls": self._trace.llm_calls_count,
            "total_react_iterations": self._react_iteration,
            "total_duration_ms": self._trace.total_duration_ms,
        }
    
    def print_summary(self) -> None:
        """
        NON stampa nulla — il flusso è già stato stampato step-by-step.
        
        Il vecchio print_summary() duplicava tutto con troncamenti.
        Ora l'output è già completo e lineare durante l'esecuzione.
        """
        pass
    
    # ==============================
    # PRIVATE HELPERS
    # ==============================
    
    @staticmethod
    def _extract_tool_input(input_str: str) -> dict:
        """Parsa l'input del tool e restituisce TUTTI i parametri come dict."""
        if not input_str:
            return {"query": ""}
        
        try:
            parsed = json.loads(input_str)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        
        try:
            import ast
            parsed = ast.literal_eval(input_str)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError):
            pass
        
        return {"query": input_str}
    
    @staticmethod
    def _parse_retrieved_docs(tool_output: str) -> List[Dict[str, Any]]:
        """Parsa l'output dei tool search_* per estrarre i documenti."""
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
                
                if len(parts) >= 2:
                    doc_info["doc_type"] = parts[1].strip()
                if len(parts) >= 3:
                    doc_info["doc_category"] = parts[2].strip()
                if len(parts) >= 4:
                    doc_info["source_url"] = parts[3].strip()
                if len(parts) >= 5:
                    score_part = parts[4].strip()
                    if score_part.startswith("score:"):
                        try:
                            doc_info["relevance_score"] = float(
                                score_part.split(":")[1].strip()
                            )
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


# ============================================================
# FACTORY
# ============================================================

def create_observability_handler(
    settings: "ObservabilityConfig",
    conversation_turn: int = 1,
) -> RAGObservabilityHandler:
    """Factory Method: crea un RAGObservabilityHandler configurato."""
    return RAGObservabilityHandler(
        verbose=settings.enable_verbose_callbacks,
        max_preview_chars=settings.max_chunk_preview_chars,
        conversation_turn=conversation_turn,
    )