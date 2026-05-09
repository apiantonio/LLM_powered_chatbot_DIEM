"""
RAG Observability Handler — Callback LangChain per tracciamento completo della pipeline.

Questo modulo implementa un BaseCallbackHandler che si aggancia al grafo
dell'agente LangChain per fornire visibilità totale su:
  1. Input ricevuto dall'utente.
  2. Tool invocati dall'agente (nome, argomenti, output).
  3. Documenti recuperati dal retriever (chunk, metadata, source_url, doc_type).
  4. Risposta finale generata.

Pattern: Observer (GoF) — il callback osserva il grafo senza modificarne il comportamento.

Uso:
    handler = RAGObservabilityHandler()
    agent.invoke({"messages": [...]}, config={"callbacks": [handler]})
    
    # Dopo l'invocazione, i dati catturati sono disponibili per RAGAS:
    handler.get_trace()  →  {"input": ..., "tools": [...], "retrieved_docs": [...], "output": ...}

KPI Impact:
  - Debugging: visione cristallina del flusso runtime.
  - Sprint 6 (RAGAS): i dati catturati alimentano direttamente l'evaluation pipeline.
"""

import logging
import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class ToolInvocation:
    """Record di una singola invocazione di tool."""
    tool_name: str
    tool_input: str
    tool_output: str = ""
    documents_retrieved: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PipelineTrace:
    """Trace completo di una singola interazione utente-agente."""
    user_input: str = ""
    tools_invoked: List[ToolInvocation] = field(default_factory=list)
    final_output: str = ""
    query_rewritten: str = ""
    total_candidates: int = 0
    total_after_rerank: int = 0


class RAGObservabilityHandler(BaseCallbackHandler):
    """
    Callback handler per tracciamento completo del ciclo ReAct dell'agente.
    
    Si aggancia ai seguenti eventi LangChain:
      - on_tool_start  → cattura nome tool e input
      - on_tool_end    → cattura output e, per search_knowledge_base, i documenti
      - on_llm_start   → cattura il prompt inviato al modello
      - on_chain_end   → cattura la risposta finale
    
    Thread-safe: ogni istanza traccia una singola invocazione.
    Per sessioni multiple, creare un handler per ogni invoke().
    """
    
    name = "RAGObservabilityHandler"
    
    def __init__(self, verbose: bool = True, max_preview_chars: int = 200):
        super().__init__()
        self._verbose = verbose
        self._max_preview = max_preview_chars
        self._trace = PipelineTrace()
        self._current_tool: Optional[ToolInvocation] = None
    
    # ==============================
    # TOOL TRACKING
    # ==============================
    
    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        tool_name = serialized.get("name", "unknown_tool")
        self._current_tool = ToolInvocation(
            tool_name=tool_name,
            tool_input=input_str,
        )
        
        if self._verbose:
            logger.info(
                f"\n{'='*60}\n"
                f"🔧 TOOL INVOCATO: {tool_name}\n"
                f"   Input: {input_str[:self._max_preview]}\n"
                f"{'='*60}"
            )
    
    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        if self._current_tool is None:
            return
        
        self._current_tool.tool_output = output
        
        # Se il tool è search_knowledge_base, parsa i documenti recuperati
        if self._current_tool.tool_name == "search_knowledge_base":
            self._current_tool.documents_retrieved = self._parse_retrieved_docs(output)
            
            if self._verbose:
                self._log_retrieved_documents(self._current_tool.documents_retrieved)
        
        if self._verbose:
            preview = output[:self._max_preview]
            logger.info(
                f"   ✅ Output ({len(output)} chars): {preview}..."
            )
        
        self._trace.tools_invoked.append(self._current_tool)
        self._current_tool = None
    
    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        if self._current_tool:
            self._current_tool.tool_output = f"ERRORE: {error}"
            self._trace.tools_invoked.append(self._current_tool)
            self._current_tool = None
        
        logger.error(f"   ❌ Tool error: {error}")
    
    # ==============================
    # LLM TRACKING
    # ==============================
    
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        if self._verbose:
            model_name = serialized.get("id", ["unknown"])[-1] if serialized.get("id") else "unknown"
            logger.info(f"🧠 LLM chiamato: {model_name}")
    
    # ==============================
    # AGENT LIFECYCLE
    # ==============================
    
    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List, **kwargs: Any
    ) -> None:
        """Cattura l'input dell'utente dal primo messaggio."""
        if not self._trace.user_input and messages:
            # Il primo batch di messaggi contiene l'input utente
            for msg_batch in messages:
                for msg in msg_batch:
                    if hasattr(msg, 'type') and msg.type == 'human':
                        self._trace.user_input = msg.content
                        if self._verbose:
                            logger.info(
                                f"\n{'#'*60}\n"
                                f"👤 INPUT UTENTE: {msg.content}\n"
                                f"{'#'*60}"
                            )
                        return
    
    # ==============================
    # PUBLIC API
    # ==============================
    
    def get_trace(self) -> PipelineTrace:
        """Restituisce il trace completo dell'interazione corrente."""
        return self._trace
    
    def get_trace_dict(self) -> Dict[str, Any]:
        """
        Restituisce il trace come dizionario — formato diretto per RAGAS evaluation.
        
        Struttura:
        {
            "user_input": "...",
            "tools": [{"name": ..., "input": ..., "output": ..., "docs": [...]}],
            "retrieved_contexts": ["chunk1", "chunk2", ...],
            "source_urls": ["https://...", "https://..."],
            "final_output": "..."
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
            "tools": [
                {
                    "name": t.tool_name,
                    "input": t.tool_input,
                    "output_preview": t.tool_output[:500],
                    "docs_count": len(t.documents_retrieved),
                }
                for t in self._trace.tools_invoked
            ],
            "retrieved_contexts": all_contexts,
            "source_urls": all_sources,
            "final_output": self._trace.final_output,
        }
    
    def print_summary(self) -> None:
        """Stampa un riepilogo leggibile dell'intera interazione."""
        trace = self._trace
        
        print(f"\n{'='*70}")
        print(f"📊 RIEPILOGO INTERAZIONE RAG")
        print(f"{'='*70}")
        print(f"👤 Input: {trace.user_input}")
        print(f"🔧 Tool invocati: {len(trace.tools_invoked)}")
        
        for i, tool in enumerate(trace.tools_invoked, 1):
            print(f"\n  [{i}] {tool.tool_name}")
            print(f"      Query: {tool.tool_input}")
            if tool.documents_retrieved:
                print(f"      Documenti recuperati: {len(tool.documents_retrieved)}")
                for j, doc in enumerate(tool.documents_retrieved, 1):
                    source = doc.get("source_url", "N/D")
                    doc_type = doc.get("doc_type", "N/D")
                    score = doc.get("relevance_score", "N/D")
                    print(f"        📄 [{j}] {doc_type} — score: {score}")
                    print(f"           Fonte: {source}")
        
        print(f"\n💬 Risposta finale: {trace.final_output[:300]}...")
        print(f"{'='*70}\n")
    
    # ==============================
    # PRIVATE HELPERS
    # ==============================
    
    @staticmethod
    def _parse_retrieved_docs(tool_output: str) -> List[Dict[str, Any]]:
        """
        Parsa l'output del tool search_knowledge_base per estrarre i documenti.
        
        Il formato atteso è:
        [Documento N — tipo — source_url]
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
                # Parse header: [Documento N — tipo — url]
                header = lines[0].strip("[]")
                parts = [p.strip() for p in header.split("—")]
                if len(parts) >= 3:
                    doc_info["doc_type"] = parts[1]
                    doc_info["source_url"] = parts[2]
                elif len(parts) >= 2:
                    doc_info["doc_type"] = parts[1]
                
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