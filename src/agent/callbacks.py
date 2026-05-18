import logging
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum, auto

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)

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
    "search_persone": "persone",
    "search_offerta_formativa": "offerta_formativa",
    "search_dipartimento": "dipartimento",
    "search_all": "ALL (cross-collection)",
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



class RAGObservabilityHandler(BaseCallbackHandler):
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
        self._react_iteration: int = 0
        self._header_printed: bool = False
        self._system_prompt: str = ""

    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List, **kwargs: Any
    ) -> None:
        self._llm_call_index += 1
        self._trace.llm_calls_count = self._llm_call_index

        if not messages:
            return

        flat_messages = []
        for batch in messages:
            if isinstance(batch, list):
                flat_messages.extend(batch)
            else:
                flat_messages.append(batch)

        if self._llm_call_index == 1:
            for msg in flat_messages:
                if hasattr(msg, 'type') and msg.type == 'system':
                    self._system_prompt = msg.content
                    break

        if not self._trace.user_input:
            for msg in reversed(flat_messages):
                if hasattr(msg, 'type') and msg.type == 'human':
                    raw_content = msg.content
                    clean_input = raw_content.split("\n\n[SISTEMA:")[0].strip()
                    clean_input = clean_input.split("\n\n[Invoca un tool")[0].strip()
                    self._trace.user_input = clean_input
                    break

            if self._verbose and self._trace.user_input:
                self._header_printed = True
                turn = self._trace.conversation_turn
                print(f"\n{'═' * 70}")
                print(f"  TURNO #{turn}")
                print(f"{'═' * 70}")
                print(f"  INPUT │ \"{self._trace.user_input}\"")

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        tool_name = serialized.get("name", "unknown_tool")
        self._tool_start_time = time.time()
        self._react_iteration += 1

        tool_params = self._extract_tool_input(input_str)
        clean_query = tool_params.get(
            "query",
            str(next(iter(tool_params.values()), ""))
        )

        collection_queried = _TOOL_COLLECTION_MAP.get(tool_name, "")

        self._current_tool = ToolInvocation(
            tool_name=tool_name,
            tool_input=clean_query,
            tool_params=tool_params,
            collection_queried=collection_queried,
            react_iteration=self._react_iteration,
        )

        if self._verbose:
            print(f"  TOOL  │ {tool_name} → {collection_queried}")

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

        if self._current_tool.tool_name.startswith(_SEARCH_TOOL_PREFIX):
            self._current_tool.documents_retrieved = self._parse_retrieved_docs(output_str)

        if self._verbose:
            docs = self._current_tool.documents_retrieved
            if docs:
                print(
                    f"  DOCS  │ {len(docs)} documenti recuperati "
                    f"({self._current_tool.duration_ms}ms)"
                )
            elif "Errore" in output_str:
                print(f"  ERR   │ Errore tool ({self._current_tool.duration_ms}ms)")
            else:
                print(f"  DOCS  │ Nessun documento trovato ({self._current_tool.duration_ms}ms)")

        self._trace.tools_invoked.append(self._current_tool)
        self._current_tool = None

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        if self._current_tool:
            self._current_tool.tool_output = f"ERRORE: {error}"
            self._trace.tools_invoked.append(self._current_tool)
            self._current_tool = None
        if self._verbose:
            print(f"  ERR   │ ❌ {error}")

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        pass

    def get_system_prompt(self) -> str:
        """Restituisce il system prompt catturato."""
        return self._system_prompt

    def set_final_output(self, output: str) -> None:
        self._trace.final_output = output
        self._trace.total_duration_ms = round(
            (time.time() - self._pipeline_start_time) * 1000, 1
        )

        if self._verbose:
            turn = self._trace.conversation_turn
            llm_calls = self._trace.llm_calls_count
            duration = self._trace.total_duration_ms
            tools_count = len(self._trace.tools_invoked)
            preview = output[:200] + "..." if len(output) > 200 else output
            print(f"  RESP  │ {preview}")
            print(f"{'═' * 70}")
            print(
                f"  FINE #{turn}  │  "
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
        """Noop — output già stampato inline."""
        pass

    @staticmethod
    def _extract_tool_input(input_str: str) -> dict:
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
        docs = []
        if not tool_output or "Errore" in tool_output or "Non ho trovato" in tool_output:
            return docs

        blocks = tool_output.split("\n\n---\n\n")
        for block in blocks:
            doc_info: Dict[str, Any] = {}
            lines = block.strip().split("\n", 2)

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

                content_start = 1
                if len(lines) > 1 and lines[1].startswith("[meta:"):
                    meta_line = lines[1].strip("[meta: ").rstrip("]")
                    for pair in meta_line.split(" | "):
                        if "=" in pair:
                            k, v = pair.split("=", 1)
                            doc_info[k.strip()] = v.strip()
                    content_start = 2

                if len(lines) > content_start:
                    doc_info["content"] = lines[content_start].strip()
                else:
                    doc_info["content"] = ""
            else:
                doc_info["content"] = block.strip()

            if doc_info.get("content"):
                docs.append(doc_info)

        return docs


class InteractionLogHandler:
    def __init__(self, output_dir: str = "logs/interactions"):
        self._output_dir = output_dir
        os.makedirs(self._output_dir, exist_ok=True)

    def save_interaction(
        self,
        turn_number: int,
        system_prompt: str,
        history: str,
        user_query: str,
        rewritten_query: str,
        multi_queries: List[str],
        tool_name: str,
        metadata_info: str,
        top_links: List[str],
        final_response: str,
    ) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        lines = []

        lines.append("=== SYSTEM PROMPT ===")
        lines.append(system_prompt if system_prompt else "(non catturato)")
        lines.append("")

        lines.append("=== HISTORY ===")
        lines.append(history if history else "(nessuna history)")
        lines.append("")

        lines.append("=== USER QUERY ===")
        lines.append(user_query)
        lines.append("")

        lines.append("=== REWRITTEN QUERY ===")
        lines.append(rewritten_query if rewritten_query else user_query)
        lines.append("")

        lines.append("=== MULTIQUERIES ===")
        if multi_queries:
            for i, mq in enumerate(multi_queries, 1):
                lines.append(f"{i}. {mq}")
        else:
            lines.append("(nessuna multiquery generata)")
        lines.append("")

        lines.append("=== TOOL CALLED & METADATA ===")
        lines.append(f"{tool_name} - {metadata_info}")
        lines.append("")

        lines.append("=== TOP 5 RETRIEVED LINKS ===")
        if top_links:
            for i, link in enumerate(top_links[:5], 1):
                lines.append(f"{i}. {link}")
        else:
            lines.append("(nessun link recuperato)")
        lines.append("")

        lines.append("=== FINAL RESPONSE ===")
        lines.append(final_response)

        filename = f"interaction_turn{turn_number}_{timestamp}.txt"
        filepath = os.path.join(self._output_dir, filename)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            logger.info(f"Log interazione salvato: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Errore salvataggio log interazione: {e}")
            return ""


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


def create_interaction_log_handler(
    output_dir: str = "logs/interactions",
) -> InteractionLogHandler:
    """Factory Method: crea un InteractionLogHandler."""
    return InteractionLogHandler(output_dir=output_dir)