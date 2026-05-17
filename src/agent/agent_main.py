"""
Entry point per l'esecuzione interattiva dell'Agente RAG DIEM.

v4: Il QueryOptimizer usa un LLM dedicato (Groq Llama 70B)
    configurabile via REWRITER_PROVIDER / REWRITER_MODEL / GROQ_API_KEY.
"""

import sys
import os
import json
import logging
import argparse
from typing import Optional

_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from config.settings import AppSettings, load_settings
from ingestion.indexer import KnowledgeBaseIndexer
from retrieval.engine import RetrievalEngine, QueryOptimizer, CrossEncoderReranker
from agent.agent import RAGAgentFactory, RAGAgent
from agent.llm_providers import create_chat_model, create_rewriter_llm
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


def build_retrieval_engine(settings: AppSettings) -> RetrievalEngine:
    logger.info("[BOOT] Inizializzazione Indexer...")
    indexer = KnowledgeBaseIndexer(settings)

    logger.info("[BOOT] Inizializzazione Cross-Encoder Reranker...")
    reranker = CrossEncoderReranker(settings.reranker)

    rewriter_provider = os.getenv("REWRITER_PROVIDER", "").strip()
    if rewriter_provider:
        logger.info(
            f"[BOOT] QueryOptimizer: LLM dedicato via {rewriter_provider.upper()} "
            f"({os.getenv('REWRITER_MODEL', 'llama-3.3-70b-versatile')})"
        )
    else:
        logger.info(
            f"[BOOT] QueryOptimizer: LLM principale ({settings.llm.model_name}) "
            f"con temperature=0.0"
        )

    rewriter_llm = create_rewriter_llm(fallback_config=settings.llm)

    logger.info("[BOOT] Inizializzazione QueryOptimizer...")
    query_optimizer = QueryOptimizer(rewriter_llm)

    logger.info("[BOOT] Assemblaggio RetrievalEngine...")
    engine = RetrievalEngine(
        indexer=indexer,
        reranker=reranker,
        query_optimizer=query_optimizer,
    )
    logger.info("[BOOT] ✅ RetrievalEngine pronto.")
    return engine


def build_embedding_model(settings: AppSettings) -> HuggingFaceEmbeddings:
    logger.info(f"[BOOT] Inizializzazione HuggingFaceEmbeddings: {settings.embedding.model_name}")
    embedding_model = HuggingFaceEmbeddings(
        model_name=settings.embedding.model_name,
        encode_kwargs={"normalize_embeddings": settings.embedding.normalize_embeddings},
    )
    logger.info("[BOOT] ✅ Embedding model pronto.")
    return embedding_model


def build_agent(
    settings: AppSettings,
    engine: RetrievalEngine,
    enable_scope_guardrail: bool = True,
    max_memory_turns: int = 10,
    embedding_model: Optional[HuggingFaceEmbeddings] = None,
) -> RAGAgent:
    return RAGAgentFactory.create(
        retrieval_engine=engine,
        settings=settings,
        enable_scope_guardrail=enable_scope_guardrail,
        max_memory_turns=max_memory_turns,
        embedding_model=embedding_model,
    )


WELCOME_BANNER = """
╔══════════════════════════════════════════════════════════════╗
║           🎓  AGENTE RAG DIEM — UniSA                      ║
║                                                              ║
║  Assistente virtuale del Dipartimento DIEM                   ║
║  Università degli Studi di Salerno                           ║
║                                                              ║
║  Query Rewriting: v4 (Llama 3.3 70B via Groq)                ║
║                                                              ║
║  Comandi:                                                    ║
║    /reset   — nuova sessione (cancella memoria)              ║
║    /memory  — mostra lo storico conversazione                ║
║    /traces  — esporta le trace di osservabilità (JSON)       ║
║    /quit    — esci                                           ║
╚══════════════════════════════════════════════════════════════╝
"""


def run_repl(agent: RAGAgent) -> None:
    print(WELCOME_BANNER)

    while True:
        try:
            user_input = input("\n🧑 Tu: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Alla prossima!")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            command = user_input.lower()

            if command in ("/quit", "/exit"):
                print("\n👋 Alla prossima!")
                break
            elif command == "/reset":
                agent.reset_memory()
                print("🔄 Memoria resettata. Nuova sessione avviata.")
                continue
            elif command == "/memory":
                summary = agent.memory.get_history_summary()
                print(f"\n📜 Storico conversazione:\n{summary}")
                continue
            elif command == "/traces":
                traces = agent.get_all_traces()
                if not traces:
                    print("📭 Nessuna trace disponibile.")
                else:
                    filename = "traces_export.json"
                    with open(filename, "w", encoding="utf-8") as f:
                        json.dump(traces, f, indent=2, ensure_ascii=False)
                    print(f"💾 {len(traces)} trace esportate in '{filename}'")
                continue
            else:
                print(f"⚠️  Comando sconosciuto: {command}")
                print("   Comandi disponibili: /reset, /memory, /traces, /quit")
                continue

        result = agent.chat(user_input)

        if result.get("blocked"):
            print(f"\n🚫 [{result['block_reason']}] {result['response']}")
        else:
            print(f"\n🤖 Agente (turno #{result['turn']}):\n{result['response']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agente RAG DIEM — Assistente virtuale DIEM UniSA",
    )
    parser.add_argument("--no-scope-guard", action="store_true")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    parser.add_argument("--single-query", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    settings = load_settings()

    logger.info("=" * 60)
    logger.info("🚀 AVVIO AGENTE RAG DIEM (v4)")
    logger.info("=" * 60)

    engine = build_retrieval_engine(settings)
    embedding_model = build_embedding_model(settings)

    agent = build_agent(
        settings=settings,
        engine=engine,
        enable_scope_guardrail=not args.no_scope_guard,
        max_memory_turns=args.max_turns,
        embedding_model=embedding_model,
    )

    if args.single_query:
        result = agent.chat(args.single_query)
        print(f"\n{result['response']}")
    else:
        run_repl(agent)


if __name__ == "__main__":
    main()