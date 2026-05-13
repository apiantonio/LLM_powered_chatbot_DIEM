"""
Entry point per l'esecuzione interattiva dell'Agente RAG DIEM.

Bootstrapping completo: Settings → Indexer → RetrievalEngine → Agent → REPL.

AGGIORNAMENTO v3 — Middleware-based Guardrails:
  I guardrails non sono più oggetti standalone ma middleware LangChain
  integrati nel grafo dell'agente. Il bootstrap passa il chat_model
  alla factory che lo usa sia per l'agente sia per il classificatore
  topicale (Layer 2).

Uso:
  cd src/
  python -m agent.agent_main                    # avvio standard
  python -m agent.agent_main --no-scope-guard   # senza scope guardrail LLM
  python -m agent.agent_main --skip-crawl       # senza re-indicizzazione

Il REPL interattivo supporta:
  - Conversazione multi-turno con memoria intelligente
    (filtro similarità + summarization)
  - Comandi speciali: /reset, /traces, /memory, /quit
  - Osservabilità completa a terminale (callbacks glass-box)
"""

import sys
import os
import json
import logging
import argparse
from typing import Optional

# ============================================================
# PATH SETUP — assicura che 'src/' sia nel PYTHONPATH
# ============================================================
_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from config.settings import AppSettings, load_settings
from ingestion.indexer import KnowledgeBaseIndexer
from retrieval.engine import RetrievalEngine, QueryOptimizer, CrossEncoderReranker
from agent.agent import RAGAgentFactory, RAGAgent
from agent.llm_providers import create_chat_model
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


# ============================================================
# BOOTSTRAP: costruzione dell'intera pipeline
# ============================================================

def build_retrieval_engine(settings: AppSettings) -> RetrievalEngine:
    """
    Costruisce il RetrievalEngine assemblando Indexer, Reranker e QueryOptimizer.
    """
    logger.info("[BOOT] Inizializzazione Indexer (Vector Store + Parent Store)...")
    indexer = KnowledgeBaseIndexer(settings)

    logger.info("[BOOT] Inizializzazione Cross-Encoder Reranker...")
    reranker = CrossEncoderReranker(settings.reranker)

    logger.info("[BOOT] Inizializzazione ChatModel per QueryOptimizer...")
    chat_model = create_chat_model(settings.llm)
    query_optimizer = QueryOptimizer(chat_model)

    logger.info("[BOOT] Assemblaggio RetrievalEngine...")
    engine = RetrievalEngine(
        indexer=indexer,
        reranker=reranker,
        query_optimizer=query_optimizer,
    )

    logger.info("[BOOT] ✅ RetrievalEngine pronto.")
    return engine


def build_embedding_model(settings: AppSettings) -> HuggingFaceEmbeddings:
    """
    Costruisce il modello di embedding condiviso per la SmartConversationMemory.
    """
    logger.info(
        f"[BOOT] Inizializzazione HuggingFaceEmbeddings per SmartMemory: "
        f"{settings.embedding.model_name}"
    )
    embedding_model = HuggingFaceEmbeddings(
        model_name=settings.embedding.model_name,
        encode_kwargs={
            "normalize_embeddings": settings.embedding.normalize_embeddings,
        },
    )
    logger.info("[BOOT] ✅ Embedding model per SmartMemory pronto.")
    return embedding_model


def build_agent(
    settings: AppSettings,
    engine: RetrievalEngine,
    enable_scope_guardrail: bool = True,
    max_memory_turns: int = 10,
    embedding_model: Optional[HuggingFaceEmbeddings] = None,
) -> RAGAgent:
    """
    Costruisce l'agente RAG completo tramite la Factory.

    I guardrails sono ora middleware LangChain integrati nel grafo
    dell'agente — non servono più oggetti separati.
    """
    return RAGAgentFactory.create(
        retrieval_engine=engine,
        settings=settings,
        enable_scope_guardrail=enable_scope_guardrail,
        max_memory_turns=max_memory_turns,
        embedding_model=embedding_model,
    )


# ============================================================
# REPL: loop interattivo a terminale
# ============================================================

WELCOME_BANNER = """
╔══════════════════════════════════════════════════════════════╗
║           🎓  AGENTE RAG DIEM — UniSA                      ║
║                                                              ║
║  Assistente virtuale del Dipartimento DIEM                   ║
║  Università degli Studi di Salerno                           ║
║                                                              ║
║  Memoria: SmartConversationMemory                            ║
║    • Stadio 1: Filtro similarità coseno                      ║
║    • Stadio 2: Summarization con token budget                ║
║                                                              ║
║  Guardrails: Middleware-based (v3)                            ║
║    • Injection Guard    • Toxicity Filter                    ║
║    • Topical Guard      • Hallucination Guard                ║
║    • Code Gen Guard     • PII Guard (CF only)                ║
║    • Model Call Limit   • Tool Call Limit                    ║
║                                                              ║
║  Comandi:                                                    ║
║    /reset   — nuova sessione (cancella memoria)              ║
║    /memory  — mostra lo storico conversazione                ║
║    /traces  — esporta le trace di osservabilità (JSON)       ║
║    /quit    — esci                                           ║
╚══════════════════════════════════════════════════════════════╝
"""


def run_repl(agent: RAGAgent) -> None:
    """Loop REPL interattivo per conversazione con l'agente."""
    print(WELCOME_BANNER)

    while True:
        try:
            user_input = input("\n🧑 Tu: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Alla prossima!")
            break

        if not user_input:
            continue

        # --- Comandi speciali ---
        if user_input.startswith("/"):
            command = user_input.lower()

            if command == "/quit" or command == "/exit":
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

        # --- Invocazione agente ---
        result = agent.chat(user_input)

        if result.get("blocked"):
            print(f"\n🚫 [{result['block_reason']}] {result['response']}")
        else:
            print(f"\n🤖 Agente (turno #{result['turn']}):\n{result['response']}")


# ============================================================
# CLI: parsing argomenti
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agente RAG DIEM — Assistente virtuale del Dipartimento DIEM UniSA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--no-scope-guard",
        action="store_true",
        help="Disabilita il TopicalGuardrail Layer 2 (classificatore LLM). "
             "Il Layer 1 (regex) rimane sempre attivo.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Numero massimo di turni in memoria (default: 10).",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Livello di logging (default: INFO).",
    )
    parser.add_argument(
        "--single-query",
        type=str,
        default=None,
        help="Esegui una singola query e termina (senza REPL).",
    )
    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    # --- Logging ---
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- Settings ---
    settings = load_settings()

    # --- Bootstrap pipeline ---
    logger.info("=" * 60)
    logger.info("🚀 AVVIO AGENTE RAG DIEM (v3 — Middleware Guardrails)")
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

    # --- Modalità single-query o REPL ---
    if args.single_query:
        result = agent.chat(args.single_query)
        print(f"\n{result['response']}")
    else:
        run_repl(agent)


if __name__ == "__main__":
    main()