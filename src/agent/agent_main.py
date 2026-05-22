"""Entry point per l'agente RAG DIEM.

Gestisce il bootstrap dei componenti (RetrievalEngine, embedding, agente)
e fornisce un REPL interattivo per l'interazione con l'assistente.
"""

import sys
import os
import json
import logging
import argparse
from typing import Optional

from src.config.settings import AppSettings, load_settings
from src.config.logging_config import setup_logging
from src.ingestion.indexer import KnowledgeBaseIndexer
from src.retrieval.engine import RetrievalEngine, QueryOptimizer, CrossEncoderReranker
from src.agent.agent import RAGAgentFactory, RAGAgent
from src.agent.llm_providers import create_chat_model, create_rewriter_llm
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


def build_retrieval_engine(settings: AppSettings) -> RetrievalEngine:
    logger.info("[BOOT] Inizializzazione Indexer...")
    indexer = KnowledgeBaseIndexer(settings)

    logger.info("[BOOT] Inizializzazione Cross-Encoder Reranker...")
    reranker = CrossEncoderReranker(settings.reranker)

    rewriter_provider = settings.llm.rewriter_provider
    if rewriter_provider:
        logger.info(
            "[BOOT] QueryOptimizer: provider dedicato richiesto (%s/%s)",
            rewriter_provider.upper(),
            settings.llm.rewriter_model,
        )
    else:
        logger.info(
            "[BOOT] QueryOptimizer: usa LLM principale (%s/%s) con temperature=%.2f",
            settings.llm.provider.upper(),
            settings.llm.model_name,
            settings.llm.rewriter_temperature,
        )

    rewriter_llm = create_rewriter_llm(config=settings.llm)

    logger.info("[BOOT] Inizializzazione QueryOptimizer...")
    query_optimizer = QueryOptimizer(rewriter_llm, config=settings.query_optimizer)

    logger.info("[BOOT] Assemblaggio RetrievalEngine...")
    engine = RetrievalEngine(
        indexer=indexer,
        reranker=reranker,
        query_optimizer=query_optimizer,
        dedup_hash_chars=settings.query_optimizer.dedup_hash_chars,
    )
    logger.info("[BOOT] RetrievalEngine pronto.")
    return engine


def build_embedding_model(settings: AppSettings) -> HuggingFaceEmbeddings:
    logger.info("[BOOT] Inizializzazione HuggingFaceEmbeddings: %s", settings.embedding.model_name)
    embedding_model = HuggingFaceEmbeddings(
        model_name=settings.embedding.model_name,
        encode_kwargs={"normalize_embeddings": settings.embedding.normalize_embeddings},
    )
    logger.info("[BOOT] Embedding model pronto.")
    return embedding_model


def build_agent(
    settings: AppSettings,
    engine: RetrievalEngine,
    enable_scope_guardrail: bool = True,
    embedding_model: Optional[HuggingFaceEmbeddings] = None,
) -> RAGAgent:
    return RAGAgentFactory.create(
        retrieval_engine=engine,
        settings=settings,
        enable_scope_guardrail=enable_scope_guardrail,
        embedding_model=embedding_model,
    )


WELCOME_BANNER = """
================================================================
             AGENTE RAG DIEM -- UniSA
                                                              
  Assistente virtuale del Dipartimento DIEM                   
  Universita degli Studi di Salerno                           
                                                              
  Comandi:                                                    
    /reset   -- nuova sessione (cancella memoria)             
    /memory  -- mostra lo storico conversazione               
    /traces  -- esporta le trace di osservabilita (JSON)      
    /quit    -- esci                                          
================================================================
"""


def run_repl(agent: RAGAgent) -> None:
    print(WELCOME_BANNER)

    while True:
        try:
            user_input = input("\n Tu: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n Alla prossima!")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            handled = _handle_command(user_input, agent)
            if handled == "quit":
                break
            continue

        result = agent.chat(user_input)

        if result.get("blocked"):
            print(f"\n [{result['block_reason']}] {result['response']}")
        else:
            print(f"\n Agente (turno #{result['turn']}):\n{result['response']}")


def _handle_command(user_input: str, agent: RAGAgent) -> Optional[str]:
    command = user_input.lower()

    if command in ("/quit", "/exit"):
        print("\n Alla prossima!")
        return "quit"

    if command == "/reset":
        agent.reset_memory()
        print(" Memoria resettata. Nuova sessione avviata.")
        return None

    if command == "/memory":
        summary = agent.memory.get_history_summary()
        print(f"\n Storico conversazione:\n{summary}")
        return None

    if command == "/traces":
        traces = agent.get_all_traces()
        if not traces:
            print(" Nessuna trace disponibile.")
        else:
            filename = "traces_export.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(traces, f, indent=2, ensure_ascii=False)
            print(f" {len(traces)} trace esportate in '{filename}'")
        return None

    print(f"  Comando sconosciuto: {command}")
    print("   Comandi disponibili: /reset, /memory, /traces, /quit")
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agente RAG DIEM -- Assistente virtuale DIEM UniSA",
    )
    parser.add_argument("--no-scope-guard", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
    )
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--single-query", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    settings = load_settings()

    log_level = args.log_level or settings.logging.level
    log_file = args.log_file or settings.logging.log_file

    setup_logging(
        level=log_level,
        log_file=log_file,
        log_to_console=settings.logging.log_to_console,
        log_format=settings.logging.log_format,
        date_format=settings.logging.date_format,
    )

    logger.info("=" * 60)
    logger.info(" AVVIO AGENTE RAG DIEM")
    logger.info("=" * 60)

    engine = build_retrieval_engine(settings)
    embedding_model = build_embedding_model(settings)

    agent = build_agent(
        settings=settings,
        engine=engine,
        enable_scope_guardrail=not args.no_scope_guard,
        embedding_model=embedding_model,
    )

    if args.single_query:
        result = agent.chat(args.single_query)
        print(f"\n{result['response']}")
    else:
        run_repl(agent)


if __name__ == "__main__":
    main()