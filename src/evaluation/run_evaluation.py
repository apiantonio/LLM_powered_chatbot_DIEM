"""Script di esecuzione per la valutazione RAGAS dell'Agente RAG DIEM.

Lancia l'intero flusso di valutazione dalla command line:
bootstrap dei componenti, caricamento test set, raccolta dati,
valutazione RAGAS e generazione report.

Utilizzo:
    python run_evaluation.py
    python run_evaluation.py --testset evaluation/testset.json
    python run_evaluation.py --output-dir evaluation/results --log-level DEBUG
    python run_evaluation.py --no-guardrails --max-turns 5
    python run_evaluation.py --llm-judge-provider groq --llm-judge-model llama-3.3-70b-versatile
"""

import sys
import os
import argparse
import logging
from datetime import datetime
from typing import Any, Dict, Optional

_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from config.settings import AppSettings, load_settings
from config.logging_config import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parsing degli argomenti da riga di comando.

    Returns:
        Namespace con tutti gli argomenti parsati.
    """
    parser = argparse.ArgumentParser(
        description="Valutazione RAGAS dell'Agente RAG DIEM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi di utilizzo:
  python run_evaluation.py
  python run_evaluation.py --testset mio_testset.json
  python run_evaluation.py --output-dir risultati/ --log-level DEBUG
  python run_evaluation.py --no-guardrails
  python run_evaluation.py --llm-judge-provider groq
  python run_evaluation.py --run-note "Test con reranker Qwen3 e soglia 0.0"
        """,
    )

    parser.add_argument(
        "--testset",
        type=str,
        default="src/evaluation/testset.json",
        help="Percorso del file testset.json (default: src/evaluation/testset.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/evaluation/results",
        help="Directory di output per i report (default: src/evaluation/results)",
    )

    parser.add_argument(
        "--no-guardrails",
        action="store_true",
        help="Disabilita i guardrails durante la valutazione",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=5,
        help="Numero massimo di turni di memoria (default: 5)",
    )

    parser.add_argument(
        "--llm-judge-provider",
        type=str,
        default=None,
        help=(
            "Provider LLM per il judge RAGAS (groq, ollama). "
            "Se non specificato, usa lo stesso LLM dell'agente "
            "(da LLM_PROVIDER nel .env)."
        ),
    )
    parser.add_argument(
        "--llm-judge-model",
        type=str,
        default=None,
        help=(
            "Modello LLM per il judge RAGAS. "
            "Se non specificato, usa lo stesso modello dell'agente "
            "(da LLM_MODEL nel .env)."
        ),
    )
    parser.add_argument(
        "--llm-judge-api-key",
        type=str,
        default=None,
        help="API key per il judge RAGAS (se provider=groq).",
    )

    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Lista delle metriche da calcolare. "
            "Valori: context_precision, context_recall, "
            "response_relevancy, faithfulness, factual_correctness. "
            "Default: tutte."
        ),
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help=(
            "Numero massimo di chiamate LLM concorrenti durante la "
            "valutazione RAGAS. Impostare a 1 per API con rate limit "
            "stringenti (Ollama locale, Groq free tier). Aumentare a "
            "2-4 se l'API lo consente. Default: 1 (sequenziale)."
        ),
    )

    parser.add_argument(
        "--run-note",
        type=str,
        default=None,
        help=(
            "Nota descrittiva della run, inclusa nel JSON summary "
            "per facilitare il confronto tra esperimenti diversi "
            "(es. 'Test con chunk_size=500 e reranker attivo')."
        ),
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Livello di logging (default: da LOG_LEVEL nel .env)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="File di log (default: da LOG_FILE nel .env)",
    )

    return parser.parse_args()


def build_llm_judge(
    args: argparse.Namespace,
    settings: AppSettings,
):
    """Costruisce il LLM judge per la valutazione RAGAS.

    Strategia di selezione del judge:
    1. Se --llm-judge-provider e' specificato, crea un LLM dedicato
       con quel provider e modello.
    2. Altrimenti, usa lo stesso provider/modello configurato nel .env
       (LLM_PROVIDER / LLM_MODEL), creando l'istanza corrispondente.

    In entrambi i casi, se il provider e' "groq" e manca una API key
    valida, cade in fallback sul modello Ollama locale (FALLBACK_MODEL).

    NOTA: Per i modelli Ollama, viene impostato format="json" per
    forzare output JSON strutturato, necessario affinche' RAGAS possa
    parsare correttamente le risposte del judge.

    Args:
        args: Argomenti da riga di comando.
        settings: Configurazione dell'applicazione (da .env).

    Returns:
        Istanza di BaseChatModel per il judge RAGAS.
    """

    provider = (args.llm_judge_provider or settings.llm.provider).lower()
    model = args.llm_judge_model

    logger.info(
        "Configurazione LLM judge: provider=%s, model=%s",
        provider,
        model or "(default dal .env)",
    )

    if provider == "groq":
        effective_api_key = (
            args.llm_judge_api_key
            or os.getenv("GROQ_JUDGE_API_KEY")
            or settings.llm.groq_rewriter_api_key
            or settings.llm.groq_chat_api_key
        )

        if not effective_api_key:
            logger.warning(
                "Nessuna API key Groq disponibile per il judge. "
                "Fallback su Ollama locale (%s).",
                settings.llm.fallback_model,
            )
            return _build_ollama_judge(
                model=settings.llm.fallback_model,
                base_url=settings.llm.fallback_base_url,
            )

        effective_model = model or settings.llm.rewriter_model
        try:
            from langchain_groq import ChatGroq

            judge = ChatGroq(
                model=effective_model,
                temperature=0.0,
                max_tokens=1024,
                api_key=effective_api_key,
            )
            logger.info("LLM judge creato: Groq/%s", effective_model)
            return judge
        except Exception as e:
            logger.warning(
                "Errore creazione judge Groq: %s. Fallback su Ollama locale.",
                e,
            )
            return _build_ollama_judge(
                model=settings.llm.fallback_model,
                base_url=settings.llm.fallback_base_url,
            )

    if provider == "ollama":
        effective_model = model or settings.llm.model_name
        return _build_ollama_judge(
            model=effective_model,
            base_url=settings.llm.ollama_base_url,
        )

    logger.warning(
        "Provider judge '%s' non supportato. Fallback su Ollama locale (%s).",
        provider,
        settings.llm.fallback_model,
    )
    return _build_ollama_judge(
        model=settings.llm.fallback_model,
        base_url=settings.llm.fallback_base_url,
    )


def _build_ollama_judge(model: str, base_url: str):
    """Crea un'istanza ChatOllama da usare come judge RAGAS.

    Configura il modello con:
    - format="json": forza output JSON strutturato, fondamentale per il
      parsing interno di RAGAS (PydanticPrompt). Senza questo, i modelli
      thinking (Nemotron, Qwen3, DeepSeek-R1) producono testo libero con
      tag <think> che causa RagasOutputParserException.
    - temperature=0.0: output deterministico per riproducibilita'.
    - num_ctx=8192: contesto esteso necessario per i prompt complessi
      di RAGAS (statement_generator, faithfulness) che includono
      l'intero contesto recuperato + la risposta dell'agente.
    - num_predict=2048: limite generazione sufficiente per le risposte
      JSON strutturate di RAGAS.

    Args:
        model: Nome del modello Ollama (es. 'nemotron-3-super:cloud', 'qwen2.5').
        base_url: URL del server Ollama (es. 'http://localhost:11434').

    Returns:
        Istanza di ChatOllama configurata per la valutazione RAGAS.
    """
    from langchain_ollama import ChatOllama

    judge = ChatOllama(
        model=model,
        temperature=0.0,
        num_predict=2048,
        num_ctx=8192,
        format="json",
        base_url=base_url,
    )
    logger.info(
        "LLM judge creato: Ollama/%s su %s "
        "(format=json, num_ctx=8192, num_predict=2048)",
        model,
        base_url,
    )
    return judge


def build_run_metadata(
    args: argparse.Namespace,
    settings: AppSettings,
) -> Dict[str, Any]:
    """Costruisce i metadati della run per il JSON summary.

    Cattura automaticamente la configurazione corrente (modelli, parametri
    di retrieval, embedding, reranker) cosi' da poter confrontare run
    diverse senza dover ricordare cosa era attivo in ciascuna.

    Args:
        args: Argomenti da riga di comando.
        settings: Configurazione dell'applicazione.

    Returns:
        Dizionario con tutti i metadati rilevanti della run.
    """
    judge_provider = (args.llm_judge_provider or settings.llm.provider).lower()
    judge_model = args.llm_judge_model or (
        settings.llm.rewriter_model if judge_provider == "groq"
        else settings.llm.model_name
    )

    metadata = {
        "agent_provider": settings.llm.provider,
        "agent_model": settings.llm.model_name,
        "agent_temperature": settings.llm.temperature,
        "agent_max_tokens": settings.llm.max_tokens,
        "guardrails_enabled": not args.no_guardrails,
        "max_memory_turns": args.max_turns,
        "judge_provider": judge_provider,
        "judge_model": judge_model,
        "judge_format": "json" if judge_provider == "ollama" else "default",
        "embedding_model": settings.embedding.model_name,
        "reranker_model": settings.reranker.model_name,
        "reranker_score_threshold": settings.reranker.score_treshold,
        "reranker_top_n": settings.reranker.top_n,
        "search_k": settings.vectorstore.search_k,
        "rewriter_provider": settings.llm.rewriter_provider,
        "rewriter_model": settings.llm.rewriter_model,
        "testset_path": args.testset,
        "metrics": args.metrics or [
            "context_precision",
            "context_recall",
            "response_relevancy",
            "faithfulness",
            "factual_correctness",
        ],
        "max_workers": args.max_workers,
    }

    if args.run_note:
        metadata["note"] = args.run_note

    return metadata


def main() -> None:
    """Entry point principale dello script di valutazione."""
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

    judge_provider = (args.llm_judge_provider or settings.llm.provider).lower()
    judge_model = args.llm_judge_model or (
        settings.llm.rewriter_model if judge_provider == "groq"
        else settings.llm.model_name
    )

    print("\n" + "=" * 60)
    print("  VALUTAZIONE RAGAS - Agente RAG DIEM")
    print("=" * 60)
    print(f"  Data:          {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"  Test set:      {args.testset}")
    print(f"  Output:        {args.output_dir}")
    print(f"  Guardrails:    {'OFF' if args.no_guardrails else 'ON'}")
    print(f"  Max turns:     {args.max_turns}")
    print(f"  Metriche:      {args.metrics or 'tutte (default)'}")
    print(f"  Max workers:   {args.max_workers}")
    print(f"  Agente:        {settings.llm.provider}/{settings.llm.model_name}")
    print(f"  Judge:         {judge_provider}/{judge_model}")
    if judge_provider == "ollama":
        print(f"  Judge format:  json (forzato per compatibilita' RAGAS)")
    print(f"  Embedding:     {settings.embedding.model_name}")
    print(f"  Reranker:      {settings.reranker.model_name}")
    if args.run_note:
        print(f"  Nota:          {args.run_note}")
    print("=" * 60 + "\n")

    if not os.path.exists(args.testset):
        logger.error("File test set non trovato: %s", args.testset)
        print(f"\n  ERRORE: File test set non trovato: {args.testset}")
        print("  Crea il file testset.json con le domande e le ground truth.")
        sys.exit(1)

    logger.info("Fase 1: Bootstrap dei componenti")
    print("  [1/5] Bootstrap dei componenti...")

    try:
        from agent.agent_main import (
            build_retrieval_engine,
            build_embedding_model,
            build_agent,
        )

        engine = build_retrieval_engine(settings)
        embedding_model = build_embedding_model(settings)
        agent = build_agent(
            settings=settings,
            engine=engine,
            enable_scope_guardrail=not args.no_guardrails,
            max_memory_turns=args.max_turns,
            embedding_model=embedding_model,
        )
        print("  [1/5] Bootstrap completato.\n")

    except Exception as e:
        logger.error("Errore durante il bootstrap: %s", e, exc_info=True)
        print(f"\n  ERRORE nel bootstrap: {e}")
        sys.exit(1)

    logger.info("Fase 2: Configurazione LLM judge")
    print("  [2/5] Configurazione LLM judge...")

    llm_judge = build_llm_judge(args, settings)
    print("  [2/5] LLM judge configurato.\n")

    logger.info("Fase 3: Inizializzazione RAGASEvaluator")
    print("  [3/5] Inizializzazione evaluator...")

    try:
        from evaluation.ragas_evaluator import RAGASEvaluator

        evaluator = RAGASEvaluator(
            llm_judge=llm_judge,
            embedding_model=embedding_model,
            metrics=args.metrics,
            max_workers=args.max_workers,
        )
        print("  [3/5] Evaluator inizializzato.\n")

    except ImportError as e:
        logger.error("Errore importazione modulo evaluation: %s", e)
        print(f"\n  ERRORE: {e}")
        print("  Verifica che la directory evaluation/ sia nel PYTHONPATH.")
        sys.exit(1)

    run_metadata = build_run_metadata(args, settings)

    logger.info("Fase 4-5: Esecuzione valutazione completa")
    print("  [4/5] Raccolta dati e valutazione RAGAS in corso...")
    print("         (questo puo' richiedere diversi minuti)\n")

    try:
        results = evaluator.run_full_evaluation(
            agent=agent,
            testset_path=args.testset,
            output_dir=args.output_dir,
            run_metadata=run_metadata,
        )
    except Exception as e:
        logger.error("Errore durante la valutazione: %s", e, exc_info=True)
        print(f"\n  ERRORE durante la valutazione: {e}")
        sys.exit(1)

    print("\n  [5/5] Valutazione completata!")

    if "error" in results:
        print(f"\n  ATTENZIONE: {results['error']}")
        sys.exit(1)

    scores = results.get("scores", {})
    if scores:
        print("\n  --- Riepilogo Score ---")
        for metric_name, score in scores.items():
            if isinstance(score, (int, float)):
                print(f"    {metric_name:<30s} {score:.4f}")

    print(f"\n  Report salvati in: {args.output_dir}/")
    print(f"  (CSV, Markdown e JSON summary con metadati della run)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()