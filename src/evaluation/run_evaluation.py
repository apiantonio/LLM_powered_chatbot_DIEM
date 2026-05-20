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
from typing import Optional

# Assicura che la directory src sia nel path
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
        """,
    )

    # Percorsi
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

    # Configurazione agente
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

    # Configurazione LLM judge per RAGAS
    parser.add_argument(
        "--llm-judge-provider",
        type=str,
        default=None,
        help=(
            "Provider LLM per il judge RAGAS (groq, ollama, huggingface). "
            "Se non specificato, usa lo stesso LLM dell'agente."
        ),
    )
    parser.add_argument(
        "--llm-judge-model",
        type=str,
        default=None,
        help=(
            "Modello LLM per il judge RAGAS. "
            "Se non specificato, usa lo stesso modello dell'agente."
        ),
    )
    parser.add_argument(
        "--llm-judge-api-key",
        type=str,
        default=None,
        help="API key per il judge RAGAS (se provider=groq).",
    )

    # Metriche
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

    # Parallelismo
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help=(
            "Numero massimo di chiamate LLM concorrenti durante la "
            "valutazione RAGAS. Impostare a 1 per API con rate limit "
            "stringenti (Ollama cloud, Groq free tier). Aumentare a "
            "2-4 se l'API lo consente. Default: 1 (sequenziale)."
        ),
    )

    # Logging
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Livello di logging (default: da settings)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="File di log (default: da settings)",
    )

    return parser.parse_args()


def build_llm_judge(
    args: argparse.Namespace,
    settings: AppSettings,
):
    """Costruisce il LLM judge per la valutazione RAGAS.

    Se il provider del judge e' specificato negli argomenti, crea un LLM
    dedicato. Altrimenti usa lo stesso LLM dell'agente.

    Args:
        args: Argomenti da riga di comando.
        settings: Configurazione dell'applicazione.

    Returns:
        Istanza di BaseChatModel per il judge RAGAS, oppure None
        per usare il default.
    """
    provider = args.llm_judge_provider
    if not provider:
        logger.info(
            "LLM judge: usa lo stesso LLM dell'agente (%s/%s)",
            settings.llm.provider,
            settings.llm.model_name,
        )
        return None

    model = args.llm_judge_model
    api_key = args.llm_judge_api_key

    logger.info(
        "Configurazione LLM judge dedicato: provider=%s, model=%s",
        provider,
        model or "(default del provider)",
    )

    if provider.lower() == "groq":
        from langchain_groq import ChatGroq

        effective_api_key = (
            api_key
            or os.getenv("GROQ_JUDGE_API_KEY")
            or os.getenv("GROQ_REWRITER_API_KEY")
            or os.getenv("GROQ_CHAT_API_KEY")
        )
        if not effective_api_key:
            logger.warning(
                "Nessuna API key Groq disponibile per il judge. "
                "Fallback al LLM dell'agente."
            )
            return None

        effective_model = model or "llama-3.3-70b-versatile"
        judge = ChatGroq(
            model=effective_model,
            temperature=0.0,
            max_tokens=1024,
            api_key=effective_api_key,
        )
        logger.info("LLM judge creato: Groq/%s", effective_model)
        return judge

    if provider.lower() == "ollama":
        from langchain_ollama import ChatOllama

        effective_model = model or settings.llm.model_name
        judge = ChatOllama(
            model=effective_model,
            temperature=0.0,
            num_predict=1024,
            base_url=settings.llm.ollama_base_url,
        )
        logger.info(
            "LLM judge creato: Ollama/%s su %s",
            effective_model,
            settings.llm.ollama_base_url,
        )
        return judge

    logger.warning(
        "Provider judge '%s' non supportato. Fallback al LLM dell'agente.",
        provider,
    )
    return None


def main() -> None:
    """Entry point principale dello script di valutazione."""
    args = parse_args()

    # Carica configurazione
    settings = load_settings()

    # Configura logging
    log_level = args.log_level or settings.logging.level
    log_file = args.log_file or settings.logging.log_file

    setup_logging(
        level=log_level,
        log_file=log_file,
        log_to_console=settings.logging.log_to_console,
        log_format=settings.logging.log_format,
        date_format=settings.logging.date_format,
    )

    print("\n" + "=" * 60)
    print("  VALUTAZIONE RAGAS - Agente RAG DIEM")
    print("=" * 60)
    print(f"  Data:        {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"  Test set:    {args.testset}")
    print(f"  Output:      {args.output_dir}")
    print(f"  Guardrails:  {'OFF' if args.no_guardrails else 'ON'}")
    print(f"  Max turns:   {args.max_turns}")
    print(f"  Metriche:    {args.metrics or 'tutte (default)'}")
    print(f"  Max workers: {args.max_workers}")
    print("=" * 60 + "\n")

    # Verifica esistenza del test set
    if not os.path.exists(args.testset):
        logger.error("File test set non trovato: %s", args.testset)
        print(f"\n  ERRORE: File test set non trovato: {args.testset}")
        print("  Crea il file testset.json con le domande e le ground truth.")
        sys.exit(1)

    # Fase 1: Bootstrap dei componenti
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

    # Fase 2: Configurazione LLM judge
    logger.info("Fase 2: Configurazione LLM judge")
    print("  [2/5] Configurazione LLM judge...")

    llm_judge = build_llm_judge(args, settings)
    print("  [2/5] LLM judge configurato.\n")

    # Fase 3: Inizializzazione RAGASEvaluator
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

    # Fase 4-5: Esecuzione valutazione completa
    logger.info("Fase 4-5: Esecuzione valutazione completa")
    print("  [4/5] Raccolta dati e valutazione RAGAS in corso...")
    print("         (questo puo' richiedere diversi minuti)\n")

    try:
        results = evaluator.run_full_evaluation(
            agent=agent,
            testset_path=args.testset,
            output_dir=args.output_dir,
        )
    except Exception as e:
        logger.error("Errore durante la valutazione: %s", e, exc_info=True)
        print(f"\n  ERRORE durante la valutazione: {e}")
        sys.exit(1)

    # Riepilogo finale
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
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()