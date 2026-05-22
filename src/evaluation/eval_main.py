"""Entry point CLI per il modulo di valutazione del sistema RAG DIEM.

Gestisce il bootstrap dei componenti (RetrievalEngine, embedding, agente)
e avvia il flusso di evaluation completo. Puo' essere eseguito come script
standalone dalla directory src/:

    python -m evaluation.eval_main --input data/evaluation/questions.json

Oppure importato e utilizzato programmaticamente:

    from evaluation.eval_main import run_evaluation
    report = run_evaluation(input_file="data/evaluation/questions.json")

Variabili d'ambiente rilevanti (oltre a quelle del sistema RAG):
    EVAL_JUDGE_API_KEY       - API key dedicata al Judge (o riusa GROQ_*)
    EVAL_JUDGE_PROVIDER      - Provider Judge (default: groq)
    EVAL_JUDGE_MODEL         - Modello Judge (default: llama-3.3-70b-versatile)
    EVAL_INPUT_FILE          - Percorso file JSON di input
    EVAL_OUTPUT_DIR          - Directory output report
    EVAL_BATCH_DELAY         - Pausa in secondi tra domande (rate limit)
    EVAL_MAX_RETRIES         - Max tentativi per chiamata Judge
    EVAL_RATE_LIMIT_WAIT     - Attesa su rate limit 429 (secondi)
"""

import sys
import os
import logging
import argparse
from typing import Optional, Dict, Any

# Assicura che src/ sia nel path per gli import relativi
_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from config.settings import load_settings
from config.logging_config import setup_logging

logger = logging.getLogger(__name__)


def _bootstrap_agent(settings, disable_guardrails: bool = False):
    """Inizializza tutti i componenti e assembla l'agente RAG.

    Replica la logica di bootstrap di agent_main.py per garantire
    compatibilita' completa con il sistema di produzione.

    Args:
        settings: Configurazione dell'applicazione.
        disable_guardrails: Se True, disabilita i guardrails durante
            l'evaluation per evitare blocchi sulle domande di test.

    Returns:
        Istanza di RAGAgent pronta per l'evaluation.
    """
    from agent.agent_main import build_retrieval_engine, build_embedding_model
    from agent.agent import RAGAgentFactory

    logger.info("[EVAL BOOT] Inizializzazione RetrievalEngine...")
    engine = build_retrieval_engine(settings)

    logger.info("[EVAL BOOT] Inizializzazione Embedding Model...")
    embedding_model = build_embedding_model(settings)

    logger.info("[EVAL BOOT] Assemblaggio agente RAG...")
    agent = RAGAgentFactory.create(
        retrieval_engine=engine,
        settings=settings,
        enable_scope_guardrail=not disable_guardrails,
        max_memory_turns=10,
        embedding_model=embedding_model,
    )

    logger.info("[EVAL BOOT] Agente RAG pronto per evaluation.")
    return agent


def run_evaluation(
    input_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    disable_guardrails: bool = False,
    log_level: str = "INFO",
) -> Dict[str, Any]:
    """Funzione principale per eseguire l'evaluation programmaticamente.

    Inizializza tutti i componenti, esegue il flusso completo di
    evaluation e restituisce il report.

    Args:
        input_file: Percorso del file JSON con domande e ground truth.
            Se None, usa il valore dalla configurazione o env.
        output_dir: Directory per i file di output. Se None, usa il default.
        disable_guardrails: Se True, disabilita i guardrails per evitare
            che blocchino domande di test legittime.
        log_level: Livello di logging (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Dizionario con il report completo dell'evaluation.

    Raises:
        FileNotFoundError: Se il file di input non esiste.
        ValueError: Se la configurazione non e' valida.
        RuntimeError: Se il bootstrap dei componenti fallisce.
    """
    # --- Configurazione logging ---
    settings = load_settings()
    setup_logging(
        level=log_level,
        log_file=settings.logging.log_file,
        log_to_console=settings.logging.log_to_console,
        log_format=settings.logging.log_format,
        date_format=settings.logging.date_format,
    )

    logger.info("=" * 60)
    logger.info("  AVVIO EVALUATION RAG DIEM")
    logger.info("=" * 60)

    # --- Configurazione evaluation ---
    from src.evaluation.config import load_evaluation_config
    eval_config = load_evaluation_config()

    # Override da parametri espliciti
    if input_file:
        eval_config = _override_config(eval_config, input_file=input_file)
    if output_dir:
        eval_config = _override_config(eval_config, output_dir=output_dir)

    # --- Bootstrap agente ---
    agent = _bootstrap_agent(settings, disable_guardrails=disable_guardrails)

    # --- Esecuzione evaluation ---
    from src.evaluation.runner import EvaluationRunner
    runner = EvaluationRunner(config=eval_config)

    report = runner.run(agent=agent, input_file=input_file)

    # --- Stampa riepilogo ---
    summary = runner.get_summary()
    print(summary)

    return report


def _override_config(config, input_file=None, output_dir=None):
    """Crea una nuova istanza di EvaluationConfig con override specifici.

    Poiche' EvaluationConfig e' frozen (immutabile), questa funzione
    ricrea l'oggetto con i valori modificati.

    Args:
        config: Configurazione originale.
        input_file: Nuovo percorso file di input (opzionale).
        output_dir: Nuova directory di output (opzionale).

    Returns:
        Nuova istanza di EvaluationConfig con gli override applicati.
    """
    from src.evaluation.config import EvaluationConfig, OutputConfig

    new_output = config.output
    if output_dir:
        new_output = OutputConfig(
            output_dir=output_dir,
            export_csv=config.output.export_csv,
            export_excel=config.output.export_excel,
            export_json=config.output.export_json,
            json_indent=config.output.json_indent,
        )

    return EvaluationConfig(
        judge=config.judge,
        retry=config.retry,
        metrics=config.metrics,
        output=new_output,
        pipeline=config.pipeline,
        input_file=input_file or config.input_file,
    )


def parse_args() -> argparse.Namespace:
    """Parsing degli argomenti da linea di comando.

    Returns:
        Namespace con gli argomenti parsati.
    """
    parser = argparse.ArgumentParser(
        description="Evaluation del sistema RAG DIEM con framework RAGAS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi di utilizzo:

  # Evaluation con file JSON di default
  python -m evaluation.eval_main

  # Evaluation con file JSON specifico
  python -m evaluation.eval_main --input data/evaluation/questions.json

  # Evaluation senza guardrails (per test)
  python -m evaluation.eval_main --input questions.json --no-guardrails

  # Evaluation con output in directory specifica
  python -m evaluation.eval_main --input questions.json --output results/run_01

  # Evaluation con logging verboso
  python -m evaluation.eval_main --input questions.json --log-level DEBUG
        """,
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Percorso del file JSON con domande e ground truth.",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Directory per i file di output (default: results/evaluation).",
    )

    parser.add_argument(
        "--no-guardrails",
        action="store_true",
        default=False,
        help="Disabilita i guardrails durante l'evaluation.",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Livello di logging (default: INFO).",
    )

    return parser.parse_args()


def main() -> None:
    """Entry point principale per l'esecuzione da terminale."""
    args = parse_args()

    try:
        report = run_evaluation(
            input_file=args.input,
            output_dir=args.output,
            disable_guardrails=args.no_guardrails,
            log_level=args.log_level,
        )

        # Stampa percorsi dei file generati
        output_files = report.get("metadata", {}).get("run", {}).get("output_files", {})
        if output_files:
            print("\nFile generati:")
            for fmt, path in output_files.items():
                print(f"  [{fmt.upper()}] {path}")

    except FileNotFoundError as e:
        print(f"\nERRORE: {e}", file=sys.stderr)
        print(
            "\nCrea il file JSON con la struttura:\n"
            '{\n'
            '    "dataset_name": "Nome del dataset",\n'
            '    "samples": [\n'
            '        {\n'
            '            "question": "Domanda...",\n'
            '            "ground_truth": "Risposta attesa..."\n'
            '        }\n'
            '    ]\n'
            '}',
            file=sys.stderr,
        )
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nEvaluation interrotta dall'utente.", file=sys.stderr)
        print(
            "I risultati intermedi sono salvati in results/evaluation/.intermediate/",
            file=sys.stderr,
        )
        sys.exit(130)

    except Exception as e:
        logger.error("Errore fatale: %s", e, exc_info=True)
        print(f"\nERRORE FATALE: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()