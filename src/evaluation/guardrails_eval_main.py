"""Entry point CLI per la valutazione dei guardrails del sistema RAG DIEM.

Eseguibile dalla root del progetto:
    python -m src.evaluation.guardrails_eval_main

Oppure con parametri opzionali:
    python -m src.evaluation.guardrails_eval_main --input data/evaluation/guardrails_testset.json
    python -m src.evaluation.guardrails_eval_main --log-level DEBUG
"""

import sys
import os
import logging
import argparse

_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from src.config.settings import load_settings
from src.config.logging_config import setup_logging
from src.evaluation.guardrails_eval import GuardrailsEvaluationRunner

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluation dei Guardrails del sistema RAG DIEM",
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Percorso del file JSON con il testset (default: data/evaluation/guardrails_testset.json).",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Livello di logging (default: INFO).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()

    setup_logging(
        level=args.log_level,
        log_file=settings.logging.log_file,
        log_to_console=settings.logging.log_to_console,
        log_format=settings.logging.log_format,
        date_format=settings.logging.date_format,
    )

    logger.info("=" * 60)
    logger.info("  AVVIO EVALUATION GUARDRAILS RAG DIEM")
    logger.info("=" * 60)

    testset_path = args.input or "data/evaluation/guardrails_testset.json"

    try:
        runner = GuardrailsEvaluationRunner(testset_path=testset_path)
        runner.run()

    except FileNotFoundError as e:
        print(f"\nERRORE: {e}", file=sys.stderr)
        sys.exit(1)

    except RuntimeError as e:
        print(f"\nERRORE: {e}", file=sys.stderr)
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nEvaluation interrotta dall'utente.", file=sys.stderr)
        sys.exit(130)

    except Exception as e:
        logger.error("Errore fatale: %s", e, exc_info=True)
        print(f"\nERRORE FATALE: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()