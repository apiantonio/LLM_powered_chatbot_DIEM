"""Configurazione centralizzata del logging per l'applicazione."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_to_console: bool = True,
    log_format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    date_format: str = "%Y-%m-%d %H:%M:%S",
) -> None:
    """Inizializza il sistema di logging con livello, destinazione e formato configurabili.

    Args:
        level: Livello di verbosita (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Percorso opzionale del file di log. Se None, nessun file viene scritto.
        log_to_console: Se True, i log vengono stampati anche su terminale.
        log_format: Formato delle righe di log.
        date_format: Formato del timestamp.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.handlers.clear()

    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    if not log_to_console and not log_file:
        root_logger.addHandler(logging.NullHandler())