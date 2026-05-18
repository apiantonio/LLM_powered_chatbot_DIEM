"""Componente responsabile della persistenza dei documenti HTML e del registro PDF.

Separa la logica di I/O su filesystem dal crawler, migliorando la
Separation of Concerns e la testabilita.
"""

import logging
import os
import re
import threading
from pathlib import Path
from typing import Optional, Set

logger = logging.getLogger(__name__)


class CrawlerPersistence:
    """Gestisce il salvataggio dei documenti HTML e del ledger PDF su disco."""

    def __init__(self, output_dir: str):
        """Inizializza il componente di persistenza.

        Args:
            output_dir: Directory di destinazione per i file salvati.
        """
        self.output_dir = output_dir
        self._write_lock = threading.Lock()
        os.makedirs(self.output_dir, exist_ok=True)

    def save_html_document(
        self, doc, current_depth: int, doc_index: int
    ) -> Optional[str]:
        """Salva un documento HTML su disco con metadati di provenienza.

        Args:
            doc: Documento con page_content e metadata['source'].
            current_depth: Profondita di crawling corrente.
            doc_index: Indice progressivo del documento.

        Returns:
            Percorso del file salvato, oppure None in caso di errore.
        """
        raw_url = doc.metadata.get("source", "URL_sconosciuto")
        safe_name = re.sub(
            r'[<>:"/\\|?*]',
            "-",
            raw_url.replace("https://", "").replace("http://", ""),
        )
        safe_name = safe_name.replace("__", "_")
        filepath = os.path.join(
            self.output_dir,
            f"doc{doc_index}_depth{current_depth}_{safe_name}.html",
        )

        with self._write_lock:
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(
                        f"<meta charset='utf-8'>\n"
                        f"<!-- SOURCE: {raw_url} -->\n"
                        f"<!-- DEPTH: {current_depth} -->\n"
                    )
                    f.write(doc.page_content)
                return filepath
            except Exception as exc:
                logger.error("Errore scrittura %s: %s", filepath, exc)
                return None

    def save_pdf_ledger(self, pdf_links: Set[str]) -> None:
        """Salva il registro dei link PDF validi su file.

        Args:
            pdf_links: Insieme di URL PDF da persistere.
        """
        if not pdf_links:
            return
        pdf_path = os.path.join(self.output_dir, "pdf_links.txt")
        with self._write_lock:
            try:
                with open(pdf_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(sorted(pdf_links)))
            except Exception as exc:
                logger.error("Errore scrittura registro PDF: %s", exc)
