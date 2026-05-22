"""Componenti di persistenza e post-processing del crawler.

Contiene:
- CrawlerPersistence: gestisce il salvataggio dei documenti HTML e del registro PDF su disco.
- PostProcessor: esegue le operazioni di pulizia post-crawling sui file HTML salvati.
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


class PostProcessor:
    """Esegue le operazioni di pulizia post-crawling sui file HTML salvati."""

    def __init__(self, output_dir: str, cutoff_year: int):
        """Inizializza il post-processor.

        Args:
            output_dir: Directory contenente i file HTML salvati.
            cutoff_year: Anno di cutoff per il filtraggio temporale.
        """
        self._output_path = Path(output_dir)
        self._cutoff_year = cutoff_year

    @staticmethod
    def _natural_sort_key(text: str):
        """Genera una chiave di ordinamento naturale per stringhe alfanumeriche."""
        return [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", text)
        ]

    def run_didattica_cleanup(self) -> int:
        """Rimuove le occorrenze ridondanti di pagine didattica.

        Elimina i file HTML di didattica che risultano superati da versioni
        con cId maggiore o che hanno figli (pId) piu specifici.

        Returns:
            Numero di file eliminati.
        """
        deleted_count = 0

        max_cid_per_anno_id = {}
        has_cid_per_anno_id = set()
        has_pid_per_anno_id_cid = set()

        pattern_anno = re.compile(r"anno=(\d+)", re.IGNORECASE)
        pattern_id = re.compile(r"id=(\d+)", re.IGNORECASE)
        pattern_cid = re.compile(r"cid=([^&.]+)", re.IGNORECASE)
        pattern_pid = re.compile(r"pid=([^&.]+)", re.IGNORECASE)

        for saved_file in self._output_path.glob("*.html"):
            filename = saved_file.name

            if "-didattica-" not in filename.lower():
                continue

            match_anno = pattern_anno.search(filename)
            match_id = pattern_id.search(filename)
            match_cid = pattern_cid.search(filename)
            match_pid = pattern_pid.search(filename)

            anno = match_anno.group(1) if match_anno else None
            current_id = match_id.group(1) if match_id else None
            current_cid = match_cid.group(1) if match_cid else None
            current_pid = match_pid.group(1) if match_pid else None

            if anno and current_id:
                key_anno_id = (anno, current_id)

                if current_cid:
                    has_cid_per_anno_id.add(key_anno_id)

                    existing_max_cid = max_cid_per_anno_id.get(key_anno_id)
                    if not existing_max_cid:
                        max_cid_per_anno_id[key_anno_id] = current_cid
                    else:
                        if self._natural_sort_key(current_cid) > self._natural_sort_key(
                            existing_max_cid
                        ):
                            max_cid_per_anno_id[key_anno_id] = current_cid

                    if current_pid:
                        has_pid_per_anno_id_cid.add((anno, current_id, current_cid))

        for saved_file in self._output_path.glob("*.html"):
            filename = saved_file.name

            if "-didattica-" not in filename.lower():
                continue

            match_anno = pattern_anno.search(filename)
            match_id = pattern_id.search(filename)
            match_cid = pattern_cid.search(filename)
            match_pid = pattern_pid.search(filename)

            anno = match_anno.group(1) if match_anno else None
            current_id = match_id.group(1) if match_id else None
            current_cid = match_cid.group(1) if match_cid else None
            current_pid = match_pid.group(1) if match_pid else None

            if anno and not current_id:
                continue

            if not (anno and current_id):
                continue

            should_delete = False
            reason = ""
            key_anno_id = (anno, current_id)

            if current_cid:
                max_cid = max_cid_per_anno_id.get(key_anno_id)

                if self._natural_sort_key(current_cid) < self._natural_sort_key(max_cid):
                    should_delete = True
                    reason = (
                        f"per anno={anno} e id={current_id} esiste un cId "
                        f"maggiore ({max_cid}). Questo ha cId={current_cid}."
                    )
                elif not current_pid:
                    if (anno, current_id, current_cid) in has_pid_per_anno_id_cid:
                        should_delete = True
                        reason = (
                            f"esiste un file Figlio (con pId) per l'anno={anno}, "
                            f"id={current_id} e cId={current_cid}."
                        )

            elif not current_cid and not current_pid:
                if key_anno_id in has_cid_per_anno_id:
                    should_delete = True
                    reason = (
                        f"esiste almeno un file Padre/Figlio (con cId) per "
                        f"l'anno={anno} e id={current_id}."
                    )

            if should_delete:
                try:
                    saved_file.unlink()
                    deleted_count += 1
                    logger.debug(
                        "Cleanup Didattica: rimosso '%s' -> MOTIVO: %s",
                        filename,
                        reason,
                    )
                except OSError as exc:
                    logger.warning("Errore eliminazione '%s': %s", filename, exc)

        logger.info(
            "Post-processing completato: rimosse %d occorrenze ridondanti.",
            deleted_count,
        )
        return deleted_count

    def run_anno_cleanup(self) -> int:
        """Rimuove file HTML con parametro anno anteriore al cutoff.

        I file con anno=0 sono preservati (convenzione per 'tutti gli anni').

        Returns:
            Numero di file eliminati.
        """
        deleted_count = 0
        anno_pattern = re.compile(r"-anno=(\d+)")

        for saved_file in self._output_path.glob("*.html"):
            filename = saved_file.name

            match = anno_pattern.search(filename)
            if not match:
                continue

            anno_str = match.group(1)
            try:
                anno_value = int(anno_str)
            except ValueError:
                continue

            if anno_value == 0:
                continue

            if anno_value < self._cutoff_year:
                try:
                    saved_file.unlink()
                    deleted_count += 1
                    logger.debug(
                        "Post-ingestion anno cleanup: eliminato %s (anno=%d < %d)",
                        filename,
                        anno_value,
                        self._cutoff_year,
                    )
                except OSError as exc:
                    logger.warning("Impossibile eliminare %s: %s", filename, exc)

        if deleted_count > 0:
            logger.info(
                "Post-ingestion anno cleanup: %d file eliminati (anno < %d, eccezione anno=0 preservata)",
                deleted_count,
                self._cutoff_year,
            )
        else:
            logger.info("Post-ingestion anno cleanup: nessun file da eliminare.")

        return deleted_count