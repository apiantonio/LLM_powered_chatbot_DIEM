"""Scheduler per l'aggiornamento periodico della Knowledge Base.

Orchestratore che esegue il ciclo completo o parziale della pipeline:
  1. Crawl (scraping incrementale HTML + raccolta link PDF)
  2. Index (chunking HTML/PDF/Markdown + embedding + Chroma upsert)
  3. Verify (verifica post-ingestion delle collezioni)

Modalita di esecuzione:
  - Solo scraping:           python -m ingestion.scheduler_main --mode scrape
  - Solo indicizzazione:     python -m ingestion.scheduler_main --mode index
  - Pipeline completa:       python -m ingestion.scheduler_main --mode full
  - Solo verifica:           python -m ingestion.scheduler_main --mode verify
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from config.settings import AppSettings, load_settings
from config.logging_config import setup_logging
from ingestion.indexer import KnowledgeBaseIndexer
from ingestion.router import CollectionTarget

logger = logging.getLogger(__name__)


class IngestionScheduler:
    """Facade che orchestra la pipeline di ingestion della Knowledge Base.

    Espone un'interfaccia semplificata per eseguire crawling,
    indicizzazione, verifica o la pipeline completa, delegando il
    lavoro effettivo a UnisaCrawler e KnowledgeBaseIndexer.
    """

    def __init__(self, settings: Optional[AppSettings] = None):
        """Inizializza lo scheduler con le impostazioni fornite o di default.

        Args:
            settings: Configurazione dell'applicazione. Se None, viene
                      caricata automaticamente tramite load_settings().
        """
        self._settings = settings or load_settings()
        self._indexer = KnowledgeBaseIndexer(self._settings)

    def run_crawl(self) -> dict:
        """Esegue la fase di crawling con le regole di filtraggio HTML e PDF.

        Le regole vengono costruite tramite RuleFactory e PdfRuleFactory
        a partire dai nomi configurati in IngestionConfig.

        Returns:
            Dizionario con statistiche del crawling o errore.
        """
        logger.info("[CRAWL] Avvio crawling...")
        try:
            from scraping.factories import RuleFactory, PdfRuleFactory
            from scraping.scrapers import UnisaCrawler

            ingestion_cfg = self._settings.ingestion
            output_dir = ingestion_cfg.html_raw_dir

            html_factory = RuleFactory(
                directory=Path(output_dir),
                cutoff_year=ingestion_cfg.cutoff_year,
                target_department=ingestion_cfg.target_department,
            )
            html_rules = html_factory.create_rules(list(ingestion_cfg.html_rule_names))

            pdf_factory = PdfRuleFactory(cutoff_year=ingestion_cfg.cutoff_year)
            pdf_rules = pdf_factory.create_rules(list(ingestion_cfg.pdf_rule_names))

            logger.info(
                "Regole caricate: %d HTML, %d PDF",
                len(html_rules), len(pdf_rules),
            )

            crawler = UnisaCrawler(
                max_depth=ingestion_cfg.max_depth,
                batch_size=ingestion_cfg.batch_size,
                delay=ingestion_cfg.crawl_delay_seconds,
                output_dir=output_dir,
                html_rules=html_rules,
                pdf_rules=pdf_rules,
                crawler_config=self._settings.crawler,
                ingestion_config=ingestion_cfg,
            )
            crawler.run()

            result = {
                "urls_visited": len(crawler.visited_urls),
                "html_saved": crawler.processed_count,
                "pdf_links_found": len(crawler.found_pdf_links),
            }
            logger.info("Crawling completato: %s", result)
            return result

        except Exception as e:
            logger.error("Errore durante il crawling: %s", e, exc_info=True)
            return {"error": str(e)}

    def run_indexing(self) -> dict:
        """Esegue la fase di indicizzazione incrementale HTML, PDF e Markdown.

        Returns:
            Dizionario con statistiche di indicizzazione per ciascun formato.
        """
        result = {
            "html_indexing": None,
            "pdf_indexing": None,
            "md_indexing": None,
        }

        try:
            logger.info("[INDEX] Indicizzazione HTML (incrementale)...")
            result["html_indexing"] = self._indexer.index_html_directory()
        except Exception as e:
            logger.error("Errore indicizzazione HTML: %s", e, exc_info=True)
            result["html_indexing"] = {"error": str(e)}

        try:
            logger.info("[INDEX] Indicizzazione Markdown (incrementale)...")
            result["md_indexing"] = self._indexer.index_markdown_directory()
        except Exception as e:
            logger.error("Errore indicizzazione Markdown: %s", e, exc_info=True)
            result["md_indexing"] = {"error": str(e)}

        try:
            logger.info("[INDEX] Indicizzazione PDF (incrementale)...")
            result["pdf_indexing"] = self._indexer.index_pdf_list()
        except Exception as e:
            logger.error("Errore indicizzazione PDF: %s", e, exc_info=True)
            result["pdf_indexing"] = {"error": str(e)}

        return result

    def run_verify(self, sample_per_collection: int = 3) -> dict:
        """Esegue la verifica post-ingestion delle collezioni.

        Args:
            sample_per_collection: Numero di documenti campione per collezione.

        Returns:
            Dizionario con conteggio chunk per collezione e stato complessivo.
        """
        logger.info("[VERIFY] Verifica delle collezioni...")
        verification = self._verify_collections()
        self._log_sample_documents(sample_per_collection)
        return verification

    def run_full_pipeline(self) -> dict:
        """Esegue la pipeline completa: crawling, indicizzazione e verifica.

        Returns:
            Dizionario con timestamp, statistiche complete e durata.
        """
        start_time = time.time()
        report = {
            "timestamp": datetime.now().isoformat(),
            "crawl": None,
            "html_indexing": None,
            "pdf_indexing": None,
            "md_indexing": None,
            "verification": None,
            "duration_seconds": 0,
            "success": False,
        }

        logger.info("=" * 60)
        logger.info("INGESTION PIPELINE -- Avvio: %s", report["timestamp"])
        logger.info("=" * 60)

        report["crawl"] = self.run_crawl()

        indexing_result = self.run_indexing()
        report["html_indexing"] = indexing_result["html_indexing"]
        report["pdf_indexing"] = indexing_result["pdf_indexing"]
        report["md_indexing"] = indexing_result["md_indexing"]

        report["verification"] = self.run_verify()

        report["duration_seconds"] = round(time.time() - start_time, 2)
        report["success"] = (
            report["verification"].get("ok", False)
            and self._no_critical_errors(report)
        )

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETATA in %ss", report["duration_seconds"])
        logger.info("HTML: %s", report["html_indexing"])
        logger.info("PDF:  %s", report["pdf_indexing"])
        logger.info("MD:   %s", report["md_indexing"])
        logger.info(
            "Stato: %s",
            "SUCCESSO" if report["success"] else "CON PROBLEMI",
        )
        logger.info("=" * 60)

        return report

    @property
    def indexer(self) -> KnowledgeBaseIndexer:
        """Restituisce l'istanza dell'indicizzatore sottostante."""
        return self._indexer

    def _verify_collections(self) -> Dict[str, Any]:
        """Verifica il conteggio dei chunk in ciascuna collezione del vector store.

        Returns:
            Dizionario con conteggio per collezione, totale e flag di stato.
        """
        verification: Dict[str, Any] = {
            "collections": {},
            "total_chunks": 0,
            "ok": True,
        }

        for target in CollectionTarget:
            collection = self._indexer._collections[target]
            try:
                count = collection._collection.count()
            except Exception as e:
                logger.error("Errore verifica %s: %s", target.value, e)
                count = 0
                verification["ok"] = False

            verification["collections"][target.value] = count
            verification["total_chunks"] += count
            logger.info("  %s: %d chunks", target.value, count)

        pc_collection_name = self._settings.vectorstore.parent_child_collection_name
        try:
            pc_count = self._indexer._pc_child_vectorstore._collection.count()
        except Exception as e:
            logger.error("Errore verifica Parent-Child: %s", e)
            pc_count = 0
            verification["ok"] = False

        verification["collections"][pc_collection_name] = pc_count
        verification["total_chunks"] += pc_count
        logger.info(
            "  %s (Parent-Child childs): %d chunks", pc_collection_name, pc_count
        )

        if verification["total_chunks"] == 0:
            verification["ok"] = False
            logger.warning("Nessun chunk indicizzato")

        return verification

    def _log_sample_documents(self, max_per_collection: int = 3) -> None:
        """Scrive nel log un campione di documenti per ciascuna collezione.

        Args:
            max_per_collection: Numero massimo di documenti campione per collezione.
        """
        logger.info("=" * 60)
        logger.info("CAMPIONE DOCUMENTI PER COLLECTION (verifica routing)")
        logger.info("=" * 60)

        for target in CollectionTarget:
            collection = self._indexer._collections[target]
            try:
                data = collection._collection.get(
                    limit=max_per_collection,
                    include=["metadatas", "documents"],
                )

                ids = data.get("ids", [])
                metadatas = data.get("metadatas", [])
                documents = data.get("documents", [])
                sample_size = len(ids)

                logger.info(
                    "%s (mostro %d chunks in sample):", target.value, sample_size
                )

                for i in range(sample_size):
                    meta = metadatas[i] if i < len(metadatas) else {}
                    content = (
                        documents[i][:120]
                        if i < len(documents) and documents[i]
                        else "(vuoto)"
                    )
                    source = meta.get(
                        "url_originale", meta.get("source_url", "N/D")
                    )
                    sotto_area = meta.get("sotto_area", "N/D")
                    formato = meta.get(
                        "formato_sorgente", meta.get("doc_type", "N/D")
                    )
                    logger.info(
                        "    [%d] formato=%s | sotto_area=%s | source: %s | "
                        "content: %s...",
                        i + 1, formato, sotto_area, source, content,
                    )
            except Exception as e:
                logger.warning("Errore lettura campione %s: %s", target.value, e)

    @staticmethod
    def _no_critical_errors(report: Dict[str, Any]) -> bool:
        """Verifica l'assenza di errori critici nel report di ingestion.

        Args:
            report: Dizionario di report della pipeline.

        Returns:
            True se nessuna sezione contiene errori critici.
        """
        for key in ("html_indexing", "pdf_indexing", "md_indexing"):
            section = report.get(key)
            if isinstance(section, dict) and "error" in section:
                return False
        return True


def _save_report(report: Dict[str, Any], output_path: str) -> None:
    """Persiste il report di ingestion su disco in formato JSON.

    Args:
        report: Dizionario di report da salvare.
        output_path: Percorso del file JSON di destinazione.
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Report salvato in: %s", output_path)
    except Exception as e:
        logger.error("Errore salvataggio report: %s", e)


def _build_argument_parser() -> argparse.ArgumentParser:
    """Costruisce il parser degli argomenti da riga di comando.

    Returns:
        ArgumentParser configurato con le opzioni di modalita e logging.
    """
    parser = argparse.ArgumentParser(
        description="Pipeline di ingestion della Knowledge Base DIEM.",
    )
    parser.add_argument(
        "--mode",
        choices=["scrape", "index", "full", "verify"],
        default="full",
        help=(
            "Modalita di esecuzione: "
            "'scrape' (solo crawling), "
            "'index' (solo indicizzazione), "
            "'full' (crawling + indicizzazione + verifica), "
            "'verify' (solo verifica delle collezioni)."
        ),
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help=(
            "Percorso del file di report JSON. "
            "Sovrascrive INGESTION_REPORT_PATH da .env/settings."
        ),
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help=(
            "Livello di log (DEBUG, INFO, WARNING, ERROR, CRITICAL). "
            "Sovrascrive LOG_LEVEL da .env/settings."
        ),
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Percorso del file di log. Sovrascrive LOG_FILE da .env/settings.",
    )
    parser.add_argument(
        "--no-console-log",
        action="store_true",
        default=False,
        help="Disabilita l'output dei log su console.",
    )
    return parser


def main() -> None:
    """Entry point unificato per la pipeline di ingestion."""
    parser = _build_argument_parser()
    args = parser.parse_args()

    settings = load_settings()

    log_level = args.log_level or settings.logging.level
    log_file = args.log_file or settings.logging.log_file
    log_to_console = not args.no_console_log and settings.logging.log_to_console

    setup_logging(
        level=log_level,
        log_file=log_file,
        log_to_console=log_to_console,
        log_format=settings.logging.log_format,
        date_format=settings.logging.date_format,
    )

    report_path = args.report_path or settings.observability.report_path
    scheduler = IngestionScheduler(settings)

    if args.mode == "scrape":
        logger.info("Modalita: solo scraping")
        result = scheduler.run_crawl()

    elif args.mode == "index":
        logger.info("Modalita: solo indicizzazione")
        result = scheduler.run_indexing()

    elif args.mode == "verify":
        logger.info("Modalita: solo verifica")
        result = scheduler.run_verify(sample_per_collection=5)

    else:
        logger.info("Modalita: pipeline completa (scraping + indicizzazione + verifica)")
        result = scheduler.run_full_pipeline()

    _save_report(result, report_path)
    logger.info("Report JSON: %s", report_path)


if __name__ == "__main__":
    main()