"""
Scheduler per l'aggiornamento periodico della Knowledge Base.

Orchestratore cron-ready che esegue il ciclo completo:
  1. Crawl (scraping incrementale HTML + raccolta link PDF)
  2. Transform (pulizia HTML con regole Strategy, filtraggio PDF)
  3. Index (chunking duale HTML/PDF + embedding + Chroma upsert)

Modalità di esecuzione:
  - Manuale: python -m ingestion.scheduler
  - Cron job: 0 3 * * 0  cd /path/to/project && python -m ingestion.scheduler
    (ogni domenica alle 3:00 di notte)
  - APScheduler: per embedding in un processo long-running (es. il web server)

Pattern: Facade (GoF) — espone un'interfaccia semplice che orchestra
         l'intera pipeline di ingestion composta da più moduli.

KPI Impact: Knowledge Freshness (il vector store è sempre allineato ai siti).
"""

import logging
import time
from datetime import datetime
from typing import Optional

from config.settings import AppSettings, load_settings
from ingestion.indexer import KnowledgeBaseIndexer

logger = logging.getLogger(__name__)


class IngestionScheduler:
    
    def __init__(self, settings: Optional[AppSettings] = None):
        self._settings = settings or load_settings()
        self._indexer = KnowledgeBaseIndexer(self._settings)
    
    def run_full_pipeline(self, skip_crawl: bool = False) -> dict:
        start_time = time.time()
        report = {
            "timestamp": datetime.now().isoformat(),
            "crawl": None,
            "html_indexing": None,
            "pdf_indexing": None,
            "duration_seconds": 0,
        }
        
        logger.info("=" * 60)
        logger.info(f"INGESTION PIPELINE — Avvio: {report['timestamp']}")
        logger.info("=" * 60)
        
        if not skip_crawl:
            try:
                logger.info("[FASE 1/3] Avvio crawling...")
                from src.scraping.scrapers import UnisaCrawler
                
                crawler = UnisaCrawler(
                    max_depth=self._settings.ingestion.max_depth,
                    batch_size=self._settings.ingestion.batch_size,
                    delay=self._settings.ingestion.crawl_delay_seconds,
                    output_dir=self._settings.ingestion.html_raw_dir,
                )
                crawler.run()
                
                report["crawl"] = {
                    "urls_visited": len(crawler.visited_urls),
                    "html_saved": crawler.processed_count,
                    "pdf_links_found": len(crawler.found_pdf_links),
                }
                logger.info(f"Crawling completato: {report['crawl']}")
                
            except Exception as e:
                logger.error(f"Errore durante il crawling: {e}")
                report["crawl"] = {"error": str(e)}
        else:
            logger.info("[FASE 1/3] Crawling saltato (skip_crawl=True)")
        
        try:
            logger.info("[FASE 2/3] Indicizzazione HTML (incrementale)...")
            html_stats = self._indexer.index_html_directory()
            report["html_indexing"] = html_stats
        except Exception as e:
            logger.error(f"Errore indicizzazione HTML: {e}")
            report["html_indexing"] = {"error": str(e)}
        
        try:
            logger.info("[FASE 3/3] Indicizzazione PDF Parent-Child (incrementale)...")
            pdf_stats = self._indexer.index_pdf_list()
            report["pdf_indexing"] = pdf_stats
        except Exception as e:
            logger.error(f"Errore indicizzazione PDF: {e}")
            report["pdf_indexing"] = {"error": str(e)}
        
        report["duration_seconds"] = round(time.time() - start_time, 2)
        
        logger.info("=" * 60)
        logger.info(f"PIPELINE COMPLETATA in {report['duration_seconds']}s")
        logger.info(f"HTML: {report['html_indexing']}")
        logger.info(f"PDF:  {report['pdf_indexing']}")
        logger.info("=" * 60)
        
        return report
    
    @property
    def indexer(self) -> KnowledgeBaseIndexer:
        return self._indexer


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    settings = load_settings()
    scheduler = IngestionScheduler(settings)
    
    report = scheduler.run_full_pipeline(skip_crawl=True)
    
    # Per esecuzione completa (crawl + index):
    # report = scheduler.run_full_pipeline(skip_crawl=False)
    
    import json
    print(json.dumps(report, indent=2, ensure_ascii=False))
