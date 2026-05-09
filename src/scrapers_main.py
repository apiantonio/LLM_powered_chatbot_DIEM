"""
Entry point per l'esecuzione del crawler con filtri docenti avanzati.

Uso:
  cd src/
  python scrapers_main.py
"""

from pathlib import Path
from ingestion.scrapers import UnisaCrawler
from transform.factory.cleaner_factory import RuleFactory
from transform.factory.pdf_factory import PdfRuleFactory
from config.settings import load_settings
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

if __name__ == "__main__":
    # --- Caricamento settings centralizzato ---
    settings = load_settings()

    output_dir = settings.ingestion.html_raw_dir

    # --- Costruzione regole con le Factory esistenti ---
    html_factory = RuleFactory(
        directory=Path(output_dir),
        cutoff_year=settings.ingestion.cutoff_year,
        target_department=settings.ingestion.target_department,
    )
    html_rules = html_factory.create_rules([
        "obsolete_url", "publication_tip", "exact_publications",
        "department_bandi", "calendar", "news",
        "404", "nocontent", "empty_body", "didattica", "filename",
    ])

    pdf_factory = PdfRuleFactory(cutoff_year=settings.ingestion.cutoff_year)
    pdf_rules = pdf_factory.create_rules([
        "domain_whitelist", "semantic_trap", "obsolete_year", "english_pdf"
    ])

    # --- Crawler con config centralizzata ---
    crawler = UnisaCrawler(
        max_depth=settings.ingestion.max_depth,
        batch_size=settings.ingestion.batch_size,
        delay=settings.ingestion.crawl_delay_seconds,
        output_dir=output_dir,
        html_rules=html_rules,
        pdf_rules=pdf_rules,
        crawler_config=settings.crawler,
        ingestion_config=settings.ingestion,
    )
    crawler.run()