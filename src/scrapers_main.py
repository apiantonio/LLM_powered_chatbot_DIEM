"""Entry point per il processo di scraping UNISA.

Configura il logging e avvia il crawler con le regole di filtraggio
HTML e PDF specificate nella configurazione.
"""

from pathlib import Path

from config.logging_config import setup_logging
from config.settings import load_settings
from scraping.factory.cleaner_factory import RuleFactory
from scraping.factory.pdf_factory import PdfRuleFactory
from scraping.scrapers import UnisaCrawler

if __name__ == "__main__":
    settings = load_settings()

    setup_logging(
        level=settings.logging.level,
        log_file=settings.logging.log_file,
        log_to_console=settings.logging.log_to_console,
        log_format=settings.logging.log_format,
        date_format=settings.logging.date_format,
    )

    output_dir = settings.ingestion.html_raw_dir

    html_factory = RuleFactory(
        directory=Path(output_dir),
        cutoff_year=settings.ingestion.cutoff_year,
        target_department=settings.ingestion.target_department,
    )
    html_rules = html_factory.create_rules([
        "publication_tip",
        "exact_publications",
        "department_bandi",
        "calendar",
        "news",
        "404",
        "nocontent",
        "empty_body",
        "filename",
    ])

    pdf_factory = PdfRuleFactory(cutoff_year=settings.ingestion.cutoff_year)
    pdf_rules = pdf_factory.create_rules([
        "domain_whitelist",
        "semantic_trap",
        "obsolete_year",
        "english_pdf",
    ])

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
