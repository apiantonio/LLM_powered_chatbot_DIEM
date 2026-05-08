from pathlib import Path
from ingestion.scrapers import UnisaCrawler
from transform.factory.cleaner_factory import RuleFactory
from transform.core.html_cleaner import HTMLCleaner
from transform.factory.pdf_factory import PdfRuleFactory
from transform.core.pdf_extractor import LocalPdfExtractorEngine
import logging

logging.basicConfig(
    level=logging.INFO, # Forza Python a mostrare tutti i logger.info()
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)   

if __name__ == "__main__":
    from transform.factory.cleaner_factory import RuleFactory
    from transform.factory.pdf_factory import PdfRuleFactory
    
    # Costruzione regole con le Factory esistenti
    html_factory = RuleFactory(
        directory=Path("data/raw/html_samples"),
        cutoff_year=2020,
    )
    html_rules = html_factory.create_rules([
        "obsolete_url", "publication_tip", "exact_publications",
        "department_bandi", "calendar", "news",
        "404", "nocontent", "empty_body", "didattica", "filename",
    ])
    # NOTA: "didattica" e "filename" escluse — "didattica" richiede
    # pre-scansione della directory (non compatibile con filtro inline),
    # "filename" opera su nomi file del crawler (non su URL).
    
    pdf_factory = PdfRuleFactory(cutoff_year=2020)
    pdf_rules = pdf_factory.create_rules([
        "domain_whitelist", "semantic_trap", "obsolete_year",
    ])
    
    crawler = UnisaCrawler(
        max_depth=5,
        batch_size=1024,
        output_dir="./data/raw/html_aggiornato_claude",
        html_rules=html_rules,
        pdf_rules=pdf_rules,
    )
    crawler.run()