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

def main():
    # 1. Configurazioni base
    DIRECTORY_PATH = "./data/raw/html_samples_v7"
    directory = Path(DIRECTORY_PATH)
    
    # 2. Definisci A RUNTIME quali regole vuoi applicare
    # Puoi cambiare questa lista senza toccare la logica delle classi!
    active_rules = [
        "filename",
        "didattica",
        "obsolete_url",
        "publication_tip",
        "exact_publications",
        "department_bandi",
        "calendar",
        "news",
        "404",        # Decommenta per attivarla
        "nocontent",    # Decommenta per attivarla
        "empty_body"    # Decommenta per attivarla
    ]

    # 3. Usa la Factory per costruire le dipendenze
    factory = RuleFactory(directory=directory, cutoff_year=2020)
    rules_to_apply = factory.create_rules(active_rules)

    # 4. Inietta le regole nel Motore ed esegui
    cleaner = HTMLCleaner(
        directory=DIRECTORY_PATH, 
        rules=rules_to_apply, 
        report_filename="eliminazioni_bandi.txt"
    )
    cleaner.run()
    
    # ==========================================
    # FASE 2: ESTRAZIONE DEI LINK PDF
    # ==========================================
    DIRECTORY_PATH = "./data/raw/html_samples_v7"
    directory = Path(DIRECTORY_PATH)
    PDF_OUTPUT_FILE = directory / "pdf_links_cleaned_new.txt"
    CUTOFF_YEAR = 2020
    
    print("\n" + "="*40 + "\n")

    
    # Specifica qui quali filtri PDF applicare a runtime
    active_pdf_filters = [
        "domain_whitelist", 
        "semantic_trap", 
        "obsolete_year"
    ]
    
    pdf_factory = PdfRuleFactory(cutoff_year=CUTOFF_YEAR)
    pdf_rules = pdf_factory.create_rules(active_pdf_filters)
    
    extractor = LocalPdfExtractorEngine(
        input_dir=DIRECTORY_PATH, 
        output_file=str(PDF_OUTPUT_FILE),
        rules=pdf_rules
    )
    extractor.run()

    print("\n=== PIPELINE COMPLETATA CON SUCCESSO ===")
    

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
        output_dir="./data/raw/html_samples_claude",
        html_rules=html_rules,
        pdf_rules=pdf_rules,
    )
    crawler.run()