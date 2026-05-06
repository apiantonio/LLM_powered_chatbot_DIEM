from pathlib import Path
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
    main()