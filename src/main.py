from pathlib import Path
from transform.factory.cleaner_factory import RuleFactory
from transform.core.engine import HTMLCleaner

def main():
    # 1. Configurazioni base
    DIRECTORY_PATH = "./data/raw/html_samples_v7_filtrato"
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
        report_filename="eliminazioni_news.txt"
    )
    cleaner.run()

if __name__ == "__main__":
    main()