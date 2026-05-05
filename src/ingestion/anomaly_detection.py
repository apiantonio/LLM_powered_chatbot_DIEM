import os
import re
import csv
import logging
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# CONFIGURAZIONE LOGGING E COSTANTI
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- PARAMETRIZZA QUI I TUOI VALORI ---
INPUT_DIR = "./data/raw/html_samples_v7"  # Sostituisci con la cartella che vuoi analizzare
OUTPUT_CSV = "./data/raw/html_samples_v7/report_anomalie_documenti.csv"

# Lista delle keyword da cercare (Case Insensitive)
KEYWORDS_TO_SEARCH = [
    "errore",
    "404",
    "nessun contenuto",
    "pagina non trovata",
    "accesso negato",
    "forbidden",
    "not found",
    "non autorizzato"
]

def process_single_file(filepath: str, keywords: list) -> dict:
    """
    Legge un singolo file HTML, ne estrae il testo pulito e cerca le keyword.
    Ritorna un dizionario con i risultati o None se nessuna keyword è presente.
    """
    filename = os.path.basename(filepath)
    
    # Estrazione dell'ID documento dal nome del file (es: doc_123_depth1_...)
    match = re.match(r"^doc_(\d+)_", filename)
    doc_id = match.group(1) if match else "N/D"

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
        
        # 1. Estrazione del solo testo visibile (Ignora tag, classi e attributi)
        soup = BeautifulSoup(html_content, 'html.parser')
        testo_visibile = soup.get_text(separator=' ', strip=True).lower()
        
        keyword_trovate = []
        
        # 2. Ricerca esatta delle keyword
        for kw in keywords:
            kw_lower = kw.lower()
            # Usiamo \b per i confini di parola (evita che 'error' matchi 'terror')
            # re.escape() protegge da eventuali caratteri speciali nelle keyword
            pattern = r'\b' + re.escape(kw_lower) + r'\b'
            
            if re.search(pattern, testo_visibile):
                keyword_trovate.append(kw)
                
        # 3. Se abbiamo trovato qualcosa, restituiamo il record
        if keyword_trovate:
            return {
                "ID_Documento": doc_id,
                "Keywords": ", ".join(keyword_trovate),
                "File_Name": filename
            }
            
    except Exception as e:
        logger.error(f"Errore nella lettura del file {filename}: {e}")
        
    return None

def scan_directory_for_keywords(input_dir: str, output_csv: str, keywords: list, max_workers=None):
    """
    Scansiona l'intera directory in parallelo e salva i risultati in un CSV.
    """
    if not os.path.exists(input_dir):
        logger.error(f"La cartella {input_dir} non esiste!")
        return

    # Raccoglie tutti i file HTML nella cartella
    html_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.html')]
    total_files = len(html_files)
    
    logger.info(f"Inizio scansione di {total_files} file. Keyword ricercate: {len(keywords)}")
    
    anomalies_found = []

    # Elaborazione multithread per massimizzare la velocità su migliaia di file
    workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Sottomette tutti i task
        futures = {executor.submit(process_single_file, filepath, keywords): filepath for filepath in html_files}
        
        processed = 0
        for future in as_completed(futures):
            processed += 1
            if processed % 1000 == 0:
                logger.info(f"Progresso: {processed}/{total_files} file analizzati...")
                
            result = future.result()
            if result:
                anomalies_found.append(result)

    # Scrittura del CSV finale
    if anomalies_found:
        logger.info(f"Trovati {len(anomalies_found)} documenti con anomalie. Scrittura CSV in corso...")
        
        # Ordina per ID_Documento (convertito in intero per ordinamento logico corretto)
        anomalies_found.sort(key=lambda x: int(x["ID_Documento"]) if x["ID_Documento"].isdigit() else 999999)
        
        try:
            with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
                fieldnames = ["ID_Documento", "Keywords", "File_Name"]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=',')
                
                writer.writeheader()
                for row in anomalies_found:
                    writer.writerow(row)
            logger.info(f"✅ Report completato e salvato in: {output_csv}")
        except Exception as e:
            logger.error(f"Errore durante il salvataggio del CSV: {e}")
    else:
        logger.info("🎉 Nessuna anomalia trovata! Tutti i documenti sembrano puliti.")

if __name__ == "__main__":
    scan_directory_for_keywords(
        input_dir=INPUT_DIR, 
        output_csv=OUTPUT_CSV, 
        keywords=KEYWORDS_TO_SEARCH
    )