import os
import re
import logging
from urllib.parse import urljoin, urldefrag
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin, urldefrag
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

INPUT_DIR = "data/raw/html_samples_v7_filtrato/html_samples_v7"   
OUTPUT_FILE = "data/raw/tutti_i_pdf_estratti_dai_file_locali.txt"


# Assicurati che questa costante sia definita all'inizio del file
ALLOWED_DOMAINS = {
    "www.diem.unisa.it",
    "docenti.unisa.it",
    "corsi.unisa.it"
}

def process_local_html(filepath: str) -> set:
    """
    Legge un file HTML locale, cerca il commento con l'URL sorgente,
    estrae tutti i link PDF e blocca rigorosamente quelli esterni.
    """
    local_pdfs = set()
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
    except Exception as e:
        logger.error(f"Impossibile leggere il file {filepath}: {e}")
        return local_pdfs

    source_match = re.search(r'<!--\s*SOURCE:\s*(https?://[^\s]+)\s*-->', html_content)
    
    if not source_match:
        logger.warning(f"URL sorgente mancante nel file: {os.path.basename(filepath)}")
        return local_pdfs
        
    source_url = source_match.group(1).strip()
    soup = BeautifulSoup(html_content, "html.parser")
    
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href'].strip()
        clean_path = href.lower().split('?')[0] 
        
        if clean_path.endswith('.pdf'):
            
            # 1. Fix percorsi relativi malformati
            if href.startswith('uploads/'):
                href = '/' + href
            
            # 2. Risoluzione dell'URL assoluto
            full_url = urljoin(source_url, href)
            full_url, _ = urldefrag(full_url)
            
            # 3. STRICT DOMAIN CHECK (Blocca i PDF esterni)
            try:
                parsed_url = urlparse(full_url)
                if parsed_url.netloc not in ALLOWED_DOMAINS:
                    continue  # Scarta immediatamente il PDF, è fuori perimetro
            except ValueError:
                continue # Scarta se l'URL non è parsabile
            
            local_pdfs.add(full_url)
            
    return local_pdfs

def run_offline_extraction():
    logger.info(f"--- AVVIO ESTRAZIONE PDF OFFLINE DA {INPUT_DIR} ---")
    
    if not os.path.exists(INPUT_DIR):
        logger.error(f"La cartella di input '{INPUT_DIR}' non esiste!")
        return

    # Lista di tutti i file HTML nella cartella
    html_files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith('.html')]
    total_files = len(html_files)
    
    if total_files == 0:
        logger.warning("Nessun file HTML trovato nella cartella specificata.")
        return
        
    logger.info(f"Trovati {total_files} file da analizzare. Calcolo in corso...")

    all_found_pdfs = set()
    processed_count = 0
    
    # Elaborazione multithread per processare migliaia di file in pochi secondi
    max_workers = min(32, (os.cpu_count() or 1) + 4)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_local_html, filepath): filepath for filepath in html_files}
        
        for future in as_completed(futures):
            processed_count += 1
            if processed_count % 1000 == 0:
                logger.info(f"Progresso: {processed_count}/{total_files} file analizzati...")
                
            try:
                extracted_pdfs = future.result()
                all_found_pdfs.update(extracted_pdfs)
            except Exception as e:
                filepath = futures[future]
                logger.error(f"Errore nell'elaborazione del thread per {os.path.basename(filepath)}: {e}")

    # Ordinamento e Salvataggio finale
    logger.info("Elaborazione completata. Salvataggio su file...")
    
    try:
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True) if os.path.dirname(OUTPUT_FILE) else None
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for link in sorted(all_found_pdfs):
                f.write(f"{link}\n")
    except Exception as e:
        logger.error(f"Impossibile salvare il file di output: {e}")
        return

    logger.info("====== RESOCONTO FINALE ESTRAZIONE OFFLINE ======")
    logger.info(f"File HTML analizzati: {total_files}")
    logger.info(f"Link PDF UNICI: {len(all_found_pdfs)}")
    logger.info(f"Salvato con successo in: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_offline_extraction()