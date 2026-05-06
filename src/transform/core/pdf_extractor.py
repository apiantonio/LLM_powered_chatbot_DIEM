import os
import re
import logging
from typing import List, Set
from urllib.parse import urljoin, urldefrag
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from transform.core.base_rule import PdfFilterRule

logger = logging.getLogger(__name__)

class LocalPdfExtractorEngine:
    """Motore multithread che applica le strategie di pulizia sui file locali."""
    
    def __init__(self, input_dir: str, output_file: str, rules: List[PdfFilterRule]):
        self.input_dir = input_dir
        self.output_file = output_file
        self.rules = rules

    def _process_single_html(self, filepath: str) -> Set[str]:
        local_pdfs = set()
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
        except Exception as e:
            logger.error(f"Errore lettura file {filepath}: {e}")
            return local_pdfs

        # Estrazione dell'URL originale dai metadati salvati dal crawler
        source_match = re.search(r'<!--\s*SOURCE:\s*(https?://[^\s]+)\s*-->', html_content)
        if not source_match:
            return local_pdfs
            
        source_url = source_match.group(1).strip()
        soup = BeautifulSoup(html_content, "html.parser")
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href'].strip()
            clean_path = href.lower().split('?')[0] 
            
            if clean_path.endswith('.pdf'):
                # 1. Fix per URL relativi malformati di Ateneo
                if href.startswith('uploads/'):
                    href = '/' + href
                
                # 2. Risoluzione in URL assoluto
                full_url = urljoin(source_url, href)
                full_url, _ = urldefrag(full_url)
                
                # 3. Esecuzione del Pattern Strategy (Valutazione Regole)
                is_rejected = False
                for rule in self.rules:
                    if rule.should_discard(full_url):
                        is_rejected = True
                        break # Fallisce alla prima regola violata
                
                if not is_rejected:
                    local_pdfs.add(full_url)
                
        return local_pdfs

    def run(self):
        logger.info(f"--- AVVIO ESTRAZIONE PDF OFFLINE DA {self.input_dir} ---")
        
        if not os.path.exists(self.input_dir):
            logger.error(f"La cartella {self.input_dir} non esiste!")
            return

        html_files = [os.path.join(self.input_dir, f) for f in os.listdir(self.input_dir) if f.endswith('.html')]
        total_files = len(html_files)
        
        if total_files == 0:
            logger.warning("Nessun file HTML da analizzare.")
            return
            
        all_valid_pdfs = set()
        processed_count = 0
        cpu_count = os.cpu_count() or 1
        max_workers = min(32, (os.cpu_count() or 1) + 4)
        
        logger.info(f"Scansione di {total_files} file con {max_workers} thread. Regole attive: {len(self.rules)}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._process_single_html, fp): fp for fp in html_files}
            
            for future in as_completed(futures):
                processed_count += 1
                if processed_count % 1000 == 0:
                    logger.info(f"Progresso: {processed_count}/{total_files} file analizzati...")
                
                try:
                    valid_pdfs = future.result()
                    all_valid_pdfs.update(valid_pdfs)
                except Exception as e:
                    logger.error(f"Errore thread sul file {os.path.basename(futures[future])}: {e}")

        # Salvataggio
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True) if os.path.dirname(self.output_file) else None
        
        with open(self.output_file, "w", encoding="utf-8") as f:
            for link in sorted(all_valid_pdfs):
                f.write(f"{link}\n")

        logger.info("====== RESOCONTO FINALE ESTRAZIONE OFFLINE ======")
        logger.info(f"File HTML analizzati: {total_files}")
        logger.info(f"Link PDF Validati e Trovati: {len(all_valid_pdfs)}")
        logger.info(f"Salvato in: {self.output_file}")