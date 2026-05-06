import os
import re
import logging
import requests
from abc import ABC, abstractmethod
from typing import List, Set
from urllib.parse import urlparse, urljoin, urldefrag
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

INPUT_DIR = "data/raw/html_samples_cleaned"
OUTPUT_FILE = "data/raw/html_samples_cleaned/pdf_links_cleaned_filtred_3.txt"


class PdfCleaningRule(ABC):
    """Interfaccia Base per le regole di pulizia dei link PDF (Strategy Pattern)."""
    @abstractmethod
    def should_discard(self, url: str) -> bool:
        """Restituisce True se l'URL deve essere scartato, False altrimenti."""
        pass

class ObsoleteYearRule(PdfCleaningRule):
    """Scarta i PDF che appartengono ad anni precedenti al cutoff_year (es. 2020)."""
    def __init__(self, cutoff_year: int = 2020):
        self.cutoff_year = cutoff_year
        # Intercetta tutti gli anni a 4 cifre che iniziano con 19 o 20
        self.year_pattern = re.compile(r'\b(19\d{2}|20\d{2})\b')

    def should_discard(self, url: str) -> bool:
        matches = self.year_pattern.findall(url)
        if not matches:
            # Nessun anno trovato nel link, lo conserviamo
            return False 
        
        years = [int(y) for y in matches]
        # Calcoliamo l'anno più recente menzionato nell'URL.
        # Es: "guida-2018-2019.pdf" -> max è 2019. 2019 < 2020 -> Scartato.
        # Es: "bando-2019-2020.pdf" -> max è 2020. 2020 non è < 2020 -> Conservato.
        max_year_found = max(years)
        
        if max_year_found < self.cutoff_year:
            return True
            
        return False

class SemanticTrapRule(PdfCleaningRule):
    """Scarta i PDF puramente burocratici o contenenti dati sensibili in base a keyword."""
    def __init__(self):
        self.traps = [
            "grad_", "graduatori", "esit", "risultat", "ammess", "verbale", "verbali", 
            "decreto", "approvazione_atti", "commissione", "contratt", "incarico",
            "valutazione", "scorrimento", "elenco", "candidat", "modulo", "richiesta", "domanda"
        ]

    def should_discard(self, url: str) -> bool:
        url_lower = url.lower()
        return any(trap in url_lower for trap in self.traps)

class DomainWhitelistRule(PdfCleaningRule):
    """Assicura che il PDF non esca dal perimetro dei domini dell'Ateneo o da docenti estranei."""
    def __init__(self):
        self.allowed_domains = {"www.diem.unisa.it", "docenti.unisa.it", "corsi.unisa.it"}
        self.allowed_prefixes = (
            "https://www.diem.unisa.it",
            "https://docenti.unisa.it",
            "https://corsi.unisa.it",
            "https://corsi.unisa.it/ingegneria-informatica",
            "https://corsi.unisa.it/ingegneria-dell-informazione-per-la-medicina-digitale",
            "https://corsi.unisa.it/ingegneria-informatica-magistrale",
            "https://corsi.unisa.it/electrical-engineering-for-digital-energy",
            "https://corsi.unisa.it/information-Engineering-for-digital-medicine",
            "https://corsi.unisa.it/ingegneria-dell-informazione",
            "https://corsi.unisa.it/photovoltaics"
        )
        self.diem_docenti_whitelist = self._fetch_docenti_whitelist()

    def _fetch_docenti_whitelist(self) -> Set[str]:
        logger.info("Inizializzazione Whitelist Docenti DIEM dal web...")
        url_personale = "https://www.diem.unisa.it/dipartimento/personale"
        whitelist = set()
        try:
            response = requests.get(url_personale, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for a_tag in soup.find_all("a", href=True):
                if "rubrica.unisa.it/persone?matricola=" in a_tag['href']:
                    match = re.search(r'matricola=(\d+)', a_tag['href'])
                    if match:
                        whitelist.add(f"https://docenti.unisa.it/{match.group(1)}/")
        except Exception as e:
            logger.error(f"Impossibile creare whitelist docenti: {e}")
        return whitelist

    def should_discard(self, url: str) -> bool:
        try:
            parsed_url = urlparse(url)
            if parsed_url.netloc not in self.allowed_domains:
                return True
        except ValueError:
            return True

        if parsed_url.netloc == "docenti.unisa.it":
            if not any(url.startswith(doc) for doc in self.diem_docenti_whitelist):
                return True
        else:
            if not url.startswith(self.allowed_prefixes):
                return True

        return False


class LocalPdfExtractorEngine:
    """Motore multithread che applica le strategie di pulizia sui file locali."""
    
    def __init__(self, input_dir: str, output_file: str, rules: List[PdfCleaningRule]):
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
        print(f"CPU Count: {cpu_count}")
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


if __name__ == "__main__":
    
    # Istanziamo le strategie desiderate (Comportamento Plug & Play)
    active_rules = [
        DomainWhitelistRule(),     # Blocca uscite dal perimetro e docenti estranei
        SemanticTrapRule(),        # Blocca burocrazia, graduatorie, concorsi
        ObsoleteYearRule(2020)     # Blocca tutti i PDF precedenti al 2020
    ]
    
    # Iniezione delle dipendenze nell'Engine
    extractor = LocalPdfExtractorEngine(
        input_dir=INPUT_DIR,
        output_file=OUTPUT_FILE,
        rules=active_rules
    )
    
    extractor.run()