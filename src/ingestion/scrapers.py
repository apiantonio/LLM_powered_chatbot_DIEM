import re
import os
import time
import requests
import logging
from collections import deque
from urllib.parse import urlparse, urljoin, urldefrag
from bs4 import BeautifulSoup
from langchain_community.document_loaders import AsyncHtmlLoader
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# CONFIGURAZIONE LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==========================================
# CLASSE PRINCIPALE DEL CRAWLER
# ==========================================
class UnisaCrawler:
    """
    Crawler BFS asincrono e multithread orientato all'estrazione di contenuti
    e file PDF dai domini UNISA, con salvataggio incrementale e whitelisting rigoroso.
    """
    
    ALLOWED_DOMAINS = {
        "www.diem.unisa.it",
        "docenti.unisa.it",
        "corsi.unisa.it"
    }

    ALLOWED_PREFIXES = (
        "https://www.diem.unisa.it",
        "https://corsi.unisa.it/ingegneria-informatica",
        "https://corsi.unisa.it/ingegneria-dell-informazione-per-la-medicina-digitale",
        "https://corsi.unisa.it/ingegneria-informatica-magistrale",
        "https://corsi.unisa.it/electrical-engineering-for-digital-energy",
        "https://corsi.unisa.it/information-Engineering-for-digital-medicine",
        "https://corsi.unisa.it/ingegneria-dell-informazione",
        "https://corsi.unisa.it/photovoltaics"
    )

    IGNORED_EXTENSIONS = (
        '.css', '.js', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', 
        '.zip', '.rar', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx'
    )

    def __init__(self, max_depth=3, batch_size=50, delay=2.0, max_workers=None, output_dir="data/raw/html_samples_final"):
        self.max_depth = max_depth
        self.batch_size = batch_size
        self.delay = delay
        
        # Gestione ottimizzata dei thread per CPU-bound tasks (BeautifulSoup)
        if max_workers is None:
            self.max_workers = min(32, (os.cpu_count() or 1) + 4)
        else:
            self.max_workers = max_workers
            
        self.output_dir = output_dir
        
        # Strutture dati incapsulate
        self.visited_urls = set()
        self.found_pdf_links = set()
        self.diem_docenti_whitelist = set()
        self.queue = deque()
        self.processed_count = 0
        
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Crawler inizializzato (Max Workers: {self.max_workers}, Batch: {self.batch_size})")

    # ==========================================
    # DISCOVERY E VALIDAZIONE (GATEKEEPER)
    # ==========================================

    def initialize_diem_docenti_whitelist(self):
        """Scansiona la pagina del personale DIEM per creare la whitelist e i seed."""
        url_personale = "https://www.diem.unisa.it/dipartimento/personale"
        logger.info("Inizializzazione Whitelist Docenti DIEM...")
        
        try:
            response = requests.get(url_personale, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            
            for a_tag in soup.find_all("a", href=True):
                href = a_tag['href']
                if "rubrica.unisa.it/persone?matricola=" in href:
                    match = re.search(r'matricola=(\d+)', href)
                    if match:
                        matricola = match.group(1)
                        base_docente_url = f"https://docenti.unisa.it/{matricola}/"
                        self.diem_docenti_whitelist.add(base_docente_url)
                        
                        # Accoda subito il seed
                        home_url = f"{base_docente_url}home"
                        self.queue.append((home_url, 0))
                        self.visited_urls.add(home_url)
                        
            logger.info(f"Trovati {len(self.diem_docenti_whitelist)} docenti afferenti al DIEM.")
        except Exception as e:
            logger.error(f"Impossibile inizializzare la whitelist dei docenti: {e}")

    def is_valid_url(self, url: str) -> bool:
        """Verifica perimetro, whitelist docenti e pattern tossici (senza limiti di anno)."""
        # 1. Strict Domain Check
        try:
            parsed_url = urlparse(url)
            if parsed_url.netloc not in self.ALLOWED_DOMAINS:
                return False
        except ValueError:
            return False

        # 2. Controllo Docenti (Blocca professori di altri dipartimenti)
        if parsed_url.netloc == "docenti.unisa.it":
            is_diem_docente = any(url.startswith(doc) for doc in self.diem_docenti_whitelist)
            if not is_diem_docente:
                return False
        else:
            if not url.startswith(self.ALLOWED_PREFIXES):
                return False
        
        url_lower = url.lower()
        
        # 3. Estensioni multimediali
        if any(url_lower.endswith(ext) for ext in self.IGNORED_EXTENSIONS):
            return False
            
        # 4. Blacklist Statica HTML
        html_traps = [
            "sitemap", ".xml", "rss", "feed", "unisa-rescue-page",
            "jsessionid", "saml2-redirect", "password-recovery", "choose-spid",
            "avviso=", "avvisi=", "&p=", "news-archive=", "eventi-archive=", 
            "category=", "news-category=", "idconcorso=", "commissioni-dettaglio=",
            "calendario-occupazione-spazi-sede", 
        ]
        if any(trap in url_lower for trap in html_traps):
            return False

        # 5. Blacklist Semantica PDF (Esclude verbali, esiti e graduatorie)
        if url_lower.endswith('.pdf'):
            pdf_traps = [
                "grad_", "graduatori", "esit", "risultat", "ammess", "verbale", "verbali", 
                "decreto", "approvazione_atti", "commissione", "contratt", "incarico",
                "valutazione", "scorrimento", "elenco", "candidat", "modulo", "richiesta", "domanda"
            ]
            if any(trap in url_lower for trap in pdf_traps):
                return False

        return True

    # ==========================================
    # PARSING THREAD-SAFE
    # ==========================================

    def process_single_html(self, html_content: str, source_url: str):
        """
        Parsing e pulizia DOM. Funzione STATELESS per permettere 
        l'esecuzione sicura in ThreadPoolExecutor.
        """
        soup = BeautifulSoup(html_content, "html.parser")
        local_new_links = set()
        local_pdfs = set()
        
        # 1. Estrazione dei link PRIMA della pulizia del DOM
        for a_tag in soup.find_all('a', href=True):
            full_url = urljoin(source_url, a_tag['href'])
            full_url, _ = urldefrag(full_url)
            
            if self.is_valid_url(full_url):
                if full_url.lower().endswith('.pdf'):
                    local_pdfs.add(full_url)
                else:
                    local_new_links.add(full_url)

        # 2. Pulizia chirurgica del rumore per estrarre solo il testo utile
        noise_selectors = [
            "#header", "#main-menu", "#menu-bar", "#unisa-left-menu", 
            "#box-agenda", "#share-dropdown", ".breadcrumb", 
            ".sr-only", "div[id$='-map']", "script", "style", "noscript"
        ]
        for selector in noise_selectors:
            for element in soup.select(selector):
                element.decompose()
                
        # 3. Estrazione Main Content
        main_content = (soup.find(id="unisa-content") or 
                        soup.find(attrs={"role": "main"}) or 
                        soup.find(id="content") or 
                        soup.find("main") or soup.body)
                       
        clean_html = ""
        if main_content:
            allowed_attrs = ['href', 'src', 'colspan', 'rowspan']
            for tag in main_content.find_all(True):
                tag.attrs = {key: value for key, value in tag.attrs.items() if key in allowed_attrs}
            clean_html = main_content.decode_contents().strip()
            
        return clean_html, local_new_links, local_pdfs

    # ==========================================
    # SALVATAGGIO INCREMENTALE
    # ==========================================

    def _save_single_doc(self, doc, current_depth):
        raw_url = doc.metadata.get('source', 'URL_sconosciuto')
        safe_name = re.sub(r'[<>:"/\\|?*]', '-', raw_url.replace("https://", "").replace("http://", ""))
        safe_name = safe_name.replace("__", "_")[:150] # Troncamento di sicurezza per nomi lunghi
        
        filepath = os.path.join(self.output_dir, f"doc{self.processed_count}_depth{current_depth}_{safe_name}.html")
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"<meta charset='utf-8'>\n"
                        f"<!-- SOURCE: {raw_url} -->\n"
                        f"<!-- DEPTH: {current_depth} -->\n")
                f.write(doc.page_content)
        except Exception as e:
            logger.error(f"Errore scrittura file {filepath}: {e}")

    def _save_pdf_ledger(self):
        if not self.found_pdf_links:
            return
            
        pdf_path = os.path.join(self.output_dir, "pdf_links.txt")
        try:
            with open(pdf_path, "w", encoding="utf-8") as f:
                f.write("\n".join(sorted(self.found_pdf_links)))
        except Exception as e:
            logger.error(f"Errore scrittura registro PDF: {e}")

    # ==========================================
    # MOTORE ORCHESTRATORE
    # ==========================================

    def run(self):
        logger.info("--- AVVIO INGESTION DINAMICA ---")
        
        # 1. Popolamento Seed di base
        base_seeds = [url for url in self.ALLOWED_PREFIXES if url != "https://docenti.unisa.it"]
        for url in base_seeds:
            self.queue.append((url, 0))
            self.visited_urls.add(url)
            
        # 2. Popolamento Seed Docenti
        self.initialize_diem_docenti_whitelist()

        # 3. Ciclo BFS Principale
        while self.queue:
            batch = []
            
            # Estrazione sicura dalla coda
            while self.queue and len(batch) < self.batch_size:
                url, depth = self.queue.popleft()
                if depth <= self.max_depth:
                    batch.append((url, depth))
            
            if not batch:
                continue
                
            urls_to_fetch = [item[0] for item in batch]
            depths_dict = dict(batch)
            
            logger.info(f"Download asincrono batch: {len(urls_to_fetch)} URL (Coda: {len(self.queue)} | Estratti: {self.processed_count})")
            
            # Download asincrono (I/O bound)
            loader = AsyncHtmlLoader(urls_to_fetch, ignore_load_errors=True)
            raw_docs = loader.load()

            # Parsing e pulizia DOM in parallelo (CPU bound)
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.process_single_html, doc.page_content, doc.metadata.get('source')): doc 
                    for doc in raw_docs if doc.metadata.get('source')
                }
                
                for future in as_completed(futures):
                    doc = futures[future]
                    source_url = doc.metadata.get('source')
                    current_depth = depths_dict.get(source_url, 0)
                    
                    try:
                        clean_html, local_new_links, local_pdfs = future.result()
                        
                        # Aggiornamento stato globale (i set in python supportano update in thread principali)
                        self.found_pdf_links.update(local_pdfs)
                        
                        if clean_html:
                            doc.page_content = clean_html
                            self._save_single_doc(doc, current_depth)
                            self.processed_count += 1
                            
                        # Accoda i nuovi link
                        if current_depth < self.max_depth:
                            for new_link in local_new_links:
                                if new_link not in self.visited_urls:
                                    self.queue.append((new_link, current_depth + 1))
                                    self.visited_urls.add(new_link)
                                    
                    except Exception as e:
                        logger.error(f"Errore nell'elaborazione di {source_url}: {e}")

            # Sincronizza stato PDF e rispetta il server
            self._save_pdf_ledger()
            time.sleep(self.delay)

        self._print_summary()

    def _print_summary(self):
        logger.info("====== RESOCONTO FINALE ======")
        logger.info(f"URL totali visitati o accodati: {len(self.visited_urls)}")
        logger.info(f"Documenti HTML validi salvati su disco: {self.processed_count}")
        logger.info(f"Link PDF unici intercettati: {len(self.found_pdf_links)}")

# ==========================================
# ESECUZIONE
# ==========================================
if __name__ == "__main__":
    # Parametri ottimizzati per stabilità e prestazioni
    crawler = UnisaCrawler(
        max_depth=5, 
        batch_size=1024,
        max_workers=None, 
        output_dir="data/raw/html_samples_v7"
    )
    
    crawler.run()