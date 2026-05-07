"""
UnisaCrawler refactorato con filtri inline.

CAMBIAMENTO ARCHITETTURALE:
  PRIMA: Crawl → salva tutto su disco → main.py applica HTMLCleaner a posteriori
  DOPO:  Crawl → filtra in-memory → salva SOLO i file che superano i filtri

  Per i PDF:
  PRIMA: I link PDF vengono raccolti da TUTTE le pagine HTML visitate
  DOPO:  I link PDF vengono raccolti SOLO dalle pagine HTML che superano i filtri
         Se la pagina HTML genitore viene scartata, i suoi PDF vengono ignorati.

Motivazione:
  - Elimina il passaggio a posteriori (main.py + HTMLCleaner + LocalPdfExtractorEngine).
  - Riduce I/O su disco (i file scartati non vengono mai scritti).
  - Garantisce coerenza: un PDF è rilevante SE E SOLO SE la sua pagina genitore lo è.

Pattern: Strategy (i filtri sono iniettati come liste di regole, come prima)
         ma ora operano in-memory durante il crawling, non su file salvati.

Le regole della cartella transform/ rimangono invariate nelle loro interfacce.
Solo il punto di applicazione cambia: da file su disco a contenuto in-memory.
"""

import re
import os
import time
import requests
import logging
from collections import deque
from pathlib import Path
from urllib.parse import urlparse, urljoin, urldefrag
from bs4 import BeautifulSoup
from langchain_community.document_loaders import AsyncHtmlLoader
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Set, Tuple

from transform.core.base_rule import CleaningRule, PdfFilterRule

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class UnisaCrawler:
    """
    Crawler BFS con filtri inline integrati.
    
    I file HTML vengono valutati in-memory PRIMA del salvataggio su disco.
    I link PDF vengono raccolti SOLO dalle pagine che superano i filtri HTML.
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

    def __init__(
        self,
        max_depth: int = 3,
        batch_size: int = 50,
        delay: float = 2.0,
        max_workers: Optional[int] = None,
        output_dir: str = "data/raw/html_samples",
        html_rules: Optional[List[CleaningRule]] = None,
        pdf_rules: Optional[List[PdfFilterRule]] = None,
    ):
        self.max_depth = max_depth
        self.batch_size = batch_size
        self.delay = delay
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.output_dir = output_dir
        
        # Filtri iniettati (Strategy Pattern)
        self._html_rules = html_rules or []
        self._pdf_rules = pdf_rules or []
        
        # Stato
        self.visited_urls: Set[str] = set()
        self.found_pdf_links: Set[str] = set()
        self.diem_docenti_whitelist: Set[str] = set()
        self.queue = deque()
        self.processed_count = 0
        self.filtered_count = 0
        
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(
            f"Crawler inizializzato — Workers: {self.max_workers}, "
            f"HTML rules: {len(self._html_rules)}, PDF rules: {len(self._pdf_rules)}"
        )

    # ==========================================
    # FILTRI INLINE
    # ==========================================

    def _should_discard_html(self, source_url: str, clean_html: str) -> Tuple[bool, str]:
        """
        Valuta il contenuto HTML in-memory contro tutte le regole attive.
        
        Adatta le regole CleaningRule (progettate per file su disco) al contesto
        in-memory: crea un Path fittizio dal source_url per le regole filename-based,
        e passa il contenuto per le regole content-based.
        
        Returns:
            (should_discard, reason)
        """
        # Costruisci un Path fittizio per le regole che operano sul nome file
        safe_name = re.sub(
            r'[<>:"/\\|?*]', '-',
            source_url.replace("https://", "").replace("http://", "")
        )
        fake_path = Path(f"{safe_name}.html")
        
        for rule in self._html_rules:
            try:
                if not rule.requires_content:
                    # Regole filename-based (ObsoleteUrl, Filename, PublicationTip, etc.)
                    if rule.should_delete(fake_path):
                        return True, rule.name
                else:
                    # Regole content-based (EmptyBody, NoContent, 404)
                    if rule.should_delete(fake_path, clean_html):
                        return True, rule.name
            except Exception as e:
                logger.debug(f"Regola {rule.name} ha sollevato eccezione: {e}")
                continue
        
        return False, ""

    def _should_discard_pdf(self, pdf_url: str) -> bool:
        """Valuta un link PDF contro le regole PDF attive."""
        for rule in self._pdf_rules:
            if rule.should_discard(pdf_url):
                return True
        return False

    # ==========================================
    # DISCOVERY E VALIDAZIONE URL
    # ==========================================

    def initialize_diem_docenti_whitelist(self):
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
                        base_url = f"https://docenti.unisa.it/{matricola}/"
                        self.diem_docenti_whitelist.add(base_url)
                        home_url = f"{base_url}home"
                        self.queue.append((home_url, 0))
                        self.visited_urls.add(home_url)
            logger.info(f"Trovati {len(self.diem_docenti_whitelist)} docenti DIEM.")
        except Exception as e:
            logger.error(f"Errore whitelist docenti: {e}")

    def is_valid_url(self, url: str) -> bool:
        try:
            parsed_url = urlparse(url)
            if parsed_url.netloc not in self.ALLOWED_DOMAINS:
                return False
        except ValueError:
            return False

        if parsed_url.netloc == "docenti.unisa.it":
            if not any(url.startswith(doc) for doc in self.diem_docenti_whitelist):
                return False
        else:
            if not url.startswith(self.ALLOWED_PREFIXES):
                return False
        
        url_lower = url.lower()
        if any(url_lower.endswith(ext) for ext in self.IGNORED_EXTENSIONS):
            return False
            
        html_traps = [
            "sitemap", ".xml", "rss", "feed", "unisa-rescue-page",
            "jsessionid", "saml2-redirect", "password-recovery", "choose-spid",
            "avviso=", "avvisi=", "&p=", "news-archive=", "eventi-archive=", 
            "category=", "news-category=", "idconcorso=", "commissioni-dettaglio=",
            "calendario-occupazione-spazi-sede", 
        ]
        if any(trap in url_lower for trap in html_traps):
            return False

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
    # PARSING
    # ==========================================

    def process_single_html(self, html_content: str, source_url: str):
        """Parsing e pulizia DOM. STATELESS per ThreadPoolExecutor."""
        soup = BeautifulSoup(html_content, "html.parser")
        local_new_links = set()
        local_pdfs = set()
        
        for a_tag in soup.find_all('a', href=True):
            full_url = urljoin(source_url, a_tag['href'])
            full_url, _ = urldefrag(full_url)
            
            # NORMALIZZAZIONE LINK PDF PER RIMUOVERE IL PATH INTERMEDIO PRIMA DI 'uploads'
            if full_url.lower().endswith('.pdf') and '/uploads/' in full_url.lower():
                parsed_url = urlparse(full_url)
                base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
                
                uploads_index = parsed_url.path.lower().find('/uploads/')
                if uploads_index != -1:
                    correct_path = parsed_url.path[uploads_index:]
                    full_url = base_domain + correct_path
                    
            if self.is_valid_url(full_url):
                if full_url.lower().endswith('.pdf'):
                    local_pdfs.add(full_url)
                else:
                    local_new_links.add(full_url)

        noise_selectors = [
            "#header", "#main-menu", "#menu-bar", "#unisa-left-menu", 
            "#box-agenda", "#share-dropdown", ".breadcrumb", 
            ".sr-only", "div[id$='-map']", "script", "style", "noscript"
        ]
        for selector in noise_selectors:
            for element in soup.select(selector):
                element.decompose()
                
        main_content = (soup.find(id="unisa-content") or 
                        soup.find(attrs={"role": "main"}) or 
                        soup.find(id="content") or 
                        soup.find("main") or soup.body)
                        
        clean_html = ""
        if main_content:
            allowed_attrs = ['href', 'src', 'colspan', 'rowspan']
            for tag in main_content.find_all(True):
                tag.attrs = {k: v for k, v in tag.attrs.items() if k in allowed_attrs}
            clean_html = main_content.decode_contents().strip()
            
        return clean_html, local_new_links, local_pdfs

    # ==========================================
    # SALVATAGGIO
    # ==========================================

    def _save_single_doc(self, doc, current_depth):
        raw_url = doc.metadata.get('source', 'URL_sconosciuto')
        safe_name = re.sub(r'[<>:"/\\|?*]', '-', raw_url.replace("https://", "").replace("http://", ""))
        safe_name = safe_name.replace("__", "_")
        filepath = os.path.join(self.output_dir, f"doc{self.processed_count}_depth{current_depth}_{safe_name}.html")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"<meta charset='utf-8'>\n<!-- SOURCE: {raw_url} -->\n<!-- DEPTH: {current_depth} -->\n")
                f.write(doc.page_content)
        except Exception as e:
            logger.error(f"Errore scrittura {filepath}: {e}")

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
        logger.info("--- AVVIO CRAWLING CON FILTRI INLINE ---")
        
        base_seeds = [url for url in self.ALLOWED_PREFIXES if url != "https://docenti.unisa.it"]
        for url in base_seeds:
            self.queue.append((url, 0))
            self.visited_urls.add(url)
        self.initialize_diem_docenti_whitelist()

        while self.queue:
            batch = []
            while self.queue and len(batch) < self.batch_size:
                url, depth = self.queue.popleft()
                if depth <= self.max_depth:
                    batch.append((url, depth))
            
            if not batch:
                continue
                
            urls_to_fetch = [item[0] for item in batch]
            depths_dict = dict(batch)
            
            logger.info(
                f"Batch: {len(urls_to_fetch)} URL "
                f"(Coda: {len(self.queue)} | Salvati: {self.processed_count} | Filtrati: {self.filtered_count})"
            )
            
            loader = AsyncHtmlLoader(urls_to_fetch, ignore_load_errors=True)
            raw_docs = loader.load()

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
                        
                        if not clean_html:
                            self.filtered_count += 1
                            continue
                        
                        # ========================================
                        # FILTRO HTML INLINE (Step 2 del prompt)
                        # ========================================
                        should_discard, reason = self._should_discard_html(source_url, clean_html)
                        
                        if should_discard:
                            logger.debug(f"HTML scartato: {source_url[:80]} — {reason}")
                            self.filtered_count += 1
                            # I PDF di questa pagina NON vengono raccolti (Step 3)
                            continue
                        
                        # ========================================
                        # FILTRO PDF INLINE (Step 3 del prompt)
                        # Solo i PDF da pagine HTML valide vengono raccolti
                        # ========================================
                        for pdf_url in local_pdfs:
                            if not self._should_discard_pdf(pdf_url):
                                self.found_pdf_links.add(pdf_url)
                        
                        # Salvataggio HTML valido
                        doc.page_content = clean_html
                        self._save_single_doc(doc, current_depth)
                        self.processed_count += 1
                            
                        # Accoda nuovi link per il BFS
                        if current_depth < self.max_depth:
                            for new_link in local_new_links:
                                if new_link not in self.visited_urls:
                                    self.queue.append((new_link, current_depth + 1))
                                    self.visited_urls.add(new_link)
                                    
                    except Exception as e:
                        logger.error(f"Errore elaborazione {source_url}: {e}")

            self._save_pdf_ledger()
            time.sleep(self.delay)

        self._print_summary()

    def _print_summary(self):
        logger.info("====== RESOCONTO FINALE ======")
        logger.info(f"URL visitati: {len(self.visited_urls)}")
        logger.info(f"HTML salvati: {self.processed_count}")
        logger.info(f"HTML filtrati (scartati): {self.filtered_count}")
        logger.info(f"PDF validi raccolti: {len(self.found_pdf_links)}")


# ==========================================
# ESECUZIONE
# ==========================================
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