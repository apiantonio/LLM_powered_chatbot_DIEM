"""
UnisaCrawler — Crawler BFS con filtri iniettati (Strategy Pattern).

ARCHITETTURA (Sprint Filtri Docenti):
  - Filtri URL docenti: delegati a classificatori in transform/rules/docenti_url_rules.py
  - Filtraggio DOM pubblicazioni: delegato a PublicationsHtmlFilter in
    transform/rules/html_content_rules.py
  - Post-processing didattica: dopo join() di tutti i thread (Req 5 bug fix)
  - Thread pool: calcolo dinamico conservativo da CrawlerConfig
  - I/O thread-safe con lock

Pattern: Strategy (filtri iniettati) + Template Method (ciclo BFS fisso)

Lo scrapers.py fa SOLO scraping: discovery URL, fetch HTML, parsing DOM, salvataggio.
Le regole di filtraggio sono TUTTE incapsulate in transform/rules/.
"""

import re
import os
import time
import requests
import logging
import threading
from collections import deque
from pathlib import Path
from urllib.parse import urlparse, urljoin, urldefrag
from bs4 import BeautifulSoup
from langchain_community.document_loaders import AsyncHtmlLoader
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Set, Tuple

from transform.core.base_rule import CleaningRule, PdfFilterRule
from transform.rules.docenti_url_rules import (
    ProgettiUrlClassifier,
    PubblicazioniUrlClassifier,
    DidatticaOrariUrlClassifier,
    DidatticaIdUrlClassifier,
)
from transform.rules.html_content_rules import PublicationsHtmlFilter
from config.settings import CrawlerConfig, IngestionConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class UnisaCrawler:
    """
    Crawler BFS con filtri iniettati (Strategy Pattern).

    Responsabilità:
      - Discovery e validazione URL (BFS)
      - Fetch HTML in batch (AsyncHtmlLoader)
      - Parsing e pulizia DOM base (noise removal)
      - Salvataggio thread-safe su disco
      - Raccolta link PDF

    Le logiche di filtraggio sono delegate ai classificatori iniettati:
      - ProgettiUrlClassifier (Req 1)
      - PubblicazioniUrlClassifier (Req 2)
      - DidatticaOrariUrlClassifier (Req 3)
      - PublicationsHtmlFilter (Req 4)
      - DidatticaIdUrlClassifier (Req 5)
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
        crawler_config: Optional[CrawlerConfig] = None,
        ingestion_config: Optional[IngestionConfig] = None,
    ):
        self.max_depth = max_depth
        self.batch_size = batch_size
        self.delay = delay
        self.output_dir = output_dir

        # --- Config centralizzata ---
        self._crawler_cfg = crawler_config or CrawlerConfig()
        self._ingestion_cfg = ingestion_config or IngestionConfig()

        # Thread pool: calcolo dinamico conservativo
        if max_workers is not None:
            self.max_workers = max_workers
        else:
            self.max_workers = self._crawler_cfg.compute_max_workers()

        # Filtri inline iniettati (Strategy Pattern)
        self._html_rules = html_rules or []
        self._pdf_rules = pdf_rules or []

        # --- Classificatori URL docenti (Strategy Pattern, da transform/rules/) ---
        self._progetti_classifier = ProgettiUrlClassifier()
        self._pubblicazioni_classifier = PubblicazioniUrlClassifier()
        self._orari_classifier = DidatticaOrariUrlClassifier()
        self._didattica_id_classifier = DidatticaIdUrlClassifier()

        # --- Filtro contenuto HTML pubblicazioni (da transform/rules/) ---
        self._publications_filter = PublicationsHtmlFilter(
            cutoff_year=self._ingestion_cfg.cutoff_year
        )

        # --- Stato ---
        self.visited_urls: Set[str] = set()
        self.found_pdf_links: Set[str] = set()
        self.diem_docenti_whitelist: Set[str] = set()
        self.queue = deque()
        self.processed_count = 0
        self.filtered_count = 0

        # --- (Req 5) Coda post-processing: file didattica da eliminare ---
        self._didattica_pending_deletions: Set[str] = set()

        # --- Lock I/O per scrittura thread-safe ---
        self._write_lock = threading.Lock()
        self._counter_lock = threading.Lock()

        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(
            f"Crawler inizializzato — Workers: {self.max_workers} "
            f"(CPU: {os.cpu_count()}, fattore: {self._crawler_cfg.thread_cpu_factor}), "
            f"HTML rules: {len(self._html_rules)}, PDF rules: {len(self._pdf_rules)}, "
            f"cutoff_year: {self._ingestion_cfg.cutoff_year}"
        )

    # ==========================================
    # FILTRI INLINE (logica preesistente, invariata)
    # ==========================================

    def _should_discard_html(self, source_url: str, clean_html: str) -> Tuple[bool, str]:
        """Valuta il contenuto HTML in-memory contro tutte le regole attive."""
        safe_name = re.sub(
            r'[<>:"/\\|?*]', '-',
            source_url.replace("https://", "").replace("http://", "")
        )
        fake_path = Path(f"{safe_name}.html")

        for rule in self._html_rules:
            try:
                if not rule.requires_content:
                    if rule.should_delete(fake_path):
                        return True, rule.name
                else:
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
        """
        Validazione URL base + filtri docenti (pre-accodamento).

        Gli URL classificati come "discard" non entrano mai nella coda BFS.
        """
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
                "grad_", "graduatori", "esit", "risultat", "ammess",
                "verbale", "verbali", "decreto", "approvazione_atti",
                "commissione", "contratt", "incarico", "valutazione",
                "scorrimento", "elenco", "candidat", "modulo",
                "richiesta", "domanda"
            ]
            if any(trap in url_lower for trap in pdf_traps):
                return False

        # --- Filtri URL docenti (Req 1, 2, 3): pre-accodamento ---
        if self._progetti_classifier.classify(url) == "discard":
            return False

        if self._pubblicazioni_classifier.classify(url) == "discard":
            return False

        # Req 3: Didattica/Orari — scartare completamente
        if self._orari_classifier.classify(url) == "discard":
            return False

        # Req 5: Didattica id/cId/pId — discard versioni incomplete
        if self._didattica_id_classifier.classify(url) == "discard":
            return False

        return True

    # ==========================================
    # PARSING (invariato nella struttura)
    # ==========================================

    def process_single_html(self, html_content: str, source_url: str):
        """Parsing e pulizia DOM. STATELESS per ThreadPoolExecutor."""
        soup = BeautifulSoup(html_content, "html.parser")
        local_new_links = set()
        local_pdfs = set()

        for a_tag in soup.find_all('a', href=True):
            raw_href = a_tag['href'].strip()

            clean_path = raw_href.lower().split('?')[0]
            is_pdf = clean_path.endswith('.pdf')

            if is_pdf and raw_href.startswith('uploads/'):
                raw_href = '/' + raw_href

            full_url = urljoin(source_url, raw_href)
            full_url, _ = urldefrag(full_url)

            if is_pdf and '/uploads/' in full_url.lower():
                parsed_url = urlparse(full_url)
                base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
                uploads_index = parsed_url.path.lower().find('/uploads/')
                if uploads_index != -1:
                    correct_path = parsed_url.path[uploads_index:]
                    query_part = f"?{parsed_url.query}" if parsed_url.query else ""
                    full_url = base_domain + correct_path + query_part

            if is_pdf:
                local_pdfs.add(full_url)
            else:
                if self.is_valid_url(full_url):
                    local_new_links.add(full_url)

        # ---- PULIZIA DOM ----
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
    # SALVATAGGIO (thread-safe con lock)
    # ==========================================

    def _save_single_doc(self, doc, current_depth, doc_index: int) -> Optional[str]:
        """Salva un documento su disco in modo thread-safe."""
        raw_url = doc.metadata.get('source', 'URL_sconosciuto')
        safe_name = re.sub(
            r'[<>:"/\\|?*]', '-',
            raw_url.replace("https://", "").replace("http://", "")
        )
        safe_name = safe_name.replace("__", "_")
        filepath = os.path.join(
            self.output_dir,
            f"doc{doc_index}_depth{current_depth}_{safe_name}.html"
        )

        with self._write_lock:
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(
                        f"<meta charset='utf-8'>\n"
                        f"<!-- SOURCE: {raw_url} -->\n"
                        f"<!-- DEPTH: {current_depth} -->\n"
                    )
                    f.write(doc.page_content)
                return filepath
            except Exception as e:
                logger.error(f"Errore scrittura {filepath}: {e}")
                return None

    def _save_pdf_ledger(self):
        if not self.found_pdf_links:
            return
        pdf_path = os.path.join(self.output_dir, "pdf_links.txt")
        with self._write_lock:
            try:
                with open(pdf_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(sorted(self.found_pdf_links)))
            except Exception as e:
                logger.error(f"Errore scrittura registro PDF: {e}")

    def _increment_counter(self) -> int:
        """Incrementa il contatore processed_count in modo thread-safe."""
        with self._counter_lock:
            self.processed_count += 1
            return self.processed_count

    # ==========================================
    # REQ 5: POST-PROCESSING DIDATTICA
    # ==========================================

    def _postprocess_didattica_cleanup(self) -> int:
        """
        Elimina i file "padre" della didattica (solo id=, senza cId/pId)
        DOPO che tutti i thread hanno terminato la scrittura.

        Risolve la race condition: tutti i file sono su disco quando
        questa funzione viene invocata (dopo join() dei thread).
        """
        if not self._didattica_pending_deletions:
            return 0

        deleted_count = 0
        output_path = Path(self.output_dir)

        for file_id in self._didattica_pending_deletions:
            for saved_file in output_path.glob(f"*-didattica-*id={file_id}*.html"):
                saved_name = saved_file.name.lower()

                if "cid=" not in saved_name and "pid=" not in saved_name:
                    try:
                        saved_file.unlink()
                        deleted_count += 1
                        logger.debug(
                            f"  Post-processing didattica: eliminato {saved_file.name}"
                        )
                    except OSError as e:
                        logger.warning(
                            f"  Impossibile eliminare {saved_file.name}: {e}"
                        )

        logger.info(
            f"Post-processing didattica: {deleted_count} file padre eliminati "
            f"({len(self._didattica_pending_deletions)} ID processati)"
        )
        self._didattica_pending_deletions.clear()
        return deleted_count

    # ==========================================
    # DECISIONE SALVATAGGIO
    # ==========================================

    def _decide_save(
        self, source_url: str, clean_html: str, doc, current_depth: int
    ) -> bool:
        """
        Punto centrale di decisione: salvare o scartare un documento.

        Integra i filtri inline con i classificatori docenti.
        """
        # --- STEP 1: Contenuto vuoto ---
        if not clean_html:
            self.filtered_count += 1
            logger.debug(f"Scartato (contenuto vuoto): {source_url}")
            return False

        # --- STEP 2: Filtro progetti (Req 1) ---
        progetti_class = self._progetti_classifier.classify(source_url)
        if progetti_class == "navigate":
            self.filtered_count += 1
            logger.debug(f"Progetti padre (navigate-only): {source_url}")
            return False

        # --- STEP 3: Classificazione pubblicazioni URL (Req 2) ---
        pubblicazioni_class = self._pubblicazioni_classifier.classify(source_url)

        # --- STEP 4: Regole di pulizia inline (invariate) ---
        should_discard, reason = self._should_discard_html(source_url, clean_html)
        if should_discard:
            self.filtered_count += 1
            logger.info(f"Scartato ({reason}): {source_url}")
            return False

        # --- STEP 5: Filtraggio DOM pubblicazioni anno=0 (Req 4) ---
        if pubblicazioni_class == "save":
            clean_html = self._publications_filter.filter_html(clean_html)
            doc.page_content = clean_html

        # --- STEP 6: Classificazione didattica id (Req 5) ---
        didattica_class = self._didattica_id_classifier.classify(source_url)
        if didattica_class == "save_and_mark":
            file_id = self._didattica_id_classifier.extract_id(source_url)
            if file_id:
                self._didattica_pending_deletions.add(file_id)

        # --- STEP 7: Salvataggio ---
        doc.page_content = clean_html
        doc_index = self._increment_counter()
        self._save_single_doc(doc, current_depth, doc_index)
        return True

    # ==========================================
    # MOTORE ORCHESTRATORE
    # ==========================================

    def run(self):
        logger.info("--- AVVIO CRAWLING CON FILTRI INLINE + FILTRI DOCENTI ---")
        logger.info(
            f"    Thread pool: {self.max_workers} workers | "
            f"Batch: {self.batch_size} | Max depth: {self.max_depth}"
        )

        base_seeds = [
            url for url in self.ALLOWED_PREFIXES
            if url != "https://docenti.unisa.it"
        ]
        for url in base_seeds:
            self.queue.append((url, 0))
            self.visited_urls.add(url)
        # self.initialize_diem_docenti_whitelist()

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
                f"(Coda: {len(self.queue)} | Salvati: {self.processed_count} "
                f"| Filtrati: {self.filtered_count})"
            )

            loader = AsyncHtmlLoader(urls_to_fetch, ignore_load_errors=True)
            raw_docs = loader.load()

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self.process_single_html,
                        doc.page_content,
                        doc.metadata.get('source')
                    ): doc
                    for doc in raw_docs if doc.metadata.get('source')
                }

                for future in as_completed(futures):
                    doc = futures[future]
                    source_url = doc.metadata.get('source')
                    current_depth = depths_dict.get(source_url, 0)

                    try:
                        clean_html, local_new_links, local_pdfs = future.result()

                        saved = self._decide_save(
                            source_url, clean_html, doc, current_depth
                        )

                        if saved:
                            for pdf_url in local_pdfs:
                                if not self._should_discard_pdf(pdf_url):
                                    self.found_pdf_links.add(pdf_url)

                        if current_depth < self.max_depth:
                            for new_link in local_new_links:
                                if new_link not in self.visited_urls:
                                    self.queue.append((new_link, current_depth + 1))
                                    self.visited_urls.add(new_link)

                    except Exception as e:
                        logger.error(f"Errore elaborazione {source_url}: {e}")

            # ThreadPoolExecutor.__exit__ → shutdown(wait=True)
            # Tutti i thread del batch hanno terminato.

            self._save_pdf_ledger()
            time.sleep(self.delay)

        # ============================================================
        # (Req 5) POST-PROCESSING: eliminazione file didattica padre
        # Eseguito DOPO la fine di TUTTI i batch e TUTTI i thread.
        # ============================================================
        didattica_deleted = self._postprocess_didattica_cleanup()

        self._print_summary(didattica_deleted)

    def _print_summary(self, didattica_deleted: int = 0):
        logger.info("====== RESOCONTO FINALE ======")
        logger.info(f"URL visitati: {len(self.visited_urls)}")
        logger.info(f"HTML salvati: {self.processed_count}")
        logger.info(f"HTML filtrati (scartati): {self.filtered_count}")
        logger.info(f"PDF validi raccolti: {len(self.found_pdf_links)}")
        logger.info(f"Thread pool utilizzato: {self.max_workers} workers")
        if didattica_deleted:
            logger.info(f"Didattica post-processing: {didattica_deleted} file padre rimossi")