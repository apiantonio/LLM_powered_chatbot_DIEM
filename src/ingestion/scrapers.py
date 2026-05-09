"""
UnisaCrawler refactorato con filtri inline + filtri docenti avanzati.

SPRINT FILTRI DOCENTI — Implementa i 5 requisiti:
  1. Filtraggio URL: Ricerca Progetti (ruolo=tutti salva, padre naviga-only, figli scartati)
  2. Filtraggio URL: Ricerca Pubblicazioni (solo anno=0 salvato)
  3. Filtraggio contenuto HTML: Pubblicazioni anno=0 (rimozione DOM anno < cutoff)
  4. Bug Fix: Eliminazione file didattica spostata in post-processing (dopo join thread)
  5. Ottimizzazione: Thread dinamici conservativi + I/O thread-safe con lock

CAMBIAMENTO ARCHITETTURALE:
  PRIMA: Crawl → salva tutto su disco → main.py applica HTMLCleaner a posteriori
  DOPO:  Crawl → filtra in-memory → salva SOLO i file che superano i filtri

Pattern: Strategy (i filtri sono iniettati come liste di regole)
         + Template Method (il ciclo BFS è fisso, i punti di estensione sono i filtri)
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
from config.settings import CrawlerConfig, IngestionConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class UnisaCrawler:
    """
    Crawler BFS con filtri inline integrati + filtri docenti avanzati.

    Novità rispetto alla versione precedente:
    - Filtri URL per ricerca/progetti e ricerca/pubblicazioni (Req 1-2)
    - Filtraggio DOM pubblicazioni anno=0 per cutoff_year (Req 3)
    - Post-processing didattica dopo join dei thread (Req 4)
    - Thread pool dinamico conservativo da CrawlerConfig (Req 5)
    - Lock I/O per scrittura thread-safe su disco (Req 5)
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

        # --- Config centralizzata (Req 5) ---
        self._crawler_cfg = crawler_config or CrawlerConfig()
        self._ingestion_cfg = ingestion_config or IngestionConfig()

        # Thread pool: calcolo dinamico conservativo (Req 5)
        # Se max_workers è passato esplicitamente, lo rispettiamo (backward compat);
        # altrimenti calcoliamo dal CrawlerConfig.
        if max_workers is not None:
            self.max_workers = max_workers
        else:
            self.max_workers = self._crawler_cfg.compute_max_workers()

        # Filtri iniettati (Strategy Pattern) — invariati
        self._html_rules = html_rules or []
        self._pdf_rules = pdf_rules or []

        # --- Stato ---
        self.visited_urls: Set[str] = set()
        self.found_pdf_links: Set[str] = set()
        self.diem_docenti_whitelist: Set[str] = set()
        self.queue = deque()
        self.processed_count = 0
        self.filtered_count = 0

        # --- (Req 4) Coda post-processing: file didattica da eliminare ---
        # Popolata durante il crawling, processata DOPO il join dei thread.
        self._didattica_pending_deletions: Set[str] = set()

        # --- (Req 5) Lock I/O per scrittura thread-safe ---
        self._write_lock = threading.Lock()
        # Lock per il contatore processed_count (accesso concorrente dai thread)
        self._counter_lock = threading.Lock()

        # --- (Req 1) Pattern compilati per filtri URL docenti ---
        # Parametri dei figli da scartare sotto ricerca/progetti
        self._progetti_discard_params = self._crawler_cfg.progetti_discard_params

        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(
            f"Crawler inizializzato — Workers: {self.max_workers} "
            f"(CPU: {os.cpu_count()}, fattore: {self._crawler_cfg.thread_cpu_factor}), "
            f"HTML rules: {len(self._html_rules)}, PDF rules: {len(self._pdf_rules)}, "
            f"cutoff_year: {self._ingestion_cfg.cutoff_year}"
        )

    # ==========================================
    # REQ 1: FILTRO URL — RICERCA PROGETTI
    # ==========================================

    def _classify_progetti_url(self, url: str) -> str:
        """
        Classifica un URL della sezione ricerca/progetti.

        Returns:
            "save"    → salvare su disco (es. ruolo=tutti)
            "navigate"→ navigare per estrarre link, ma NON salvare (pagina padre)
            "discard" → scartare completamente (figli specifici)
            "pass"    → URL non è di ricerca/progetti, nessun filtro applicato
        """
        url_lower = url.lower()

        # Verifica se l'URL appartiene alla sezione ricerca/progetti di un docente
        # Pattern: docenti.unisa.it/{matricola}/ricerca/progetti
        if "docenti.unisa.it/" not in url_lower or "/ricerca/progetti" not in url_lower:
            return "pass"

        # --- CASO 1: ruolo=tutti → SALVARE ---
        if "ruolo=tutti" in url_lower:
            return "save"

        # --- CASO 3: Figli con parametri specifici → SCARTARE ---
        # Controlla PRIMA dei genitori perché i figli hanno query params aggiuntivi
        for param in self._progetti_discard_params:
            if param in url_lower:
                return "discard"

        # --- CASO 2: Pagina padre (nessun parametro ruolo/progetto/tip/stato) ---
        # Se arriviamo qui, è la pagina base /ricerca/progetti senza parametri filtro.
        # Il pattern è: .../{matricola}/ricerca/progetti (senza query params significativi)
        # → navigare per estrarre i link figli, ma NON salvare.
        parsed = urlparse(url)
        path = parsed.path.rstrip("/")
        if path.endswith("/ricerca/progetti"):
            return "navigate"

        # Fallback: se non matcha nessun caso specifico, passa
        return "pass"

    # ==========================================
    # REQ 2: FILTRO URL — RICERCA PUBBLICAZIONI
    # ==========================================

    def _classify_pubblicazioni_url(self, url: str) -> str:
        """
        Classifica un URL della sezione ricerca/pubblicazioni.

        Returns:
            "save"    → salvare (anno=0, con filtraggio DOM successivo)
            "discard" → scartare (qualsiasi altro anno specifico)
            "pass"    → URL non è di ricerca/pubblicazioni
        """
        url_lower = url.lower()

        if "docenti.unisa.it/" not in url_lower or "/ricerca/pubblicazioni" not in url_lower:
            return "pass"

        # anno=0 → pagina "Tutti" → SALVARE (con filtraggio DOM al Req 3)
        if "anno=0" in url_lower:
            return "save"

        # Qualsiasi altro anno=YYYY → SCARTARE (non navigare, non salvare)
        if re.search(r'anno=\d+', url_lower):
            return "discard"

        # Pagina base /ricerca/pubblicazioni senza parametro anno
        # → gestita dalla regola ExactPublicationsBaseRule esistente
        return "pass"

    # ==========================================
    # REQ 3: FILTRAGGIO CONTENUTO HTML PUBBLICAZIONI
    # ==========================================

    def _filter_publications_by_year(self, html_content: str) -> str:
        """
        Filtra il DOM delle pubblicazioni anno=0, rimuovendo le entry
        con anno < cutoff_year (default 2020).

        Struttura DOM attesa (da esempio_pagina_ricerca_pubblicazioni.txt):
          Ogni pubblicazione è un <div> contenente:
            - <h4> con <small>ID</small> e titolo
            - <table> con righe <tr>, dove la PRIMA <tr> contiene l'anno
              nella seconda <td> (es. <td>2021</td>)

          Le pubblicazioni sono sibling <div> dentro un contenitore comune.

        Strategia:
          1. Trova tutti i blocchi pubblicazione (div con h4 + table).
          2. Per ciascuno, estrai l'anno dalla prima riga della tabella.
          3. Se anno < cutoff_year, rimuovi l'intero blocco dal DOM.
          4. Restituisci l'HTML ripulito.
        """
        cutoff = self._ingestion_cfg.cutoff_year
        soup = BeautifulSoup(html_content, "html.parser")

        # Ogni pubblicazione è un <div> che contiene un <h4> (titolo) e una <table> (dettagli)
        # Cerchiamo tutti i <div> che hanno sia <h4> che <table> come discendenti diretti
        removed_count = 0

        # I blocchi pubblicazione sono div con un h4 che contiene un <small> (l'ID numerico)
        for pub_div in soup.find_all("div"):
            h4 = pub_div.find("h4", recursive=False)
            if not h4:
                continue
            small_tag = h4.find("small")
            if not small_tag:
                continue

            # Verifica che il div contenga anche una tabella (struttura pubblicazione)
            table = pub_div.find("table")
            if not table:
                continue

            # Estrai l'anno dalla prima riga della tabella
            # Struttura: <tr><td><i></i></td><td>2021</td></tr>
            first_row = table.find("tr")
            if not first_row:
                continue

            tds = first_row.find_all("td")
            if len(tds) < 2:
                continue

            year_text = tds[1].get_text(strip=True)
            try:
                year = int(year_text)
            except ValueError:
                # Se non è un numero, non possiamo filtrare → conserviamo
                continue

            if year < cutoff:
                pub_div.decompose()
                removed_count += 1

        if removed_count > 0:
            logger.debug(
                f"  Pubblicazioni filtrate: {removed_count} entry "
                f"con anno < {cutoff} rimosse dal DOM"
            )

        return str(soup)

    # ==========================================
    # REQ 4: CLASSIFICAZIONE DIDATTICA (per post-processing)
    # ==========================================

    def _classify_didattica_url(self, url: str) -> str:
        """
        Classifica un URL della sezione didattica per il post-processing.

        Il bug originale: la DidatticaFilterRule tentava di eliminare file
        che non erano ancora stati scritti su disco (race condition con i thread).

        Nuova strategia:
          - Durante il crawling, TUTTI i file didattica vengono salvati normalmente.
          - I file "padre" (solo id=, senza cId= e pId=) vengono registrati
            per l'eliminazione nel post-processing.
          - Il post-processing avviene DOPO il join() di tutti i thread.

        Returns:
            "save_and_mark" → salvare E registrare per eliminazione post-processing
            "save"          → salvare normalmente (versione completa)
            "discard"       → scartare (id+cId senza pId)
            "pass"          → non è un URL didattica
        """
        url_lower = url.lower()

        if "docenti.unisa.it/" not in url_lower or "/didattica" not in url_lower:
            return "pass"

        has_id = "id=" in url_lower
        has_cid = "cid=" in url_lower
        has_pid = "pid=" in url_lower

        # id + cId senza pId → versione incompleta → scartare
        if has_id and has_cid and not has_pid:
            return "discard"

        # id + cId + pId → versione completa → salvare
        # E registrare il file padre "solo id" per eliminazione post-processing
        if has_id and has_cid and has_pid:
            return "save"

        # Solo id (senza cId e pId) → salvare ora, eliminare nel post-processing
        # Lo salviamo perché il crawler deve estrarre i link figli da qui.
        if has_id and not has_cid and not has_pid:
            return "save_and_mark"

        return "pass"

    # ==========================================
    # FILTRI INLINE (logica preesistente, invariata)
    # ==========================================

    def _should_discard_html(self, source_url: str, clean_html: str) -> Tuple[bool, str]:
        """
        Valuta il contenuto HTML in-memory contro tutte le regole attive.
        (Invariato rispetto alla versione precedente)
        """
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
    # DISCOVERY E VALIDAZIONE URL (invariato)
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
        Validazione URL base (invariata) + filtri docenti (Req 1-2).

        I filtri docenti intercettano gli URL PRIMA dell'accodamento:
        gli URL classificati come "discard" non entrano mai nella coda BFS.
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

        # --- REQ 1-2: Filtri URL docenti (pre-accodamento) ---
        # Gli URL "discard" non devono nemmeno entrare nella coda BFS.
        progetti_class = self._classify_progetti_url(url)
        if progetti_class == "discard":
            return False

        pubblicazioni_class = self._classify_pubblicazioni_url(url)
        if pubblicazioni_class == "discard":
            return False

        # "didattica" discard a livello URL
        didattica_class = self._classify_didattica_url(url)
        if didattica_class == "discard":
            return False

        return True

    # ==========================================
    # PARSING (invariato nella struttura, aggiunto hook per filtri)
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
    # SALVATAGGIO (Req 5: thread-safe con lock)
    # ==========================================

    def _save_single_doc(self, doc, current_depth, doc_index: int) -> Optional[str]:
        """
        Salva un documento su disco in modo thread-safe.

        Args:
            doc: Il documento LangChain con .page_content e .metadata.
            current_depth: Profondità BFS corrente.
            doc_index: Indice progressivo del documento (thread-safe).

        Returns:
            Il filepath del file salvato, o None in caso di errore.
        """
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

        # (Req 5) Lock per serializzare le scritture su disco
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
    # REQ 4: POST-PROCESSING DIDATTICA
    # ==========================================

    def _postprocess_didattica_cleanup(self) -> int:
        """
        Elimina i file "padre" della didattica (solo id=, senza cId/pId)
        DOPO che tutti i thread hanno terminato la scrittura.

        Questo risolve la race condition: i file vengono scritti dal
        ThreadPoolExecutor, e solo dopo il join() dei thread (fine del
        ciclo batch) questa funzione li elimina in modo sicuro.

        Returns:
            Numero di file eliminati.
        """
        if not self._didattica_pending_deletions:
            return 0

        deleted_count = 0
        output_path = Path(self.output_dir)

        for file_id in self._didattica_pending_deletions:
            # Cerca tutti i file con id={file_id} che NON hanno cId= o pId=
            for saved_file in output_path.glob(f"*-didattica-*id={file_id}*.html"):
                saved_name = saved_file.name.lower()

                # Elimina SOLO la versione padre (senza cId e pId)
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
    # DECISIONE SALVATAGGIO (integra Req 1-3)
    # ==========================================

    def _decide_save(
        self, source_url: str, clean_html: str, doc, current_depth: int
    ) -> bool:
        """
        Punto centrale di decisione: salvare o scartare un documento.

        Integra i filtri inline esistenti con i nuovi filtri docenti.
        Applica il filtraggio DOM per le pubblicazioni anno=0 (Req 3).

        Returns:
            True se il documento è stato salvato, False altrimenti.
        """
        # --- STEP 1: Contenuto vuoto ---
        if not clean_html:
            self.filtered_count += 1
            logger.debug(f"Scartato (contenuto vuoto): {source_url}")
            return False

        # --- STEP 2: Filtro progetti (Req 1) ---
        progetti_class = self._classify_progetti_url(source_url)
        if progetti_class == "navigate":
            # Navigare per link ma NON salvare
            self.filtered_count += 1
            logger.debug(f"Progetti padre (navigate-only): {source_url}")
            return False
        # "save" e "pass" proseguono normalmente

        # --- STEP 3: Filtro pubblicazioni URL (Req 2) ---
        # "discard" è già gestito in is_valid_url, qui restano "save" e "pass"
        pubblicazioni_class = self._classify_pubblicazioni_url(source_url)

        # --- STEP 4: Regole di pulizia inline (invariate) ---
        should_discard, reason = self._should_discard_html(source_url, clean_html)
        if should_discard:
            self.filtered_count += 1
            logger.info(f"Scartato ({reason}): {source_url}")
            return False

        # --- STEP 5: Filtraggio DOM pubblicazioni anno=0 (Req 3) ---
        if pubblicazioni_class == "save":
            clean_html = self._filter_publications_by_year(clean_html)
            doc.page_content = clean_html

        # --- STEP 6: Classificazione didattica (Req 4) ---
        didattica_class = self._classify_didattica_url(source_url)
        if didattica_class == "save_and_mark":
            # Estraiamo l'id= per il post-processing
            match = re.search(r'id=(\d+)', source_url, re.IGNORECASE)
            if match:
                self._didattica_pending_deletions.add(match.group(1))

        # --- STEP 7: Raccolta PDF (solo da documenti che vengono salvati) ---
        # (Nota: i PDF sono già stati estratti in process_single_html,
        #  ma li filtriamo qui con le regole PDF)

        # --- STEP 8: Salvataggio ---
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

            # (Req 5) Il ThreadPoolExecutor usa il numero di worker calcolato
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

                        # Decisione centralizzata (integra Req 1-3-4)
                        saved = self._decide_save(
                            source_url, clean_html, doc, current_depth
                        )

                        # Raccolta PDF solo se il documento è stato salvato
                        if saved:
                            for pdf_url in local_pdfs:
                                if not self._should_discard_pdf(pdf_url):
                                    self.found_pdf_links.add(pdf_url)

                        # ← SEMPRE: accodamento figli indipendente dalla decisione
                        if current_depth < self.max_depth:
                            for new_link in local_new_links:
                                if new_link not in self.visited_urls:
                                    self.queue.append((new_link, current_depth + 1))
                                    self.visited_urls.add(new_link)

                    except Exception as e:
                        logger.error(f"Errore elaborazione {source_url}: {e}")

            # Il ThreadPoolExecutor.__exit__ chiama shutdown(wait=True),
            # quindi a questo punto TUTTI i thread del batch hanno terminato.

            self._save_pdf_ledger()
            time.sleep(self.delay)

        # ============================================================
        # (Req 4) POST-PROCESSING: eliminazione file didattica padre
        # Eseguito DOPO la fine di TUTTI i batch e TUTTI i thread.
        # Nessuna race condition possibile: tutti i file sono su disco.
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