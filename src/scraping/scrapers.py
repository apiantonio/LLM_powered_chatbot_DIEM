"""Crawler principale per il sito UNISA.

Coordina il processo di crawling BFS, delegando la classificazione URL
alla UrlClassificationPipeline, la persistenza a CrawlerPersistence
e il post-processing a PostProcessor.
"""

import logging
import os
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Set, Tuple
from urllib.parse import urlparse, urljoin, urldefrag

import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import AsyncHtmlLoader

from config.settings import CrawlerConfig, IngestionConfig
from scraping.interfaces import (
    CleaningRule,
    PdfFilterRule,
    UrlClassificationPipeline,
)
from scraping.persistence import CrawlerPersistence, PostProcessor
from scraping.rules.urls import (
    CorsiUrlClassifier,
    DidatticaIdUrlClassifier,
    DidatticaOrariUrlClassifier,
    InternationalSubpagesUrlClassifier,
    InternationalUrlClassifier,
    ProgettiUrlClassifier,
    PubblicazioniUrlClassifier,
    RicercaBaseUrlClassifier,
)

logger = logging.getLogger(__name__)


class UnisaCrawler:
    """Crawler BFS multi-thread per i domini UNISA autorizzati."""

    ALLOWED_DOMAINS = frozenset({
        "www.diem.unisa.it",
        "docenti.unisa.it",
        "corsi.unisa.it",
    })

    ALLOWED_PREFIXES = (
        "https://www.diem.unisa.it",
        "https://corsi.unisa.it/ingegneria-informatica",
        "https://corsi.unisa.it/ingegneria-dell-informazione-per-la-medicina-digitale",
        "https://corsi.unisa.it/ingegneria-informatica-magistrale",
        "https://corsi.unisa.it/electrical-engineering-for-digital-energy",
        "https://corsi.unisa.it/information-Engineering-for-digital-medicine",
        "https://corsi.unisa.it/ingegneria-dell-informazione",
        "https://corsi.unisa.it/photovoltaics",
    )

    IGNORED_EXTENSIONS = (
        ".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp",
        ".zip", ".rar", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    )

    _HTML_TRAPS = (
        "sitemap", ".xml", "rss", "feed", "unisa-rescue-page",
        "jsessionid", "saml2-redirect", "password-recovery", "choose-spid",
        "avviso=", "avvisi=", "&p=", "news-archive=", "eventi-archive=",
        "category=", "news-category=", "idconcorso=", "commissioni-dettaglio=",
        "calendario-occupazione-spazi-sede",
    )

    _PDF_URL_TRAPS = (
        "grad_", "graduatori", "esit", "risultat", "ammess",
        "verbale", "verbali", "decreto", "approvazione_atti",
        "commissione", "contratt", "incarico", "valutazione",
        "scorrimento", "elenco", "candidat", "modulo",
        "richiesta", "domanda",
    )

    _NOISE_SELECTORS = (
        "#header", "#main-menu", "#menu-bar", "#unisa-left-menu",
        "#box-agenda", "#share-dropdown", ".breadcrumb",
        ".sr-only", "div[id$='-map']", "script", "style", "noscript",
    )

    _ALLOWED_ATTRS = ("href", "src", "colspan", "rowspan")

    _HOMEPAGE_URLS = frozenset({
        "https://www.diem.unisa.it",
        "http://www.diem.unisa.it",
        "https://www.diem.unisa.it/home",
        "http://www.diem.unisa.it/home",
    })

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
        """Inizializza il crawler con configurazione e regole di filtraggio.

        Args:
            max_depth: Profondita massima di crawling BFS.
            batch_size: Numero di URL per batch.
            delay: Ritardo in secondi tra un batch e il successivo.
            max_workers: Numero di thread worker (auto-calcolato se None).
            output_dir: Directory di output per i file HTML.
            html_rules: Lista di regole di pulizia HTML.
            pdf_rules: Lista di regole di filtraggio PDF.
            crawler_config: Configurazione operativa del crawler.
            ingestion_config: Configurazione di ingestion.
        """
        self.max_depth = max_depth
        self.batch_size = batch_size
        self.delay = delay
        self.output_dir = output_dir

        self._crawler_cfg = crawler_config or CrawlerConfig()
        self._ingestion_cfg = ingestion_config or IngestionConfig()

        if max_workers is not None:
            self.max_workers = max_workers
        else:
            self.max_workers = self._crawler_cfg.compute_max_workers()

        self._html_rules = html_rules or []
        self._pdf_rules = pdf_rules or []

        self._validation_pipeline = UrlClassificationPipeline([
            ProgettiUrlClassifier(),
            PubblicazioniUrlClassifier(),
            DidatticaOrariUrlClassifier(),
            DidatticaIdUrlClassifier(),
            InternationalSubpagesUrlClassifier(),
        ])

        self._save_decision_pipeline = UrlClassificationPipeline([
            PubblicazioniUrlClassifier(),
            RicercaBaseUrlClassifier(),
            InternationalUrlClassifier(),
            InternationalSubpagesUrlClassifier(),
            CorsiUrlClassifier(),
        ])

        self._didattica_id_classifier = DidatticaIdUrlClassifier()

        self._persistence = CrawlerPersistence(output_dir)
        self._post_processor = PostProcessor(output_dir, self._ingestion_cfg.cutoff_year)

        self.visited_urls: Set[str] = set()
        self.found_pdf_links: Set[str] = set()
        self.diem_docenti_whitelist: Set[str] = set()
        self.queue: deque = deque()
        self.processed_count = 0
        self.filtered_count = 0

        self._counter_lock = threading.Lock()

        logger.info(
            "Crawler inizializzato -- Workers: %d (CPU: %s, fattore: %s), "
            "HTML rules: %d, PDF rules: %d, cutoff_year: %d",
            self.max_workers,
            os.cpu_count(),
            self._crawler_cfg.thread_cpu_factor,
            len(self._html_rules),
            len(self._pdf_rules),
            self._ingestion_cfg.cutoff_year,
        )

    def _should_discard_html(self, source_url: str, clean_html: str) -> Tuple[bool, str]:
        """Valuta se una pagina HTML deve essere scartata in base alle regole configurate.

        Args:
            source_url: URL di provenienza della pagina.
            clean_html: Contenuto HTML ripulito.

        Returns:
            Tupla (deve_scartare, nome_regola).
        """
        safe_name = re.sub(
            r'[<>:"/\\|?*]',
            "-",
            source_url.replace("https://", "").replace("http://", ""),
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
            except Exception as exc:
                logger.debug("Regola %s ha sollevato eccezione: %s", rule.name, exc)
                continue

        return False, ""

    def _should_discard_pdf(self, pdf_url: str) -> bool:
        """Valuta se un PDF deve essere scartato in base alle regole configurate.

        Args:
            pdf_url: URL del PDF da valutare.

        Returns:
            True se il PDF deve essere scartato.
        """
        for rule in self._pdf_rules:
            if rule.should_discard(pdf_url):
                return True
        return False

    def initialize_diem_docenti_whitelist(self) -> None:
        """Scarica la whitelist dei docenti DIEM e popola la coda iniziale."""
        url_personale = "https://www.diem.unisa.it/dipartimento/personale"
        logger.info("Inizializzazione Whitelist Docenti DIEM...")
        try:
            response = requests.get(url_personale, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                if "rubrica.unisa.it/persone?matricola=" in href:
                    match = re.search(r"matricola=(\d+)", href)
                    if match:
                        matricola = match.group(1)
                        base_url = f"https://docenti.unisa.it/{matricola}/"
                        self.diem_docenti_whitelist.add(base_url)
                        home_url = f"{base_url}home"
                        self.queue.append((home_url, 0))
                        self.visited_urls.add(home_url)
            logger.info("Trovati %d docenti DIEM.", len(self.diem_docenti_whitelist))
        except Exception as exc:
            logger.error("Errore whitelist docenti: %s", exc)

    def is_valid_url(self, url: str) -> bool:
        """Determina se un URL rientra nel perimetro di crawling.

        Args:
            url: URL da validare.

        Returns:
            True se l'URL e' valido per il crawling.
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

        if any(trap in url_lower for trap in self._HTML_TRAPS):
            return False

        if url_lower.endswith(".pdf"):
            if any(trap in url_lower for trap in self._PDF_URL_TRAPS):
                return False

        pipeline_decision = self._validation_pipeline.classify(url)
        if pipeline_decision == "discard":
            return False

        return True

    def process_single_html(self, html_content: str, source_url: str) -> Tuple[str, Set[str], Set[str]]:
        """Esegue il parsing e la pulizia DOM di una singola pagina HTML.

        Metodo stateless, sicuro per l'esecuzione in ThreadPoolExecutor.

        Args:
            html_content: Contenuto HTML grezzo.
            source_url: URL di provenienza.

        Returns:
            Tupla (html_pulito, nuovi_link, link_pdf).
        """
        soup = BeautifulSoup(html_content, "html.parser")
        local_new_links: Set[str] = set()
        local_pdfs: Set[str] = set()

        for a_tag in soup.find_all("a", href=True):
            raw_href = a_tag["href"].strip()

            clean_path = raw_href.lower().split("?")[0]
            is_pdf = clean_path.endswith(".pdf")

            if is_pdf and raw_href.startswith("uploads/"):
                raw_href = "/" + raw_href

            full_url = urljoin(source_url, raw_href)
            full_url, _ = urldefrag(full_url)

            if is_pdf and "/uploads/" in full_url.lower():
                parsed_url = urlparse(full_url)
                base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
                uploads_index = parsed_url.path.lower().find("/uploads/")
                if uploads_index != -1:
                    correct_path = parsed_url.path[uploads_index:]
                    query_part = f"?{parsed_url.query}" if parsed_url.query else ""
                    full_url = base_domain + correct_path + query_part

            if is_pdf:
                local_pdfs.add(full_url)
            else:
                if self.is_valid_url(full_url):
                    local_new_links.add(full_url)

        for selector in self._NOISE_SELECTORS:
            for element in soup.select(selector):
                element.decompose()

        main_content = (
            soup.find(id="unisa-content")
            or soup.find(attrs={"role": "main"})
            or soup.find(id="content")
            or soup.find("main")
            or soup.body
        )

        clean_html = ""
        if main_content:
            for tag in main_content.find_all(True):
                tag.attrs = {
                    k: v for k, v in tag.attrs.items() if k in self._ALLOWED_ATTRS
                }
            clean_html = main_content.decode_contents().strip()

        return clean_html, local_new_links, local_pdfs


    def _increment_counter(self) -> int:
        """Incrementa atomicamente il contatore dei documenti processati.

        Returns:
            Valore aggiornato del contatore.
        """
        with self._counter_lock:
            self.processed_count += 1
            return self.processed_count

    def _decide_save(self, source_url: str, clean_html: str, doc, current_depth: int) -> str:
        """Decide se salvare, navigare o scartare una pagina HTML.

        Args:
            source_url: URL di provenienza.
            clean_html: Contenuto HTML ripulito.
            doc: Documento langchain.
            current_depth: Profondita di crawling corrente.

        Returns:
            Decisione: 'save', 'navigate' o 'discard'.
        """
        if not clean_html:
            return "discard"

        pipeline_decision = self._save_decision_pipeline.classify(source_url)
        if pipeline_decision == "navigate":
            return "navigate"
        if pipeline_decision == "discard":
            return "discard"

        should_discard, _ = self._should_discard_html(source_url, clean_html)
        if should_discard:
            return "discard"

        didattica_class = self._didattica_id_classifier.classify(source_url)
        if didattica_class == "discard":
            return "discard"

        doc.page_content = clean_html
        doc_index = self._increment_counter()
        self._persistence.save_html_document(doc, current_depth, doc_index)
        return "save"

    def run(self) -> None:
        """Avvia il processo di crawling BFS con post-processing finale."""
        logger.info("--- AVVIO CRAWLING CON FILTRI INLINE + FILTRI DOCENTI ---")
        logger.info(
            "    Thread pool: %d workers | Batch: %d | Max depth: %d",
            self.max_workers,
            self.batch_size,
            self.max_depth,
        )

        base_seeds = [
            url for url in self.ALLOWED_PREFIXES
            if url != "https://docenti.unisa.it"
        ]
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
                "Batch: %d URL (Coda: %d | Salvati: %d | Filtrati: %d)",
                len(urls_to_fetch),
                len(self.queue),
                self.processed_count,
                self.filtered_count,
            )

            loader = AsyncHtmlLoader(urls_to_fetch, ignore_load_errors=True)
            raw_docs = loader.load()

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self.process_single_html,
                        doc.page_content,
                        doc.metadata.get("source"),
                    ): doc
                    for doc in raw_docs
                    if doc.metadata.get("source")
                }

                for future in as_completed(futures):
                    doc = futures[future]
                    source_url = doc.metadata.get("source")
                    current_depth = depths_dict.get(source_url, 0)

                    try:
                        clean_html, local_new_links, local_pdfs = future.result()

                        decision = self._decide_save(
                            source_url, clean_html, doc, current_depth
                        )

                        if decision in ("save", "navigate"):
                            for pdf_url in local_pdfs:
                                if not self._should_discard_pdf(pdf_url):
                                    self.found_pdf_links.add(pdf_url)

                            if current_depth < self.max_depth:
                                for new_link in local_new_links:
                                    if new_link not in self.visited_urls:
                                        self.queue.append(
                                            (new_link, current_depth + 1)
                                        )
                                        self.visited_urls.add(new_link)

                        elif decision == "discard":
                            self.filtered_count += 1

                    except Exception as exc:
                        logger.error("Errore elaborazione %s: %s", source_url, exc)

            self._persistence.save_pdf_ledger(self.found_pdf_links)
            time.sleep(self.delay)

        didattica_deleted = self._post_processor.run_didattica_cleanup()
        anno_deleted = self._post_processor.run_anno_cleanup()

        self._print_summary(didattica_deleted, anno_deleted)

    def _print_summary(self, didattica_deleted: int = 0, anno_deleted: int = 0) -> None:
        """Stampa il resoconto finale del crawling.

        Args:
            didattica_deleted: File eliminati dal cleanup didattica.
            anno_deleted: File eliminati dal cleanup anno.
        """
        logger.info("====== RESOCONTO FINALE ======")
        logger.info("URL visitati: %d", len(self.visited_urls))
        logger.info("HTML salvati: %d", self.processed_count)
        logger.info("HTML filtrati (scartati): %d", self.filtered_count)
        logger.info("PDF validi raccolti: %d", len(self.found_pdf_links))
        logger.info("Thread pool utilizzato: %d workers", self.max_workers)
        if didattica_deleted:
            logger.info(
                "Didattica post-processing: %d file padre rimossi", didattica_deleted
            )
        if anno_deleted:
            logger.info(
                "Anno post-processing: %d file obsoleti rimossi", anno_deleted
            )
