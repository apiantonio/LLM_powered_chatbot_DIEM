"""Regole di filtraggio per URL che puntano a documenti PDF.

Tutte le regole ereditano da PdfFilterRule.
"""

import logging
import re
from typing import Set
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from scraping.interfaces import PdfFilterRule

logger = logging.getLogger(__name__)


class ObsoleteYearRule(PdfFilterRule):
    """Scarta PDF il cui anno piu recente nell'URL e' anteriore al cutoff."""

    def __init__(self, cutoff_year: int = 2020):
        """Inizializza la regola con l'anno di cutoff.

        Args:
            cutoff_year: Anno minimo accettabile (escluso).
        """
        self.cutoff_year = cutoff_year
        self._academic_pattern = re.compile(
            r"(?<!\d)(19\d{2}|20\d{2})[-_](19\d{2}|20\d{2})(?!\d)"
        )
        self._year_month_pattern = re.compile(
            r"(?<!\d)(19\d{2}|20\d{2})[-_](0[1-9]|1[0-2])(?!\d)"
        )
        self._single_year_pattern = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")
        self._doc_id_year_pattern = re.compile(r"doc\d{6}(\d{4})", re.IGNORECASE)
        self._ddmmyyyy_pattern = re.compile(
            r"(?<!\d)"
            r"(?:0[1-9]|[12]\d|3[01])"
            r"(?:0[1-9]|1[0-2])"
            r"(19\d{2}|20\d{2})"
            r"(?!\d)",
        )

    @property
    def name(self) -> str:
        return f"Filtro Anni Obsoleti (< {self.cutoff_year})"

    def should_discard(self, pdf_url: str) -> bool:
        """Restituisce True se tutti gli anni individuati nell'URL sono anteriori al cutoff."""
        logical_years = []
        working_url = pdf_url

        for match in self._academic_pattern.finditer(working_url):
            logical_years.append(int(match.group(1)))
        working_url = self._academic_pattern.sub("XXX", working_url)

        for match in self._year_month_pattern.finditer(working_url):
            logical_years.append(int(match.group(1)))
        working_url = self._year_month_pattern.sub("XXX", working_url)

        for match in self._doc_id_year_pattern.finditer(working_url):
            logical_years.append(int(match.group(1)))
        working_url = self._doc_id_year_pattern.sub("XXX", working_url)

        for match in self._ddmmyyyy_pattern.finditer(working_url):
            logical_years.append(int(match.group(1)))
        working_url = self._ddmmyyyy_pattern.sub("XXX", working_url)

        for match in self._single_year_pattern.finditer(working_url):
            logical_years.append(int(match.group(1)))

        if not logical_years:
            return False

        max_year = max(logical_years)
        return max_year < self.cutoff_year


class SemanticTrapRule(PdfFilterRule):
    """Scarta PDF burocratici o contenenti dati sensibili in base a keyword nell'URL."""

    _TRAPS = (
        "grad_",
        "graduatori",
        "esit",
        "risultat",
        "ammess",
        "verbale",
        "verbali",
        "decreto",
        "approvazione_atti",
        "commissione",
        "contratt",
        "incarico",
        "valutazione",
        "scorrimento",
        "elenco",
        "candidat",
        "modulo",
        "richiesta",
        "domanda",
    )

    @property
    def name(self) -> str:
        return "PDF Burocratico/Dati Sensibili"

    def should_discard(self, url: str) -> bool:
        """Restituisce True se l'URL contiene keyword burocratiche/sensibili."""
        url_lower = url.lower()
        return any(trap in url_lower for trap in self._TRAPS)


class DomainWhitelistRule(PdfFilterRule):
    """Scarta PDF provenienti da domini non autorizzati o da docenti fuori perimetro DIEM."""

    _ALLOWED_DOMAINS = frozenset(
        {"www.diem.unisa.it", "docenti.unisa.it", "corsi.unisa.it"}
    )

    _ALLOWED_PREFIXES = (
        "https://www.diem.unisa.it",
        "https://docenti.unisa.it",
        "https://corsi.unisa.it",
        "https://corsi.unisa.it/ingegneria-informatica",
        "https://corsi.unisa.it/ingegneria-dell-informazione-per-la-medicina-digitale",
        "https://corsi.unisa.it/ingegneria-informatica-magistrale",
        "https://corsi.unisa.it/electrical-engineering-for-digital-energy",
        "https://corsi.unisa.it/information-Engineering-for-digital-medicine",
        "https://corsi.unisa.it/ingegneria-dell-informazione",
        "https://corsi.unisa.it/photovoltaics",
    )

    def __init__(self):
        """Inizializza la regola e scarica la whitelist dei docenti DIEM."""
        self._diem_docenti_whitelist = self._fetch_docenti_whitelist()

    @property
    def name(self) -> str:
        return "PDF Fuori Perimetro (Whitelist Domini)"

    def _fetch_docenti_whitelist(self) -> Set[str]:
        """Scarica la lista delle matricole dei docenti DIEM dal sito istituzionale."""
        logger.info("Inizializzazione Whitelist Docenti DIEM dal web...")
        url_personale = "https://www.diem.unisa.it/dipartimento/personale"
        whitelist: Set[str] = set()
        try:
            response = requests.get(url_personale, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for a_tag in soup.find_all("a", href=True):
                if "rubrica.unisa.it/persone?matricola=" in a_tag["href"]:
                    match = re.search(r"matricola=(\d+)", a_tag["href"])
                    if match:
                        whitelist.add(
                            f"https://docenti.unisa.it/{match.group(1)}/"
                        )
        except Exception as exc:
            logger.error("Impossibile creare whitelist docenti: %s", exc)
        return whitelist

    def should_discard(self, url: str) -> bool:
        """Restituisce True se il PDF proviene da un dominio o docente non autorizzato."""
        try:
            parsed_url = urlparse(url)
            if parsed_url.netloc not in self._ALLOWED_DOMAINS:
                return True
        except ValueError:
            return True

        if parsed_url.netloc == "docenti.unisa.it":
            if not any(url.startswith(doc) for doc in self._diem_docenti_whitelist):
                return True
        else:
            if not url.startswith(self._ALLOWED_PREFIXES):
                return True

        return False


class EnglishPdfFilterRule(PdfFilterRule):
    """Scarta PDF in lingua inglese identificati dal suffisso '-eng.pdf'."""

    @property
    def name(self) -> str:
        return "Filtro PDF Inglese (-eng.pdf)"

    def should_discard(self, pdf_url: str) -> bool:
        """Restituisce True se l'URL del PDF termina con '-eng.pdf'."""
        return pdf_url.lower().endswith("-eng.pdf")