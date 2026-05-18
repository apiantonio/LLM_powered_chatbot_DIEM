import re
import requests
import logging
from typing import Set
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from scraping.core.base_rule import PdfFilterRule

logger = logging.getLogger(__name__)


class ObsoleteYearRule(PdfFilterRule):
    def __init__(self, cutoff_year: int = 2020):
        self.cutoff_year = cutoff_year

        self.academic_pattern = re.compile(r'(?<!\d)(19\d{2}|20\d{2})[-_](19\d{2}|20\d{2})(?!\d)')
        self.year_month_pattern = re.compile(r'(?<!\d)(19\d{2}|20\d{2})[-_](0[1-9]|1[0-2])(?!\d)')

        self.single_year_pattern = re.compile(r'(?<!\d)(19\d{2}|20\d{2})(?!\d)')

        self.doc_id_year_pattern = re.compile(
            r'doc\d{6}(\d{4})',
            re.IGNORECASE,
        )

        self.ddmmyyyy_pattern = re.compile(
            r'(?<!\d)'
            r'(?:0[1-9]|[12]\d|3[01])'
            r'(?:0[1-9]|1[0-2])'
            r'(19\d{2}|20\d{2})'
            r'(?!\d)',
        )

    @property
    def name(self) -> str:
        return f"Filtro Anni Obsoleti (< {self.cutoff_year})"

    def should_discard(self, pdf_url: str) -> bool:
        logical_years = []
        working_url = pdf_url

        for match in self.academic_pattern.finditer(working_url):
            y1 = int(match.group(1))
            logical_years.append(y1)
        working_url = self.academic_pattern.sub('XXX', working_url)

        for match in self.year_month_pattern.finditer(working_url):
            y1 = int(match.group(1))
            logical_years.append(y1)
        working_url = self.year_month_pattern.sub('XXX', working_url)

        for match in self.doc_id_year_pattern.finditer(working_url):
            year = int(match.group(1))
            logical_years.append(year)
        working_url = self.doc_id_year_pattern.sub('XXX', working_url)

        for match in self.ddmmyyyy_pattern.finditer(working_url):
            year = int(match.group(1))
            logical_years.append(year)
        working_url = self.ddmmyyyy_pattern.sub('XXX', working_url)

        for match in self.single_year_pattern.finditer(working_url):
            y1 = int(match.group(1))
            logical_years.append(y1)

        if not logical_years:
            return False

        max_year = max(logical_years)
        return max_year < self.cutoff_year


class SemanticTrapRule(PdfFilterRule):

    def __init__(self):
        self.traps = [
            "grad_", "graduatori", "esit", "risultat", "ammess", "verbale", "verbali",
            "decreto", "approvazione_atti", "commissione", "contratt", "incarico",
            "valutazione", "scorrimento", "elenco", "candidat", "modulo", "richiesta", "domanda"
        ]

    @property
    def name(self) -> str: return "PDF Burocratico/Dati Sensibili"

    def should_discard(self, url: str) -> bool:
        url_lower = url.lower()
        return any(trap in url_lower for trap in self.traps)


class DomainWhitelistRule(PdfFilterRule):

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

    @property
    def name(self) -> str: return "PDF Fuori Perimetro (Whitelist Domini)"

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


class EnglishPdfFilterRule(PdfFilterRule):
    @property
    def name(self) -> str:
        return "Filtro PDF Inglese (-eng.pdf)"

    def should_discard(self, pdf_url: str) -> bool:
        return pdf_url.lower().endswith('-eng.pdf')