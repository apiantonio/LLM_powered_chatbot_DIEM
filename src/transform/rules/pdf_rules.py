# src/inspection/rules/pdf_rules.py
import re
import requests
import logging
from typing import Set
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from transform.core.base_rule import PdfFilterRule

logger = logging.getLogger(__name__)

class ObsoleteYearRule(PdfFilterRule):
    """Scarta i PDF che appartengono ad anni precedenti al cutoff_year (es. 2020)."""
    def __init__(self, cutoff_year: int = 2020):
        self.cutoff_year = cutoff_year
        # Intercetta tutti gli anni a 4 cifre che iniziano con 19 o 20
        self.year_pattern = re.compile(r'\b(19\d{2}|20\d{2})\b')

    @property
    def name(self) -> str: return f"PDF Obsoleto (< {self.cutoff_year})"

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

class SemanticTrapRule(PdfFilterRule):
    """Scarta i PDF puramente burocratici o contenenti dati sensibili in base a keyword."""
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