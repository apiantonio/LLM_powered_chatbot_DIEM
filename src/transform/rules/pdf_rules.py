# src/inspection/rules/pdf_rules.py
import re
import requests
import logging
from typing import Set
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from transform.core.base_rule import PdfFilterRule

logger = logging.getLogger(__name__)

import re
from transform.core.base_rule import PdfFilterRule

class ObsoleteYearRule(PdfFilterRule):
    def __init__(self, cutoff_year: int = 2020):
        self.cutoff_year = cutoff_year
        
        # Pattern 1: Anno Accademico (es. 2019-2020, 2010_2011)
        # Cattura il primo anno in group(1) e il secondo in group(2)
        self.academic_pattern = re.compile(r'(?<!\d)(19\d{2}|20\d{2})[-_](19\d{2}|20\d{2})(?!\d)')
        
        # Pattern 2: Anno_Mese (es. 2017_09, 2018-01)
        # Cattura l'anno in group(1)
        self.year_month_pattern = re.compile(r'(?<!\d)(19\d{2}|20\d{2})[-_](0[1-9]|1[0-2])(?!\d)')
        
        # Pattern 3: Anno singolo (es. 2019, 2020, bandotesi2019)
        self.single_year_pattern = re.compile(r'(?<!\d)(19\d{2}|20\d{2})(?!\d)')

    @property
    def name(self) -> str:
        return f"Filtro Anni Obsoleti (< {self.cutoff_year})"

    def should_discard(self, pdf_url: str) -> bool:
        logical_years = []
        working_url = pdf_url

        # --- STEP 1: Estrazione Anni Accademici ---
        # Es. "2019-2020" appartiene al ciclo 2019. Estraiamo 2019.
        for match in self.academic_pattern.finditer(working_url):
            y1 = int(match.group(1))
            logical_years.append(y1)
        # Sostituiamo il match con XXX per "bruciarlo", così il pattern degli anni 
        # singoli non intercetterà erroneamente il "2020" dal "2019-2020"
        working_url = self.academic_pattern.sub('XXX', working_url)

        # --- STEP 2: Estrazione Anno_Mese ---
        for match in self.year_month_pattern.finditer(working_url):
            y1 = int(match.group(1))
            logical_years.append(y1)
        working_url = self.year_month_pattern.sub('XXX', working_url)

        # --- STEP 3: Estrazione Anni Singoli Rimanenti ---
        for match in self.single_year_pattern.finditer(working_url):
            y1 = int(match.group(1))
            logical_years.append(y1)

        # Se non ci sono anni nell'URL, per sicurezza non scartiamo
        if not logical_years:
            return False

        # Valutiamo in base all'anno PIÙ RECENTE trovato nell'URL.
        # Es: se la cartella è /2019/ ma il file è bando_2020.pdf, prevarrà il 2020.
        max_year = max(logical_years)
        return max_year < self.cutoff_year
    
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
    

class EnglishPdfFilterRule(PdfFilterRule):
    """Filtra e scarta tutti i link PDF che terminano con '-eng.pdf'."""
    
    @property
    def name(self) -> str:
        return "Filtro PDF Inglese (-eng.pdf)"

    def should_discard(self, pdf_url: str) -> bool:
        # Usiamo .lower() per essere sicuri di intercettare anche -ENG.pdf o -Eng.pdf
        return pdf_url.lower().endswith('-eng.pdf')