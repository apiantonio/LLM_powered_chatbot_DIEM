"""
Regole di filtraggio URL per la sezione docenti.unisa.it.

Incapsula le logiche di classificazione URL per:
  - Ricerca/Progetti (Req 1): parsing rigoroso della query string
  - Ricerca/Pubblicazioni (Req 2): solo anno=0 ammesso
  - Didattica/Orari (Req 3): scarto completo
  - Didattica id/cId/pId (Req 5): classificazione per post-processing
  - Ricerca base docenti (Req 6): navigate-only (estrarre link, non salvare)
  - International docenti/DIEM (Req 7): navigate-only (estrarre link, non salvare)

Design: Strategy Pattern — ogni classificatore è una classe indipendente
con interfaccia `classify(url) -> str`. Il crawler li inietta e li invoca
senza conoscere i dettagli implementativi.

NOTA: Queste regole operano a livello di URL (pre-accodamento e pre-salvataggio),
NON a livello di contenuto HTML. Il filtraggio DOM è in html_content_rules.py.
"""

import re
import logging
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)


class ProgettiUrlClassifier:
    """
    Classificatore URL per la sezione ricerca/progetti (Req 1).

    Logica con parsing rigoroso della query string:
      - "save":     URL con ESCLUSIVAMENTE ruolo=tutti (nessun altro parametro)
      - "navigate": pagina padre /ricerca/progetti senza query params → estrarre link, NON salvare
      - "discard":  qualsiasi URL con parametri aggiuntivi oltre ruolo=tutti
      - "pass":     URL non appartenente a ricerca/progetti
    """

    def classify(self, url: str) -> str:
        url_lower = url.lower()

        if "docenti.unisa.it/" not in url_lower or "/ricerca/progetti" not in url_lower:
            return "pass"

        parsed = urlparse(url)
        path = parsed.path.rstrip("/")

        # Verifica che il path termini con /ricerca/progetti
        if not path.lower().endswith("/ricerca/progetti"):
            return "pass"

        # Parse rigoroso della query string
        query_params = parse_qs(parsed.query, keep_blank_values=True)

        # --- Nessun parametro → pagina padre → navigate-only ---
        if not query_params:
            return "navigate"

        # --- ESCLUSIVAMENTE ruolo=tutti → SALVARE ---
        # La query string deve contenere SOLO il parametro "ruolo" con valore "tutti"
        if (
            len(query_params) == 1
            and "ruolo" in query_params
            and query_params["ruolo"] == ["tutti"]
        ):
            return "save"

        # --- Qualsiasi altro parametro o combinazione → SCARTARE ---
        # Copre: ruolo=tutti&tip=9, ruolo=componente, stato=..., progetto=..., ecc.
        return "discard"


class PubblicazioniUrlClassifier:
    """
    Classificatore URL per la sezione ricerca/pubblicazioni (Req 2).

    Logica:
      - "save":    URL con anno=0 → salvare (con filtraggio DOM successivo)
      - "discard": URL con qualsiasi altro anno=YYYY → scartare completamente
      - "pass":    URL non appartenente a ricerca/pubblicazioni
    """

    _anno_pattern = re.compile(r'anno=(\d+)', re.IGNORECASE)

    def classify(self, url: str) -> str:
        url_lower = url.lower()

        if "docenti.unisa.it/" not in url_lower or "/ricerca/pubblicazioni" not in url_lower:
            return "pass"

        # anno=0 → SALVARE (il filtraggio DOM avviene separatamente)
        if "anno=0" in url_lower:
            return "save"

        # Qualsiasi altro anno=YYYY → SCARTARE
        if self._anno_pattern.search(url):
            return "discard"

        # Pagina base senza parametro anno → pass (gestita da altre regole)
        return "pass"


class DidatticaOrariUrlClassifier:
    """
    Classificatore URL per la sezione didattica/orari (Req 3).

    Logica:
      - "discard": URL contenente la stringa /didattica/orari o equivalenti
                   nel filename sanitizzato (-didattica-orari)
      - "pass":    URL non corrispondente
    """

    # Pattern che matchano sia la forma URL che la forma filename sanitizzata
    _discard_patterns = (
        "/didattica/orari",
        "-didattica-orari.html",
        "-didattica-orari-include=docente.html",
        "-didattica-orari-include=docente",
    )

    def classify(self, url: str) -> str:
        url_lower = url.lower()
        for pattern in self._discard_patterns:
            if pattern in url_lower:
                return "discard"
        return "pass"


class DidatticaIdUrlClassifier:
    """
    Classificatore URL per le pagine didattica con parametri id/cId/pId (Req 5).

    Logica:
      - "save_and_mark": solo id= (senza cId/pId) → salvare ora, eliminare in post-processing
      - "save":          id + cId + pId → versione completa → salvare
      - "discard":       id + cId senza pId → versione incompleta → scartare
      - "pass":          non è un URL didattica con id=
    """

    _id_pattern = re.compile(r'id=(\d+)', re.IGNORECASE)

    def classify(self, url: str) -> str:
        url_lower = url.lower()

        if "docenti.unisa.it/" not in url_lower or "/didattica" not in url_lower:
            return "pass"

        has_id = "id=" in url_lower
        has_cid = "cid=" in url_lower
        has_pid = "pid=" in url_lower

        if has_id and has_cid and not has_pid:
            return "discard"

        if has_id and has_cid and has_pid:
            return "save"

        if has_id and not has_cid and not has_pid:
            return "save_and_mark"

        return "pass"

    def extract_id(self, url: str) -> str | None:
        """Estrae il valore numerico di id= dall'URL."""
        match = self._id_pattern.search(url)
        return match.group(1) if match else None


# ============================================================
# NUOVI CLASSIFICATORI (Sprint Filtri v2)
# ============================================================


class RicercaBaseUrlClassifier:
    """
    Classificatore URL per la pagina base /ricerca dei docenti e del DIEM (Req 6).

    Matcha:
      - docenti.unisa.it/{matricola}/ricerca  (pagina indice ricerca docente)
      - diem.unisa.it/ricerca                 (pagina indice ricerca DIEM)
      - www.diem.unisa.it/ricerca

    NON matcha (devono passare):
      - /ricerca/progetti, /ricerca/pubblicazioni, /ricerca/qualsiasi-sotto-pagina

    Logica:
      - "navigate": pagina indice → estrarre link figli, NON salvare su disco
      - "pass":     URL non corrispondente
    """

    # Matcha path che termina esattamente con /ricerca (senza sotto-path)
    _ricerca_base_pattern = re.compile(
        r'^https?://'
        r'(?:(?:www\.)?diem\.unisa\.it|docenti\.unisa\.it/\d+)'
        r'/ricerca/?$',
        re.IGNORECASE,
    )

    def classify(self, url: str) -> str:
        # Rimuovi query string e fragment per un match pulito
        clean_url = url.split('?')[0].split('#')[0]
        if self._ricerca_base_pattern.match(clean_url):
            return "navigate"
        return "pass"

class InternationalUrlClassifier:
    """
    Classificatore URL per le pagine base /international dei docenti e del DIEM (Req 7).

    Matcha esclusivamente le pagine indice:
      - docenti.unisa.it/{matricola}/international  (pagina international base docente)
      - diem.unisa.it/international                 (pagina international base DIEM)
      - www.diem.unisa.it/international

    NON matcha le sottopagine (es. /international/traineeship, /international/dottorato...), 
    che devono passare il filtro ed essere salvate su disco.

    Logica:
      - "navigate": pagina international base → estrarre link figli, NON salvare su disco
      - "pass":     URL non corrispondente o sottopagina → procedere con il salvataggio
    """

    # Matcha il path che termina esattamente con /international (senza sotto-path)
    _international_pattern = re.compile(
        r'^https?://'
        r'(?:(?:www\.)?diem\.unisa\.it|docenti\.unisa\.it/\d+)'
        r'/international/?$',
        re.IGNORECASE,
    )

    def classify(self, url: str) -> str:
        # Rimuovi query string e fragment per un match pulito del solo path
        clean_url = url.split('?')[0].split('#')[0]
        if self._international_pattern.match(clean_url):
            return "navigate"
        return "pass"