"""Classificatori di URL per il dominio docenti.unisa.it e www.diem.unisa.it."""

import re
from urllib.parse import urlparse, parse_qs

from scraping.core.url_classifier import UrlClassifier


class ProgettiUrlClassifier(UrlClassifier):
    """Classifica le pagine dei progetti di ricerca dei docenti."""

    def classify(self, url: str) -> str:
        """Restituisce la decisione per URL di tipo /ricerca/progetti.

        - 'navigate': pagina base senza query string.
        - 'save': pagina con ruolo=tutti.
        - 'discard': qualsiasi altra variante con parametri.
        - 'pass': URL non pertinente.
        """
        url_lower = url.lower()

        if "docenti.unisa.it/" not in url_lower or "/ricerca/progetti" not in url_lower:
            return "pass"

        parsed = urlparse(url)
        path = parsed.path.rstrip("/")

        if not path.lower().endswith("/ricerca/progetti"):
            return "pass"

        query_params = parse_qs(parsed.query, keep_blank_values=True)

        if not query_params:
            return "navigate"

        if (
            len(query_params) == 1
            and "ruolo" in query_params
            and query_params["ruolo"] == ["tutti"]
        ):
            return "save"

        return "discard"


class PubblicazioniUrlClassifier(UrlClassifier):
    """Classifica le pagine delle pubblicazioni dei docenti."""

    def classify(self, url: str) -> str:
        """Restituisce la decisione per URL di tipo /ricerca/pubblicazioni.

        - 'navigate': pagina base senza query string.
        - 'save': pagina con anno=0 (tutte le pubblicazioni).
        - 'discard': pagina con anno specifico.
        - 'pass': URL non pertinente.
        """
        url_lower = url.lower()

        if "docenti.unisa.it/" not in url_lower or "/ricerca/pubblicazioni" not in url_lower:
            return "pass"

        parsed = urlparse(url)
        path = parsed.path.rstrip("/")

        if not path.lower().endswith("/ricerca/pubblicazioni"):
            return "pass"

        query_params = parse_qs(parsed.query, keep_blank_values=True)

        if not query_params:
            return "navigate"

        if query_params.get("anno") == ["0"]:
            return "save"

        if "anno" in query_params:
            return "discard"

        return "pass"


class DidatticaOrariUrlClassifier(UrlClassifier):
    """Classifica le pagine degli orari di didattica (da scartare)."""

    _DISCARD_PATTERNS = (
        "/didattica/orari",
        "-didattica-orari.html",
        "-didattica-orari-include=docente.html",
        "-didattica-orari-include=docente",
    )

    def classify(self, url: str) -> str:
        """Restituisce 'discard' se l'URL corrisponde a un pattern di orari, 'pass' altrimenti."""
        url_lower = url.lower()
        for pattern in self._DISCARD_PATTERNS:
            if pattern in url_lower:
                return "discard"
        return "pass"


class DidatticaIdUrlClassifier(UrlClassifier):
    """Classifica le pagine di didattica contenenti un parametro id."""

    def classify(self, url: str) -> str:
        """Restituisce 'save' se l'URL di didattica contiene 'id=', 'pass' altrimenti."""
        url_lower = url.lower()

        if "docenti.unisa.it/" not in url_lower or "/didattica" not in url_lower:
            return "pass"

        if "id=" in url_lower:
            return "save"

        return "pass"


class RicercaBaseUrlClassifier(UrlClassifier):
    """Classifica la pagina base /ricerca dei docenti e del dipartimento."""

    _RICERCA_BASE_PATTERN = re.compile(
        r"^https?://"
        r"(?:(?:www\.)?diem\.unisa\.it|docenti\.unisa\.it/\d+)"
        r"/ricerca/?$",
        re.IGNORECASE,
    )

    def classify(self, url: str) -> str:
        """Restituisce 'navigate' per la pagina base di ricerca, 'pass' altrimenti."""
        clean_url = url.split("?")[0].split("#")[0]
        if self._RICERCA_BASE_PATTERN.match(clean_url):
            return "navigate"
        return "pass"


class InternationalUrlClassifier(UrlClassifier):
    """Classifica la pagina base /international dei docenti e del dipartimento."""

    _INTERNATIONAL_PATTERN = re.compile(
        r"^https?://"
        r"(?:(?:www\.)?diem\.unisa\.it|docenti\.unisa\.it/\d+)"
        r"/international/?$",
        re.IGNORECASE,
    )

    def classify(self, url: str) -> str:
        """Restituisce 'navigate' per la pagina base international, 'pass' altrimenti."""
        clean_url = url.split("?")[0].split("#")[0]
        if self._INTERNATIONAL_PATTERN.match(clean_url):
            return "navigate"
        return "pass"


class InternationalSubpagesUrlClassifier(UrlClassifier):
    """Classifica le sotto-pagine di /international con parametri specifici."""

    _TARGET_PATHS = (
        "/international/dottorato-con-tesi-in-cotutela",
        "/international/doppio-titolo",
        "/international/traineeship",
        "/international/cooperazione-internazionale",
    )

    def classify(self, url: str) -> str:
        """Restituisce la decisione per le sotto-pagine international.

        - 'navigate': pagina senza query string.
        - 'save': pagina con anno vuoto e stato=tutti.
        - 'discard': qualsiasi altra combinazione di parametri.
        - 'pass': URL non pertinente.
        """
        url_lower = url.lower()

        if "docenti.unisa.it/" not in url_lower:
            return "pass"

        parsed = urlparse(url)
        path = parsed.path.rstrip("/")

        if not any(path.lower().endswith(target) for target in self._TARGET_PATHS):
            return "pass"

        query_params = parse_qs(parsed.query, keep_blank_values=True)

        if not query_params:
            return "navigate"

        if (
            len(query_params) == 2
            and "anno" in query_params
            and query_params["anno"] == [""]
            and "stato" in query_params
            and query_params["stato"] == ["tutti"]
        ):
            return "save"

        return "discard"
