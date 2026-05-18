import re
import logging
from urllib.parse import urlparse, parse_qs
logger = logging.getLogger(__name__)


class ProgettiUrlClassifier:

    def classify(self, url: str) -> str:
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

class PubblicazioniUrlClassifier:

    def classify(self, url: str) -> str:
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


class DidatticaOrariUrlClassifier:
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
    def classify(self, url: str) -> str:
        url_lower = url.lower()

        if "docenti.unisa.it/" not in url_lower or "/didattica" not in url_lower:
            return "pass"

        if "id=" in url_lower:
            return "save"

        return "pass"

class RicercaBaseUrlClassifier:
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
    _international_pattern = re.compile(
        r'^https?://'
        r'(?:(?:www\.)?diem\.unisa\.it|docenti\.unisa\.it/\d+)'
        r'/international/?$',
        re.IGNORECASE,
    )

    def classify(self, url: str) -> str:
        clean_url = url.split('?')[0].split('#')[0]
        if self._international_pattern.match(clean_url):
            return "navigate"
        return "pass"
    

  

class InternationalSubpagesUrlClassifier:
    _target_paths = (
        "/international/dottorato-con-tesi-in-cotutela",
        "/international/doppio-titolo",
        "/international/traineeship",
        "/international/cooperazione-internazionale"
    )

    def classify(self, url: str) -> str:
        url_lower = url.lower()

        if "docenti.unisa.it/" not in url_lower:
            return "pass"

        parsed = urlparse(url)
        path = parsed.path.rstrip("/")

        if not any(path.lower().endswith(target) for target in self._target_paths):
            return "pass"

        query_params = parse_qs(parsed.query, keep_blank_values=True)

        if not query_params:
            return "navigate"

        if (
            len(query_params) == 2 and
            "anno" in query_params and query_params["anno"] == [""] and
            "stato" in query_params and query_params["stato"] == ["tutti"]
        ):
            return "save"

        return "discard"