"""Classificatore di URL per il dominio corsi.unisa.it."""

from scraping.core.url_classifier import UrlClassifier


class CorsiUrlClassifier(UrlClassifier):
    """Classifica URL del dominio corsi.unisa.it in base al path."""

    def classify(self, url: str) -> str:
        """Restituisce 'navigate' per pagine di piano di studi o regolamenti, 'pass' altrimenti."""
        url_lower = url.lower()

        if "corsi.unisa.it" in url_lower:
            if "piano-di-studi" in url_lower or "regolamenti" in url_lower:
                return "navigate"

        return "pass"
