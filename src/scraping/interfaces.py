"""Astrazioni del modulo di scraping.

Contiene le interfacce per:
- regole di filtraggio HTML (CleaningRule)
- regole di filtraggio PDF (PdfFilterRule)
- classificatori di URL (UrlClassifier) e pipeline composita (UrlClassificationPipeline)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional


class CleaningRule(ABC):
    """Regola astratta per il filtraggio e la pulizia di file HTML."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identificativo descrittivo della regola."""

    @property
    def requires_content(self) -> bool:
        """Indica se la regola necessita del contenuto del file per la valutazione."""
        return True

    @abstractmethod
    def should_delete(self, filepath: Path, content: Optional[str] = None) -> bool:
        """Determina se il file deve essere eliminato.

        Args:
            filepath: Percorso del file da valutare.
            content: Contenuto testuale del file (opzionale).

        Returns:
            True se il file deve essere scartato.
        """


class PdfFilterRule(ABC):
    """Regola astratta per il filtraggio di URL che puntano a PDF."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identificativo descrittivo della regola."""

    @abstractmethod
    def should_discard(self, url: str) -> bool:
        """Determina se il PDF individuato dall'URL deve essere scartato.

        Args:
            url: URL del PDF da valutare.

        Returns:
            True se il PDF deve essere scartato.
        """


class UrlClassifier(ABC):
    """Strategia astratta per la classificazione di un URL.

    Ogni implementazione restituisce una tra le seguenti decisioni:
    - 'pass': la regola non si applica a questo URL.
    - 'navigate': l'URL deve essere visitato senza salvare la pagina.
    - 'save': l'URL deve essere visitato e la pagina salvata.
    - 'discard': l'URL deve essere scartato.
    """

    @abstractmethod
    def classify(self, url: str) -> str:
        """Classifica l'URL e restituisce la decisione."""


class UrlClassificationPipeline:
    """Pipeline composita che applica una sequenza di UrlClassifier.

    Itera sui classificatori registrati e restituisce la prima decisione
    diversa da 'pass'. Se nessun classificatore intercetta l'URL,
    restituisce 'pass'.
    """

    def __init__(self, classifiers: List[UrlClassifier]) -> None:
        """Inizializza la pipeline con la lista ordinata di classificatori.

        Args:
            classifiers: Sequenza di UrlClassifier da applicare in ordine.
        """
        self._classifiers = list(classifiers)

    def classify(self, url: str) -> str:
        """Applica i classificatori in sequenza e restituisce la prima decisione non-pass.

        Args:
            url: URL da classificare.

        Returns:
            Decisione del primo classificatore che non restituisce 'pass',
            oppure 'pass' se nessuno intercetta l'URL.
        """
        for classifier in self._classifiers:
            decision = classifier.classify(url)
            if decision != "pass":
                return decision
        return "pass"