"""Classe base e pipeline per i classificatori di URL.

Introduce il pattern Strategy con un Composite (UrlClassificationPipeline)
che permette di iterare su piu classificatori in sequenza, restituendo
la prima decisione diversa da 'pass'.
"""

from abc import ABC, abstractmethod
from typing import List


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
