"""Classi base astratte per le regole di filtraggio HTML e PDF."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


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
