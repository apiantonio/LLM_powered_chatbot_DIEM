"""Regole di pulizia basate sui metadati del file (nome, URL codificato nel filename)."""

import re
from pathlib import Path
from typing import Optional

from scraping.core.base_rule import CleaningRule


class FilenameRule(CleaningRule):
    """Filtra pagine in lingua straniera identificate da marcatori nel filename."""

    _LANGUAGE_MARKERS = ("-en-", "-zh-", "-zh", "-en.", "_en_", "_zh_")

    @property
    def name(self) -> str:
        return "Filtro Lingua URL/Filename (-en-, -zh-)"

    @property
    def requires_content(self) -> bool:
        return False

    def should_delete(self, filepath: Path, content: Optional[str] = None) -> bool:
        """Restituisce True se il filename contiene marcatori di lingua non italiana."""
        return any(marker in filepath.name.lower() for marker in self._LANGUAGE_MARKERS)


class PublicationTipRule(CleaningRule):
    """Filtra pagine di pubblicazioni filtrate per attributo 'tip' nell'URL."""

    _TARGET_PATTERN = re.compile(r"-\d+-ricerca-pubblicazioni?.*tip=")

    @property
    def name(self) -> str:
        return "Pubblicazioni filtrate per attributo 'tip'"

    @property
    def requires_content(self) -> bool:
        return False

    def should_delete(self, filepath: Path, content: Optional[str] = None) -> bool:
        """Restituisce True se il filename corrisponde al pattern delle pubblicazioni filtrate."""
        return bool(self._TARGET_PATTERN.search(filepath.name))


class ExactPublicationsBaseRule(CleaningRule):
    """Filtra la pagina base delle pubblicazioni priva di parametri aggiuntivi."""

    @property
    def name(self) -> str:
        return "Pagina base pubblicazioni (senza parametri aggiuntivi)"

    @property
    def requires_content(self) -> bool:
        return False

    def should_delete(self, filepath: Path, content: Optional[str] = None) -> bool:
        """Restituisce True se il file termina con il suffisso della pagina base pubblicazioni."""
        return filepath.name.endswith("ricerca-pubblicazioni.html")


class DepartmentBandiRule(CleaningRule):
    """Filtra i bandi relativi a strutture diverse dal dipartimento target."""

    _STRUTTURA_PATTERN = re.compile(r"(?:cdsS|s)truttura=([^&.\s]+)")

    def __init__(self, target_department: str = "300638"):
        """Inizializza la regola con il codice struttura del dipartimento target.

        Args:
            target_department: Codice struttura del dipartimento da preservare.
        """
        self.target_department = target_department

    @property
    def name(self) -> str:
        return f"Filtro Bandi (struttura diversa da {self.target_department})"

    @property
    def requires_content(self) -> bool:
        return False

    def should_delete(self, filepath: Path, content: Optional[str] = None) -> bool:
        """Restituisce True se il bando appartiene a una struttura diversa dal target."""
        filename = filepath.name

        if "-home-bandi-" not in filename:
            return False

        if "truttura=more" in filename:
            return True

        match = self._STRUTTURA_PATTERN.search(filename)

        if not match:
            return True

        struttura_val = match.group(1)
        if struttura_val != self.target_department:
            return True

        return False


class CalendarRule(CleaningRule):
    """Filtra file relativi a calendari."""

    @property
    def name(self) -> str:
        return "File relativo a calendari (singolare o plurale)"

    @property
    def requires_content(self) -> bool:
        return False

    def should_delete(self, filepath: Path, content: Optional[str] = None) -> bool:
        """Restituisce True se il filename contiene riferimenti a calendari."""
        return "calendari" in filepath.name.lower()


class NewsRule(CleaningRule):
    """Filtra file relativi a news/notizie."""

    @property
    def name(self) -> str:
        return "File relativo a news/notizie (contiene 'news')"

    @property
    def requires_content(self) -> bool:
        return False

    def should_delete(self, filepath: Path, content: Optional[str] = None) -> bool:
        """Restituisce True se il filename contiene il marcatore 'news'."""
        return "news" in filepath.name.lower()
