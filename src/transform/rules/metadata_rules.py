"""
Regole di pulizia basate su metadati (filename/URL).

CODICE RIMOSSO (Sprint Filtri Docenti):
  - ObsoleteUrlRule: logica errata che scartava anno=0 e anno<2020 a livello filename.
    Sostituita da PubblicazioniUrlClassifier in docenti_url_rules.py (Req 2).
  - DidatticaFilterRule: causava race condition tentando di eliminare file
    durante il crawling. Sostituita dal post-processing nel crawler (Req 5).
"""

import re
from pathlib import Path
from typing import Optional
from transform.core.base_rule import CleaningRule


class FilenameRule(CleaningRule):
    """Scarta file con indicatori di lingua straniera nel filename (-en-, -zh-)."""

    @property
    def name(self) -> str:
        return "Filtro Lingua URL/Filename (-en-, -zh-)"

    @property
    def requires_content(self) -> bool:
        return False

    def should_delete(self, filepath: Path, content: Optional[str] = None) -> bool:
        targets = ["-en-", "-zh-", "-zh", "-en.", "_en_", "_zh_"]
        return any(target in filepath.name.lower() for target in targets)


class PublicationTipRule(CleaningRule):
    """Scarta le pagine pubblicazioni filtrate per attributo 'tip='."""

    def __init__(self):
        self.target_pattern = re.compile(r'-\d+-ricerca-pubblicazioni?.*tip=')

    @property
    def name(self) -> str:
        return "Pubblicazioni filtrate per attributo 'tip'"

    @property
    def requires_content(self) -> bool:
        return False

    def should_delete(self, filepath: Path, content: Optional[str] = None) -> bool:
        return bool(self.target_pattern.search(filepath.name))


class ExactPublicationsBaseRule(CleaningRule):
    """Scarta la pagina base delle pubblicazioni (senza parametri anno)."""

    @property
    def name(self) -> str:
        return "Pagina base pubblicazioni (senza parametri aggiuntivi)"

    @property
    def requires_content(self) -> bool:
        return False

    def should_delete(self, filepath: Path, content: Optional[str] = None) -> bool:
        return filepath.name.endswith("ricerca-pubblicazioni.html")


class DepartmentBandiRule(CleaningRule):
    """Scarta i bandi di dipartimenti diversi dal target."""

    def __init__(self, target_department: str = "300638"):
        self.target_department = target_department
        self.struttura_pattern = re.compile(r'(?:cdsS|s)truttura=([^&.\s]+)')

    @property
    def name(self) -> str:
        return f"Filtro Bandi (struttura diversa da {self.target_department})"

    @property
    def requires_content(self) -> bool:
        return False

    def should_delete(self, filepath: Path, content: Optional[str] = None) -> bool:
        filename = filepath.name

        if "-home-bandi-" not in filename:
            return False

        if "truttura=more" in filename:
            return True

        match = self.struttura_pattern.search(filename)

        if not match:
            return True

        struttura_val = match.group(1)
        if struttura_val != self.target_department:
            return True

        return False


class CalendarRule(CleaningRule):
    """Scarta file relativi a calendari."""

    @property
    def name(self) -> str:
        return "File relativo a calendari (singolare o plurale)"

    @property
    def requires_content(self) -> bool:
        return False

    def should_delete(self, filepath: Path, content: Optional[str] = None) -> bool:
        return "calendari" in filepath.name.lower()


class NewsRule(CleaningRule):
    """Scarta file relativi a news/notizie."""

    @property
    def name(self) -> str:
        return "File relativo a news/notizie (contiene 'news')"

    @property
    def requires_content(self) -> bool:
        return False

    def should_delete(self, filepath: Path, content: Optional[str] = None) -> bool:
        return "news" in filepath.name.lower()