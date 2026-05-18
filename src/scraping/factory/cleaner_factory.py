"""Factory per la creazione delle regole di pulizia HTML."""

from pathlib import Path
from typing import List

from scraping.core.base_rule import CleaningRule
from scraping.rules.content_rules import (
    EmptyBodyRule,
    NoContentInsertedRule,
    PageNotFoundRule,
)
from scraping.rules.metadata_rules import (
    CalendarRule,
    DepartmentBandiRule,
    ExactPublicationsBaseRule,
    FilenameRule,
    NewsRule,
    PublicationTipRule,
)


class RuleFactory:
    """Factory che istanzia regole di pulizia HTML a partire da nomi simbolici.

    Implementa il pattern Factory Method: dato un elenco di identificativi
    stringa, produce la lista corrispondente di oggetti CleaningRule.
    """

    _RULE_REGISTRY = {
        "filename": lambda _: FilenameRule(),
        "nocontent": lambda _: NoContentInsertedRule(),
        "publication_tip": lambda _: PublicationTipRule(),
        "404": lambda _: PageNotFoundRule(),
        "empty_body": lambda _: EmptyBodyRule(),
        "exact_publications": lambda _: ExactPublicationsBaseRule(),
        "calendar": lambda _: CalendarRule(),
        "news": lambda _: NewsRule(),
    }

    def __init__(
        self,
        directory: Path,
        cutoff_year: int = 2020,
        target_department: str = "300638",
    ):
        """Inizializza la factory con i parametri di contesto.

        Args:
            directory: Directory dei file HTML grezzi.
            cutoff_year: Anno di cutoff per le regole temporali.
            target_department: Codice struttura del dipartimento target.
        """
        self.directory = directory
        self.cutoff_year = cutoff_year
        self.target_department = target_department

    def create_rules(self, rule_names: List[str]) -> List[CleaningRule]:
        """Crea e restituisce le regole corrispondenti ai nomi forniti.

        Args:
            rule_names: Lista di identificativi delle regole da istanziare.

        Returns:
            Lista ordinata di CleaningRule.

        Raises:
            ValueError: Se un nome non corrisponde a nessuna regola registrata.
        """
        rules: List[CleaningRule] = []
        for name in rule_names:
            key = name.lower()
            if key == "department_bandi":
                rules.append(
                    DepartmentBandiRule(target_department=self.target_department)
                )
            elif key in self._RULE_REGISTRY:
                rules.append(self._RULE_REGISTRY[key](self))
            else:
                raise ValueError(f"Regola sconosciuta: {name}")
        return rules
