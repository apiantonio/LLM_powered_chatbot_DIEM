"""Factory per la creazione delle regole di filtraggio HTML e PDF.

Contiene:
- RuleFactory: istanzia regole di pulizia HTML (CleaningRule) a partire da nomi simbolici.
- PdfRuleFactory: istanzia regole di filtraggio PDF (PdfFilterRule) a partire da nomi simbolici.
- get_all_html_rules / get_all_pdf_rules: helper di convenienza che producono l'insieme completo
  delle regole con un singolo import.
"""

from pathlib import Path
from typing import List

from scraping.interfaces import CleaningRule, PdfFilterRule
from scraping.rules.html_content import (
    CalendarRule,
    DepartmentBandiRule,
    EmptyBodyRule,
    ExactPublicationsBaseRule,
    FilenameRule,
    NewsRule,
    NoContentInsertedRule,
    PageNotFoundRule,
    PublicationTipRule,
)
from scraping.rules.pdf import (
    DomainWhitelistRule,
    EnglishPdfFilterRule,
    ObsoleteYearRule,
    SemanticTrapRule,
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


class PdfRuleFactory:
    """Factory che istanzia regole di filtraggio PDF a partire da nomi simbolici.

    Implementa il pattern Factory Method: dato un elenco di identificativi
    stringa, produce la lista corrispondente di oggetti PdfFilterRule.
    """

    _RULE_REGISTRY = {
        "semantic_trap": lambda _: SemanticTrapRule(),
        "domain_whitelist": lambda _: DomainWhitelistRule(),
        "english_pdf": lambda _: EnglishPdfFilterRule(),
    }

    def __init__(self, cutoff_year: int = 2020):
        """Inizializza la factory con l'anno di cutoff.

        Args:
            cutoff_year: Anno minimo accettabile per le regole temporali.
        """
        self.cutoff_year = cutoff_year

    def create_rules(self, rule_names: List[str]) -> List[PdfFilterRule]:
        """Crea e restituisce le regole corrispondenti ai nomi forniti.

        Args:
            rule_names: Lista di identificativi delle regole da istanziare.

        Returns:
            Lista ordinata di PdfFilterRule.

        Raises:
            ValueError: Se un nome non corrisponde a nessuna regola registrata.
        """
        rules: List[PdfFilterRule] = []
        for name in rule_names:
            key = name.lower()
            if key == "obsolete_year":
                rules.append(ObsoleteYearRule(cutoff_year=self.cutoff_year))
            elif key in self._RULE_REGISTRY:
                rules.append(self._RULE_REGISTRY[key](self))
            else:
                raise ValueError(f"Regola PDF sconosciuta: {name}")
        return rules


def get_all_html_rules(**kwargs) -> List[CleaningRule]:
    """Crea e restituisce tutte le regole di pulizia HTML con parametri configurabili.

    Args:
        **kwargs: Parametri opzionali (directory, cutoff_year, target_department).

    Returns:
        Lista completa delle CleaningRule.
    """
    directory = Path(kwargs.get("directory", "data/raw/html_samples"))
    cutoff_year = kwargs.get("cutoff_year", 2020)
    target_department = kwargs.get("target_department", "300638")

    factory = RuleFactory(
        directory=directory,
        cutoff_year=cutoff_year,
        target_department=target_department,
    )
    return factory.create_rules([
        "publication_tip",
        "exact_publications",
        "department_bandi",
        "calendar",
        "news",
        "404",
        "nocontent",
        "empty_body",
        "filename",
    ])


def get_all_pdf_rules(**kwargs) -> List[PdfFilterRule]:
    """Crea e restituisce tutte le regole di filtraggio PDF con parametri configurabili.

    Args:
        **kwargs: Parametri opzionali (cutoff_year).

    Returns:
        Lista completa delle PdfFilterRule.
    """
    cutoff_year = kwargs.get("cutoff_year", 2020)
    factory = PdfRuleFactory(cutoff_year=cutoff_year)
    return factory.create_rules([
        "domain_whitelist",
        "semantic_trap",
        "obsolete_year",
        "english_pdf",
    ])