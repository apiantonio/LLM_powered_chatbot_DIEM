"""Factory per la creazione delle regole di filtraggio PDF."""

from typing import List

from scraping.core.base_rule import PdfFilterRule
from scraping.rules.pdf_rules import (
    DomainWhitelistRule,
    EnglishPdfFilterRule,
    ObsoleteYearRule,
    SemanticTrapRule,
)


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
