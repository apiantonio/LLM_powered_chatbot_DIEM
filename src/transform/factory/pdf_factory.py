from typing import List
from transform.core.base_rule import PdfFilterRule
from transform.rules.pdf_rules import (
    ObsoleteYearRule, 
    SemanticTrapRule, 
    DomainWhitelistRule,
)

class PdfRuleFactory:
    """Factory per creare le regole di filtraggio PDF dinamicamente."""
    
    def __init__(self, cutoff_year: int = 2020):
        self.cutoff_year = cutoff_year

    def create_rules(self, rule_names: List[str]) -> List[PdfFilterRule]:
        """Istanzia e restituisce le regole richieste per i PDF."""
        rules = []
        for name in rule_names:
            name = name.lower()
            if name == "obsolete_year":
                rules.append(ObsoleteYearRule(cutoff_year=self.cutoff_year))
            elif name == "semantic_trap":
                rules.append(SemanticTrapRule())
            elif name == "domain_whitelist":
                rules.append(DomainWhitelistRule())
            else:
                raise ValueError(f"Regola PDF sconosciuta: {name}")
                
        return rules