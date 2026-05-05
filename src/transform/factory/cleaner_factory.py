from pathlib import Path
from typing import List
from transform.core.base_rule import CleaningRule
from transform.rules.content_rules import EmptyBodyRule, NoContentInsertedRule, PageNotFoundRule
from transform.rules.metadata_rules import FilenameRule, ObsoleteUrlRule, CalendarRule, PublicationTipRule, DidatticaFilterRule, ExactPublicationsBaseRule, DepartmentBandiRule

class RuleFactory:
    """Factory per creare le regole di pulizia dinamicamente."""
    
    def __init__(self, directory: Path, cutoff_year: int = 2020, target_department: int = "300638"):
        self.directory = directory
        self.cutoff_year = cutoff_year
        self.target_departement = target_department

    def create_rules(self, rule_names: List[str]) -> List[CleaningRule]:
        """Istanzia e restituisce le regole richieste."""
        rules = []
        for name in rule_names:
            name = name.lower()
            if name == "filename":
                rules.append(FilenameRule())
            elif name == "nocontent":
                rules.append(NoContentInsertedRule())
            elif name == "obsolete_url":
                rules.append(ObsoleteUrlRule(cutoff_year=self.cutoff_year))
            elif name == "publication_tip":
                rules.append(PublicationTipRule())
            elif name == "didattica":
                # La factory sa che questa regola ha bisogno della directory
                rules.append(DidatticaFilterRule(directory=self.directory))
            elif name == "404":
                rules.append(PageNotFoundRule())
            elif name == "empty_body":
                rules.append(EmptyBodyRule())
            # 2. Aggiungi questo nuovo blocco elif
            elif name == "exact_publications":
                rules.append(ExactPublicationsBaseRule())
            elif name == "department_bandi":
                # Puoi passare l'ID del dipartimento come parametro se vuoi renderlo ancora più flessibile
                rules.append(DepartmentBandiRule(target_department=self.target_departement))
            elif name == "calendar":
                rules.append(CalendarRule())
            else:
                raise ValueError(f"Regola sconosciuta: {name}")
                
        return rules