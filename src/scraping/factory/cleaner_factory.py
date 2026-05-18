from pathlib import Path
from typing import List
from scraping.core.base_rule import CleaningRule
from scraping.rules.content_rules import (
    EmptyBodyRule,
    NoContentInsertedRule,
    PageNotFoundRule,
)
from scraping.rules.metadata_rules import (
    FilenameRule,
    NewsRule,
    CalendarRule,
    PublicationTipRule,
    ExactPublicationsBaseRule,
    DepartmentBandiRule,
)


class RuleFactory:

    def __init__(
        self,
        directory: Path,
        cutoff_year: int = 2020,
        target_department: str = "300638",
    ):
        self.directory = directory
        self.cutoff_year = cutoff_year
        self.target_department = target_department

    def create_rules(self, rule_names: List[str]) -> List[CleaningRule]:
        rules = []
        for name in rule_names:
            name = name.lower()
            if name == "filename":
                rules.append(FilenameRule())
            elif name == "nocontent":
                rules.append(NoContentInsertedRule())
            elif name == "publication_tip":
                rules.append(PublicationTipRule())
            elif name == "404":
                rules.append(PageNotFoundRule())
            elif name == "empty_body":
                rules.append(EmptyBodyRule())
            elif name == "exact_publications":
                rules.append(ExactPublicationsBaseRule())
            elif name == "department_bandi":
                rules.append(DepartmentBandiRule(
                    target_department=self.target_department,
                ))
            elif name == "calendar":
                rules.append(CalendarRule())
            elif name == "news":
                rules.append(NewsRule())
            else:
                raise ValueError(f"Regola sconosciuta: {name}")

        return rules