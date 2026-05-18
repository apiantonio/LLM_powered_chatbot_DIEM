from typing import List
from scraping.core.base_rule import CleaningRule, PdfFilterRule
from scraping.factory.pdf_factory import PdfRuleFactory
from pathlib import Path
from scraping.factory.cleaner_factory import RuleFactory

def get_all_html_rules(**kwargs) -> List[CleaningRule]:

    directory = Path(kwargs.get("directory", "data/raw/html_samples"))
    cutoff_year = kwargs.get("cutoff_year", 2020)
    target_department = kwargs.get("target_department", "300638")

    factory = RuleFactory(
        directory=directory,
        cutoff_year=cutoff_year,
        target_department=target_department,
    )
    return factory.create_rules([
        "publication_tip", "exact_publications",
        "department_bandi", "calendar", "news",
        "404", "nocontent", "empty_body", "filename",
    ])


def get_all_pdf_rules(**kwargs) -> List[PdfFilterRule]:
    
    cutoff_year = kwargs.get("cutoff_year", 2020)
    factory = PdfRuleFactory(cutoff_year=cutoff_year)
    return factory.create_rules([
        "domain_whitelist", "semantic_trap", "obsolete_year", "english_pdf",
    ])