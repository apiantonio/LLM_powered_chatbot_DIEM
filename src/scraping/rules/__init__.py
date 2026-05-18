"""Funzioni di convenienza per ottenere l'insieme completo delle regole."""

from pathlib import Path
from typing import List
from scraping.core.base_rule import CleaningRule, PdfFilterRule

def get_all_html_rules(**kwargs) -> List[CleaningRule]:
    """Crea e restituisce tutte le regole di pulizia HTML con parametri configurabili.

    Args:
        **kwargs: Parametri opzionali (directory, cutoff_year, target_department).

    Returns:
        Lista completa delle CleaningRule.
    """
    from scraping.factory.cleaner_factory import RuleFactory

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
    from scraping.factory.pdf_factory import PdfRuleFactory

    cutoff_year = kwargs.get("cutoff_year", 2020)
    factory = PdfRuleFactory(cutoff_year=cutoff_year)
    return factory.create_rules([
        "domain_whitelist",
        "semantic_trap",
        "obsolete_year",
        "english_pdf",
    ])
