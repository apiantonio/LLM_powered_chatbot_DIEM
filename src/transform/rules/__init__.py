"""
Package delle regole di filtraggio e pulizia.

Espone le factory function per caricare le regole HTML e PDF
utilizzate dal crawler e dalla pipeline di ingestion.
"""

from typing import List
from transform.core.base_rule import CleaningRule, PdfFilterRule


def get_all_html_rules(**kwargs) -> List[CleaningRule]:
    """
    Costruisce e restituisce tutte le regole di pulizia HTML attive.

    Delega alla RuleFactory per la creazione delle istanze.
    Chiamata da ingestion_main.py quando il modulo transform è disponibile.
    """
    from pathlib import Path
    from transform.factory.cleaner_factory import RuleFactory

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
    """
    Costruisce e restituisce tutte le regole di filtro PDF attive.
    """
    from transform.factory.pdf_factory import PdfRuleFactory

    cutoff_year = kwargs.get("cutoff_year", 2020)
    factory = PdfRuleFactory(cutoff_year=cutoff_year)
    return factory.create_rules([
        "domain_whitelist", "semantic_trap", "obsolete_year", "english_pdf",
    ])