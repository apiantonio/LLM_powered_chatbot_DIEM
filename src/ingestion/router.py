"""
ingestion/router.py — Routing aggiornato con docente_sezione.

Modifiche rispetto alla versione corrente:
  - extract_metadata() ora estrae docente_sezione dal path URL.
  - Nuova regola HTML per spostare strutture-didattiche in Collection 4.
  - Helper _classify_docente_sezione() per il mapping URL → sezione.
"""

import re
import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class CollectionTarget(str, Enum):
    """Enum delle collection — INVARIATO."""
    DOCENTI_DIDATTICA = "docenti_e_didattica"
    OFFERTA_FORMATIVA = "offerta_formativa_e_corsi"
    BANDI_AMMINISTRAZIONE = "bandi_e_amministrazione"
    DIPARTIMENTO_RICERCA = "dipartimento_e_ricerca"


# ============================================================
# NUOVO: Mapping sottosezione docente
# ============================================================

class DocenteSezione(str, Enum):
    """Sottosezioni logiche delle pagine docente."""
    PROFILO = "profilo"          # home + curriculum
    DIDATTICA = "didattica"      # corsi insegnati
    RICERCA = "ricerca"          # aree di ricerca, progetti
    INTERNATIONAL = "international"  # attività internazionali


def _classify_docente_sezione(source_url: str) -> Optional[str]:
    """
    Determina la sottosezione del docente dal path URL.
    
    Mapping:
      /home, /curriculum     → "profilo"
      /didattica             → "didattica"
      /ricerca               → "ricerca"
      /international         → "international"
      /risorse               → None (skip se vuota, altrimenti "didattica")
    """
    url_lower = source_url.lower()
    
    if "/home" in url_lower or "/curriculum" in url_lower:
        return DocenteSezione.PROFILO.value
    elif "/didattica" in url_lower:
        return DocenteSezione.DIDATTICA.value
    elif "/ricerca" in url_lower:
        return DocenteSezione.RICERCA.value
    elif "/international" in url_lower:
        return DocenteSezione.INTERNATIONAL.value
    elif "/risorse" in url_lower:
        # Le risorse vengono trattate come didattica se non vuote.
        # Il filtro per contenuto vuoto è nel crawler/transform.
        return DocenteSezione.DIDATTICA.value
    
    # Fallback per URL docente senza sottopagina specifica
    if "docenti.unisa.it/" in url_lower:
        return DocenteSezione.PROFILO.value
    
    return None


class DocumentRouter:
    """Router aggiornato con supporto docente_sezione e strutture fisiche."""

    _HTML_RULES = [
        # --- Docenti ---
        (r"docenti\.unisa\.it/", CollectionTarget.DOCENTI_DIDATTICA),
        
        # --- Corsi ---
        (r"corsi\.unisa\.it/", CollectionTarget.OFFERTA_FORMATIVA),
        
        # --- DIEM: regole specifiche ---
        (r"diem\.unisa\.it/.*bandi", CollectionTarget.BANDI_AMMINISTRAZIONE),
        (r"diem\.unisa\.it/didattica", CollectionTarget.OFFERTA_FORMATIVA),
        (r"diem\.unisa\.it/dipartimento/personale", CollectionTarget.DOCENTI_DIDATTICA),
        
        # MODIFICA: strutture-didattiche ora va in DIPARTIMENTO_RICERCA
        (r"diem\.unisa\.it/.*strutture-didattiche", CollectionTarget.DIPARTIMENTO_RICERCA),
        (r"diem\.unisa\.it/.*strutture.*laboratori", CollectionTarget.DIPARTIMENTO_RICERCA),
        
        (r"diem\.unisa\.it/ricerca", CollectionTarget.DIPARTIMENTO_RICERCA),
        (r"diem\.unisa\.it/terza-missione", CollectionTarget.DIPARTIMENTO_RICERCA),
        (r"diem\.unisa\.it/international", CollectionTarget.DIPARTIMENTO_RICERCA),
        (r"diem\.unisa\.it/dipartimento", CollectionTarget.DIPARTIMENTO_RICERCA),
        (r"diem\.unisa\.it/", CollectionTarget.DIPARTIMENTO_RICERCA),
    ]

    _PDF_RULES = [
        (r"__regolamenti-cds/", CollectionTarget.OFFERTA_FORMATIVA),
        (r"__piano-studi-cds/", CollectionTarget.OFFERTA_FORMATIVA),
        (r"__statistiche-corsi/", CollectionTarget.OFFERTA_FORMATIVA),
        (r"diem\.unisa\.it/uploads/rescue/\d+/\d+/", CollectionTarget.BANDI_AMMINISTRAZIONE),
        (r"corsi\.unisa\.it/", CollectionTarget.OFFERTA_FORMATIVA),
        (r"diem\.unisa\.it/", CollectionTarget.DIPARTIMENTO_RICERCA),
    ]

    @classmethod
    def route_html(cls, source_url: str) -> CollectionTarget:
        for pattern, target in cls._HTML_RULES:
            if re.search(pattern, source_url, re.IGNORECASE):
                return target
        return CollectionTarget.DIPARTIMENTO_RICERCA

    @classmethod
    def route_pdf(cls, pdf_url: str) -> CollectionTarget:
        for pattern, target in cls._PDF_RULES:
            if re.search(pattern, pdf_url, re.IGNORECASE):
                return target
        return CollectionTarget.BANDI_AMMINISTRAZIONE

    @classmethod
    def use_parent_child(cls, collection: CollectionTarget, pdf_url: str) -> bool:
        if collection != CollectionTarget.OFFERTA_FORMATIVA:
            return False
        return bool(re.search(r"(__regolamenti-cds|__piano-studi-cds)/", pdf_url))

    @classmethod
    def extract_metadata(cls, source_url: str, collection: CollectionTarget) -> dict:
        """
        Metadati arricchiti — AGGIORNATO con docente_sezione e doc_category
        granulare per strutture fisiche.
        """
        metadata = {
            "doc_category": collection.value,
            "source_domain": _extract_domain(source_url),
        }

        # --- Docenti: matricola + sezione ---
        matricola_match = re.search(r"docenti\.unisa\.it/(\d+)/", source_url)
        if matricola_match:
            metadata["docente_matricola"] = matricola_match.group(1)
            
            # NUOVO: classificazione della sottosezione
            sezione = _classify_docente_sezione(source_url)
            if sezione:
                metadata["docente_sezione"] = sezione

        # --- Corsi ---
        corso_match = re.search(r"corsi\.unisa\.it/([^/]+)", source_url)
        if corso_match:
            metadata["corso_slug"] = corso_match.group(1)

        # --- Anno ---
        anno_match = re.search(r"/(\d{4})/", source_url)
        if anno_match:
            metadata["anno"] = anno_match.group(1)
        
        # --- NUOVO: Strutture fisiche (doc_category granulare) ---
        url_lower = source_url.lower()
        if "strutture-didattiche" in url_lower or "aule" in url_lower:
            metadata["doc_category"] = "aula"
        elif "laboratori" in url_lower:
            metadata["doc_category"] = "laboratorio"

        return metadata


def _extract_domain(url: str) -> str:
    match = re.search(r"https?://([^/]+)", url)
    return match.group(1) if match else "unknown"