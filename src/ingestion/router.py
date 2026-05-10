"""
ingestion/router.py — Routing aggiornato con docente_sezione e doc_category robusto.

FIX APPLICATI:
  - extract_metadata() ora usa regex robusti per classificare doc_category
    per strutture fisiche. Copre singolare/plurale e query parameters.
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
# Mapping sottosezione docente
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
        
        # strutture-didattiche e laboratori fisici → DIPARTIMENTO_RICERCA
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
        Metadati arricchiti — FIX con doc_category granulare robusto.

        FIX APPLICATO:
          Prima: il matching per doc_category cercava "laboratori" (plurale)
          ma gli URL reali contengono "laboratorio" (singolare nei query params).
          Inoltre il match per "aule" era troppo restrittivo.
          Ora: regex robusti che coprono singolare/plurale, path e query params.
          Il blocco di classificazione strutture fisiche viene eseguito SOLO
          per la collection dipartimento_e_ricerca.

        Mapping:
          URL con "strutture-didattiche", "aula", "aule"   → doc_category = "aula"
          URL con "laboratorio", "laboratori", "lab"        → doc_category = "laboratorio"
          URL con "sede", "sedi", "edificio", "campus"      → doc_category = "sede"
          Tutto il resto di dipartimento_e_ricerca          → doc_category = "dipartimento_e_ricerca" (default)
        """
        metadata = {
            "doc_category": collection.value,
            "source_domain": _extract_domain(source_url),
        }

        # --- Docenti: matricola + sezione ---
        matricola_match = re.search(r"docenti\.unisa\.it/(\d+)/", source_url)
        if matricola_match:
            metadata["docente_matricola"] = matricola_match.group(1)
            
            # Classificazione della sottosezione
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
        
        # --- Strutture fisiche: doc_category granulare (FIX) ---
        # Solo per la collection dipartimento_e_ricerca, dove coesistono
        # documenti su aule, laboratori, sedi E documenti su ricerca/progetti.
        # Per le altre collection il doc_category resta uguale al valore
        # della collection (che è già specifico).
        if collection == CollectionTarget.DIPARTIMENTO_RICERCA:
            url_lower = source_url.lower()
            
            # AULA: match su "strutture-didattiche", "aula", "aule"
            # (sia nel path che nei query parameters)
            # Regex: strutture-didattiche OPPURE aul + a/e (singolare/plurale)
            if re.search(r"strutture[-_]didattiche|aul[ae]", url_lower):
                metadata["doc_category"] = "aula"
            
            # LABORATORIO: match su "laboratorio", "laboratori", "lab"
            # Attenzione: "laboratorio" appare anche nei query params
            # (es. ?laboratorio=722). Includiamo anche queste pagine
            # perché descrivono laboratori fisici del dipartimento.
            # Regex: laborator + io/i (singolare/plurale) OPPURE "lab" come parola intera
            elif re.search(r"laborator[io]|laboratori|\blab\b", url_lower):
                metadata["doc_category"] = "laboratorio"
            
            # SEDE: match su "sede", "sedi", "edificio", "campus"
            # Regex: sed + e/i OPPURE edificio OPPURE campus
            elif re.search(r"\bsed[ei]\b|edifici[o]|campus", url_lower):
                metadata["doc_category"] = "sede"
            
            # Tutto il resto rimane "dipartimento_e_ricerca" (default già impostato)

        return metadata


def _extract_domain(url: str) -> str:
    match = re.search(r"https?://([^/]+)", url)
    return match.group(1) if match else "unknown"