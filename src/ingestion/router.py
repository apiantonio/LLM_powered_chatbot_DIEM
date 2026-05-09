"""
ingestion/router.py — Routing dei documenti nelle collection appropriate.
"""

import re
import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class CollectionTarget(str, Enum):
    """Enum delle collection Chroma target."""
    DOCENTI_DIDATTICA = "docenti_e_didattica"
    OFFERTA_FORMATIVA = "offerta_formativa_e_corsi"
    BANDI_AMMINISTRAZIONE = "bandi_e_amministrazione"
    DIPARTIMENTO_RICERCA = "dipartimento_e_ricerca"


class DocumentRouter:
    """
    Instrada ogni documento nella collection Chroma corretta
    basandosi sul source_url e sul path del file.
    
    Pattern: Strategy (GoF) — le regole di routing sono dati,
    la logica di matching è fissa.
    """

    # Regole HTML: pattern regex → collection
    _HTML_RULES = [
        # Dominio docenti → sempre Collection 1
        (r"docenti\.unisa\.it/", CollectionTarget.DOCENTI_DIDATTICA),
        
        # Dominio corsi → sempre Collection 2
        (r"corsi\.unisa\.it/", CollectionTarget.OFFERTA_FORMATIVA),
        
        # DIEM: bandi
        (r"diem\.unisa\.it/.*bandi", CollectionTarget.BANDI_AMMINISTRAZIONE),
        
        # DIEM: didattica
        (r"diem\.unisa\.it/didattica", CollectionTarget.OFFERTA_FORMATIVA),
        
        # DIEM: personale (pagina elenco docenti)
        (r"diem\.unisa\.it/dipartimento/personale", CollectionTarget.DOCENTI_DIDATTICA),
        
        # DIEM: ricerca, terza missione, international, strutture, home
        (r"diem\.unisa\.it/ricerca", CollectionTarget.DIPARTIMENTO_RICERCA),
        (r"diem\.unisa\.it/terza-missione", CollectionTarget.DIPARTIMENTO_RICERCA),
        (r"diem\.unisa\.it/international", CollectionTarget.DIPARTIMENTO_RICERCA),
        (r"diem\.unisa\.it/dipartimento", CollectionTarget.DIPARTIMENTO_RICERCA),
        (r"diem\.unisa\.it/progetti-finanziati", CollectionTarget.DIPARTIMENTO_RICERCA),
        
        # DIEM: fallback generico
        (r"diem\.unisa\.it/", CollectionTarget.DIPARTIMENTO_RICERCA),
    ]

    # Regole PDF: pattern sul URL del PDF → collection
    _PDF_RULES = [
        # Regolamenti CdS → Collection 2
        (r"__regolamenti-cds/", CollectionTarget.OFFERTA_FORMATIVA),
        # Piani di studio → Collection 2
        (r"__piano-studi-cds/", CollectionTarget.OFFERTA_FORMATIVA),
        # Statistiche corsi → Collection 2
        (r"__statistiche-corsi/", CollectionTarget.OFFERTA_FORMATIVA),
        # Bandi DIEM (path /uploads/rescue/292/) → Collection 3
        (r"diem\.unisa\.it/uploads/rescue/\d+/\d+/", CollectionTarget.BANDI_AMMINISTRAZIONE),
        # PDF da corsi.unisa.it non categorizzati → Collection 2
        (r"corsi\.unisa\.it/", CollectionTarget.OFFERTA_FORMATIVA),
        # Fallback PDF DIEM → Collection 4
        (r"diem\.unisa\.it/", CollectionTarget.DIPARTIMENTO_RICERCA),
    ]

    @classmethod
    def route_html(cls, source_url: str) -> CollectionTarget:
        """Determina la collection target per un documento HTML."""
        for pattern, target in cls._HTML_RULES:
            if re.search(pattern, source_url, re.IGNORECASE):
                logger.debug(f"HTML '{source_url[:80]}' → {target.value}")
                return target
        
        # Fallback assoluto
        logger.warning(f"HTML senza match di routing: {source_url}")
        return CollectionTarget.DIPARTIMENTO_RICERCA

    @classmethod
    def route_pdf(cls, pdf_url: str) -> CollectionTarget:
        """Determina la collection target per un documento PDF."""
        for pattern, target in cls._PDF_RULES:
            if re.search(pattern, pdf_url, re.IGNORECASE):
                logger.debug(f"PDF '{pdf_url[:80]}' → {target.value}")
                return target
        
        logger.warning(f"PDF senza match di routing: {pdf_url}")
        return CollectionTarget.BANDI_AMMINISTRAZIONE

    @classmethod
    def use_parent_child(cls, collection: CollectionTarget, pdf_url: str) -> bool:
        """
        Determina se un PDF deve usare il pattern Parent-Child.
        
        Solo i documenti lunghi e strutturati della collection
        offerta_formativa (regolamenti e piani di studio) lo necessitano.
        """
        if collection != CollectionTarget.OFFERTA_FORMATIVA:
            return False
        # Solo regolamenti e piani di studio
        return bool(re.search(
            r"(__regolamenti-cds|__piano-studi-cds)/", pdf_url
        ))

    @classmethod
    def extract_metadata(cls, source_url: str, collection: CollectionTarget) -> dict:
        """
        Estrae metadati arricchiti dal source_url per migliorare
        il retrieval e il filtraggio post-retrieval.
        """
        metadata = {
            "doc_category": collection.value,
            "source_domain": _extract_domain(source_url),
        }

        # Docenti: estrai matricola
        matricola_match = re.search(r"docenti\.unisa\.it/(\d+)/", source_url)
        if matricola_match:
            metadata["docente_matricola"] = matricola_match.group(1)

        # Corsi: estrai nome corso dal path
        corso_match = re.search(
            r"corsi\.unisa\.it/([^/]+)", source_url
        )
        if corso_match:
            metadata["corso_slug"] = corso_match.group(1)

        # Anno (regolamenti, piani di studio)
        anno_match = re.search(r"/(\d{4})/", source_url)
        if anno_match:
            metadata["anno"] = anno_match.group(1)

        return metadata


def _extract_domain(url: str) -> str:
    """Estrae il dominio dall'URL."""
    match = re.search(r"https?://([^/]+)", url)
    return match.group(1) if match else "unknown"