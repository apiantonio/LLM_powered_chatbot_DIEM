"""
ingestion/router.py — Routing dei documenti nelle collection appropriate.

Responsabilità:
  1. Determinare in quale collection Chroma un documento (HTML o PDF) va indicizzato.
  2. Determinare se un PDF deve usare il pattern Parent-Child.
  3. Estrarre metadati arricchiti dal source_url.

Coerenza garantita con:
  - config/settings.py: i nomi delle collection (CollectionTarget.value) corrispondono
    ai parametri di chunking in IngestionConfig.get_collection_html_params().
  - ingestion/indexer.py: usa CollectionTarget come chiave per self._collections dict.
  - ingestion/registry.py: IndexEntry.collection_name usa CollectionTarget.value.
  - agent/tools/__init__.py: ogni tool passa CollectionTarget al RetrievalEngine.
  - retrieval/engine.py: usa CollectionTarget.value come chiave nei retriever cache.
"""

import re
import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class CollectionTarget(str, Enum):
    """
    Enum delle collection Chroma target.
    
    I .value DEVONO corrispondere esattamente a:
    - Le chiavi in IngestionConfig.get_collection_html_params()
    - Le collection_name usate in Chroma(collection_name=...)
    - I valori passati a RetrievalEngine.retrieve(collection=...)
    """
    DOCENTI_DIDATTICA = "docenti_e_didattica"
    OFFERTA_FORMATIVA = "offerta_formativa_e_corsi"
    BANDI_AMMINISTRAZIONE = "bandi_e_amministrazione"
    DIPARTIMENTO_RICERCA = "dipartimento_e_ricerca"


class DocumentRouter:
    """
    Instrada ogni documento nella collection Chroma corretta
    basandosi sul source_url e sul path del file.
    
    Pattern: Strategy (GoF) — le regole di routing sono dati,
    la logica di matching è fissa. L'ordine delle regole è significativo:
    regole più specifiche prima, fallback generici alla fine.
    """

    # Regole HTML: pattern regex → collection
    # ORDINE SIGNIFICATIVO: le regole più specifiche (es. diem.unisa.it/didattica)
    # devono precedere i fallback generici (es. diem.unisa.it/)
    _HTML_RULES = [
        # --- Dominio docenti → SEMPRE Collection 1 ---
        (r"docenti\.unisa\.it/", CollectionTarget.DOCENTI_DIDATTICA),
        
        # --- Dominio corsi → SEMPRE Collection 2 ---
        (r"corsi\.unisa\.it/", CollectionTarget.OFFERTA_FORMATIVA),
        
        # --- DIEM: regole specifiche PRIMA del fallback ---
        # Bandi (home-bandi-anno, pagine bandi generiche)
        (r"diem\.unisa\.it/.*bandi", CollectionTarget.BANDI_AMMINISTRAZIONE),
        
        # Didattica DIEM → Collection 2 (coerente con corsi.unisa.it)
        (r"diem\.unisa\.it/didattica", CollectionTarget.OFFERTA_FORMATIVA),
        
        # Personale DIEM → Collection 1 (elenco docenti del dipartimento)
        (r"diem\.unisa\.it/dipartimento/personale", CollectionTarget.DOCENTI_DIDATTICA),
        
        # Ricerca, terza missione, international, strutture → Collection 4
        (r"diem\.unisa\.it/ricerca", CollectionTarget.DIPARTIMENTO_RICERCA),
        (r"diem\.unisa\.it/terza-missione", CollectionTarget.DIPARTIMENTO_RICERCA),
        (r"diem\.unisa\.it/international", CollectionTarget.DIPARTIMENTO_RICERCA),
        (r"diem\.unisa\.it/dipartimento", CollectionTarget.DIPARTIMENTO_RICERCA),
        (r"diem\.unisa\.it/progetti-finanziati", CollectionTarget.DIPARTIMENTO_RICERCA),
        
        # --- DIEM: fallback generico (home, contatti, sedi, ecc.) ---
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
        # Bandi DIEM (path /uploads/rescue/{id_numerico}/{id_numerico}/) → Collection 3
        (r"diem\.unisa\.it/uploads/rescue/\d+/\d+/", CollectionTarget.BANDI_AMMINISTRAZIONE),
        # PDF generici da corsi.unisa.it → Collection 2
        (r"corsi\.unisa\.it/", CollectionTarget.OFFERTA_FORMATIVA),
        # Fallback: PDF da diem.unisa.it non categorizzati → Collection 4
        (r"diem\.unisa\.it/", CollectionTarget.DIPARTIMENTO_RICERCA),
    ]

    @classmethod
    def route_html(cls, source_url: str) -> CollectionTarget:
        """Determina la collection target per un documento HTML."""
        for pattern, target in cls._HTML_RULES:
            if re.search(pattern, source_url, re.IGNORECASE):
                logger.debug(f"HTML routing: '{source_url[:80]}' → {target.value}")
                return target
        
        logger.warning(
            f"HTML senza match di routing, fallback a DIPARTIMENTO_RICERCA: "
            f"{source_url}"
        )
        return CollectionTarget.DIPARTIMENTO_RICERCA

    @classmethod
    def route_pdf(cls, pdf_url: str) -> CollectionTarget:
        """Determina la collection target per un documento PDF."""
        for pattern, target in cls._PDF_RULES:
            if re.search(pattern, pdf_url, re.IGNORECASE):
                logger.debug(f"PDF routing: '{pdf_url[:80]}' → {target.value}")
                return target
        
        # Fallback: i PDF senza match vanno nei bandi (la maggior parte
        # dei PDF non categorizzati dal DIEM sono bandi/avvisi)
        logger.warning(
            f"PDF senza match di routing, fallback a BANDI_AMMINISTRAZIONE: "
            f"{pdf_url}"
        )
        return CollectionTarget.BANDI_AMMINISTRAZIONE

    @classmethod
    def use_parent_child(cls, collection: CollectionTarget, pdf_url: str) -> bool:
        """
        Determina se un PDF deve usare il pattern Parent-Child.
        
        Solo regolamenti e piani di studio nella collection OFFERTA_FORMATIVA.
        Questo è il punto critico che risolve il problema dei falsi positivi:
        i bandi NON usano mai Parent-Child.
        """
        if collection != CollectionTarget.OFFERTA_FORMATIVA:
            return False
        return bool(re.search(
            r"(__regolamenti-cds|__piano-studi-cds)/", pdf_url
        ))

    @classmethod
    def extract_metadata(cls, source_url: str, collection: CollectionTarget) -> dict:
        """Estrae metadati arricchiti dal source_url."""
        metadata = {
            "doc_category": collection.value,
            "source_domain": _extract_domain(source_url),
        }

        # Docenti: estrai la matricola dal path URL
        matricola_match = re.search(r"docenti\.unisa\.it/(\d+)/", source_url)
        if matricola_match:
            metadata["docente_matricola"] = matricola_match.group(1)

        # Corsi: estrai lo slug del corso dal path URL
        corso_match = re.search(r"corsi\.unisa\.it/([^/]+)", source_url)
        if corso_match:
            metadata["corso_slug"] = corso_match.group(1)

        # Anno accademico (presente in regolamenti, piani di studio, statistiche)
        anno_match = re.search(r"/(\d{4})/", source_url)
        if anno_match:
            metadata["anno"] = anno_match.group(1)

        return metadata


def _extract_domain(url: str) -> str:
    """Estrae il dominio dall'URL."""
    match = re.search(r"https?://([^/]+)", url)
    return match.group(1) if match else "unknown"