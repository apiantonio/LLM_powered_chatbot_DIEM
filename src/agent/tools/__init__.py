"""
agent/tools/__init__.py — Tool multi-collection con metadata filtering.

Modifiche:
  - search_docenti con parametro sezione opzionale.
  - Nuovo tool search_strutture_fisiche.
  - Description ottimizzate con anti-pattern.
"""

from __future__ import annotations
import logging
from typing import Optional, TYPE_CHECKING
from langchain.tools import tool
from agent.tools.easycourse import get_easycourse_client
from ingestion.router import CollectionTarget

if TYPE_CHECKING:
    from retrieval.engine import RetrievalEngine

logger = logging.getLogger(__name__)
_retrieval_engine: "RetrievalEngine | None" = None


def set_retrieval_engine(engine: "RetrievalEngine") -> None:
    global _retrieval_engine
    _retrieval_engine = engine

from typing import Dict

_VALID_SEZIONI = frozenset({"profilo", "didattica", "ricerca", "international"})

# Contatore errori per tool (anti-loop)
_tool_error_counts: Dict[str, int] = {}
_MAX_TOOL_RETRIES = 2

def _search_collection(
    query: str,
    collection: CollectionTarget,
    metadata_filter: Optional[dict] = None,
) -> str:
    """
    Helper condiviso — AGGIORNATO con anti-loop error handling.
    
    Se lo stesso tool fallisce 2+ volte con la stessa query, restituisce
    un messaggio che ISTRUISCE il modello a NON riprovare.
    """
    if _retrieval_engine is None:
        return "Errore interno: motore di ricerca non inizializzato."
    
    tool_key = f"{collection.value}:{query[:50]}"
    
    try:
        documents, used_query = _retrieval_engine.retrieve(
            query,
            collection=collection.value,
            metadata_filter=metadata_filter,
        )
        # Reset contatore su successo
        _tool_error_counts.pop(tool_key, None)
        
        if not documents:
            return (
                f"Non ho trovato informazioni pertinenti per: '{query}'. "
                "Prova a riformulare la domanda."
            )
        return _format_results(documents)
    
    except Exception as e:
        logger.error(f"Errore ricerca {collection.value}: {e}", exc_info=True)
        
        # Conteggio errori per prevenire loop
        _tool_error_counts[tool_key] = _tool_error_counts.get(tool_key, 0) + 1
        error_count = _tool_error_counts[tool_key]
        
        if error_count >= _MAX_TOOL_RETRIES:
            _tool_error_counts.pop(tool_key, None)
            return (
                "La ricerca non è disponibile al momento per un problema tecnico. "
                "NON riprovare con questo stesso tool. "
                "Rispondi all'utente che le informazioni non sono al momento "
                "reperibili e suggerisci di consultare il sito web del DIEM."
            )
        
        return (
            "La ricerca ha riscontrato un problema temporaneo. "
            "Prova a usare un tool di ricerca diverso (es. search_dipartimento "
            "o search_all) oppure riformula la domanda."
        )



def _format_results(documents) -> str:
    """Formatta i documenti — INVARIATO."""
    context_parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source_url", "fonte non disponibile")
        doc_type = doc.metadata.get("doc_type", "sconosciuto")
        category = doc.metadata.get("doc_category", "")
        score = doc.metadata.get("relevance_score", None)
        score_str = f" — score: {score:.4f}" if score is not None else ""
        context_parts.append(
            f"[Documento {i} — {doc_type} — {category} — "
            f"{source}{score_str}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(context_parts)


# ============================================================
# TOOL 1: Docenti (AGGIORNATO con sezione)
# ============================================================

@tool("search_docenti")
def search_docenti(query: str, sezione: Optional[str] = None) -> str:
    """Cerca informazioni su un DOCENTE specifico del DIEM.

    USA QUESTO TOOL per domande su PERSONE: profilo, curriculum, email,
    ricevimento, corsi insegnati da un docente, aree di ricerca personali.

    NON usare per: corsi di laurea (usa search_offerta_formativa),
    bandi (usa search_bandi), strutture fisiche (usa search_strutture_fisiche).

    Args:
        query: Domanda sul docente. Esempi: "Chi è Mario Vento?",
               "Corsi insegnati dal prof. Capuano", "Email prof. Greco".
        sezione: Filtro opzionale sulla sottosezione del docente.
                 Valori ammessi: "profilo", "didattica", "ricerca", "international".
                 Se omesso, cerca in tutte le sezioni.
                 Usa "profilo" per: chi è, email, ufficio, ricevimento, curriculum.
                 Usa "didattica" per: corsi insegnati, materiale didattico.
                 Usa "ricerca" per: aree di ricerca, progetti, pubblicazioni.
    """
    # Validazione input
    if sezione and sezione.lower().strip() not in _VALID_SEZIONI:
        logger.warning(
            f"Parametro sezione non valido: '{sezione}'. "
            f"Valori ammessi: {_VALID_SEZIONI}. Ignoro il filtro."
        )
        sezione = None
    elif sezione:
        sezione = sezione.lower().strip()
    
    metadata_filter = None
    if sezione:
        metadata_filter = {"docente_sezione": sezione}
    
    return _search_collection(
        query, CollectionTarget.DOCENTI_DIDATTICA, metadata_filter
    )


# ============================================================
# TOOL 2-3: Offerta Formativa e Bandi — INVARIATI
# ============================================================

@tool("search_offerta_formativa")
def search_offerta_formativa(query: str) -> str:
    """Cerca informazioni su corsi di laurea, piani di studio e regolamenti.

    USA QUESTO TOOL per: corsi di laurea del DIEM, piani di studio,
    regolamenti didattici, requisiti di ammissione, crediti, programmi,
    OFA, tesi, statistiche corsi.

    NON usare per: informazioni su un docente specifico (usa search_docenti),
    orari lezioni (usa get_course_schedule).

    Args:
        query: Domanda su corsi e offerta formativa.
    """
    return _search_collection(query, CollectionTarget.OFFERTA_FORMATIVA)


@tool("search_bandi")
def search_bandi(query: str) -> str:
    """Cerca bandi, borse di studio, assegni di ricerca e avvisi del DIEM.

    USA QUESTO TOOL SOLO per: bandi di concorso, borse di studio,
    assegni di ricerca, avvisi amministrativi, dottorato.

    NON usare per: informazioni generali su un docente (usa search_docenti),
    anche se il docente è menzionato come responsabile in un bando.

    Args:
        query: Domanda su bandi e avvisi.
    """
    return _search_collection(query, CollectionTarget.BANDI_AMMINISTRAZIONE)


# ============================================================
# TOOL 4: Dipartimento e Ricerca — INVARIATO
# ============================================================

@tool("search_dipartimento")
def search_dipartimento(query: str) -> str:
    """Cerca informazioni istituzionali sul DIEM: ricerca, progetti,
    internazionalizzazione, terza missione, organi dipartimentali.

    USA QUESTO TOOL per: aree di ricerca del dipartimento, progetti
    finanziati, Erasmus, commissioni, organi. Per aule e laboratori
    usa search_strutture_fisiche.

    Args:
        query: Domanda su ricerca o servizi dipartimentali.
    """
    return _search_collection(query, CollectionTarget.DIPARTIMENTO_RICERCA)


# ============================================================
# TOOL 5: Strutture Fisiche (NUOVO)
# ============================================================

@tool("search_strutture_fisiche")
def search_strutture_fisiche(query: str) -> str:
    """Cerca informazioni su AULE, LABORATORI e SEDI del DIEM.

    USA QUESTO TOOL per: dove si trova un'aula, quali laboratori sono
    disponibili, attrezzature, sedi del dipartimento, mappa campus.

    Esempi: "Dove si trova l'aula F8?", "Laboratori DIEM",
    "Sede del dipartimento", "Attrezzature laboratorio X".

    Args:
        query: Domanda su strutture fisiche del DIEM.
    """
    metadata_filter = {
        "doc_category": ["aula", "laboratorio", "sede"]
    }
    return _search_collection(
        query, CollectionTarget.DIPARTIMENTO_RICERCA, metadata_filter
    )


# ============================================================
# TOOL 6: Ricerca Trasversale — INVARIATO
# ============================================================

@tool("search_all")
def search_all(query: str) -> str:
    """Ricerca trasversale su TUTTA la knowledge base del DIEM.

    Usa SOLO quando: la domanda è ambigua, copre più aree, o gli
    altri tool non hanno dato risultati. Per domande chiare,
    preferisci SEMPRE i tool specifici.

    Args:
        query: Domanda generica o cross-dominio.
    """
    if _retrieval_engine is None:
        return "Errore interno: motore di ricerca non inizializzato."
    try:
        documents, used_query = _retrieval_engine.retrieve_from_all(query)
        if not documents:
            return f"Non ho trovato informazioni per: '{query}'."
        return _format_results(documents)
    except Exception as e:
        logger.error(f"Errore ricerca all: {e}")
        return f"Errore durante la ricerca: {str(e)}"


# ============================================================
# TOOL 7: EasyCourse — INVARIATO
# ============================================================

@tool("get_course_schedule")
def get_course_schedule(course_or_professor: str) -> str:
    """Cerca orari delle lezioni e degli esami su EasyCourse UniSA.

    Usa SOLO per orari e calendario lezioni. Per programmi, crediti
    o prerequisiti usa search_offerta_formativa.

    Args:
        course_or_professor: Nome del corso o cognome del docente.
    """
    client = get_easycourse_client()
    words = course_or_professor.strip().split()
    if len(words) == 1 and words[0][0].isupper():
        return client.search_schedule(professor_name=course_or_professor)
    else:
        return client.search_schedule(course_name=course_or_professor)


def get_all_tools() -> list:
    """Restituisce tutti i tool disponibili per l'agente."""
    return [
        search_docenti,
        search_offerta_formativa,
        search_bandi,
        search_dipartimento,
        search_strutture_fisiche,
        search_all,
        get_course_schedule,
    ]