"""
agent/tools/__init__.py — Tool multi-collection con multiquery integrata.

REFACTORING APPLICATO:
  1. _search_collection() adattato al nuovo retrieve() che restituisce
     una tupla a 3 elementi: (documents, rewritten_query, multi_queries).
  2. I dati di multiquery e rewritten_query vengono esposti tramite
     variabili globali _last_search_meta per il sistema di logging.
  3. set_chat_history() invariato per il query rewriting contestuale.
"""

from __future__ import annotations
import logging
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from langchain.tools import tool
from agent.tools.easycourse import get_easycourse_client
from ingestion.router import CollectionTarget

if TYPE_CHECKING:
    from retrieval.engine import RetrievalEngine

logger = logging.getLogger(__name__)
_retrieval_engine: "RetrievalEngine | None" = None

# Chat history globale per il query rewriting contestuale
_chat_history: list = []

# ── NUOVO: Metadati dell'ultima ricerca per il sistema di logging ──
# Aggiornato ad ogni chiamata _search_collection() per essere letto
# dal callback di logging.
_last_search_meta: Dict[str, Any] = {
    "rewritten_query": "",
    "multi_queries": [],
    "tool_name": "",
    "collection": "",
    "metadata_filter": None,
    "top_links": [],
}


def set_retrieval_engine(engine: "RetrievalEngine") -> None:
    """Inietta il RetrievalEngine nei tool (chiamata dalla Factory)."""
    global _retrieval_engine
    _retrieval_engine = engine


def set_chat_history(history: list) -> None:
    """Inietta la chat history corrente nei tool per il query rewriting."""
    global _chat_history
    _chat_history = history


def get_last_search_meta() -> Dict[str, Any]:
    """Restituisce i metadati dell'ultima ricerca (per il logging)."""
    return _last_search_meta.copy()


_VALID_SEZIONI = frozenset({"profilo", "didattica", "ricerca", "international"})

# Contatore errori per tool (anti-loop)
_tool_error_counts: Dict[str, int] = {}
_MAX_TOOL_RETRIES = 2


def _search_collection(
    query: str,
    collection: CollectionTarget,
    tool_name: str,
    metadata_filter: Optional[dict] = None,
) -> str:
    """
    Helper condiviso — AGGIORNATO per multiquery.

    retrieve() ora restituisce (documents, rewritten_query, multi_queries).
    I metadati vengono salvati in _last_search_meta per il logging.
    """
    global _last_search_meta

    if _retrieval_engine is None:
        return "Errore interno: motore di ricerca non inizializzato."

    tool_key = f"{collection.value}:{query[:50]}"

    try:
        result = _retrieval_engine.retrieve(
            query,
            collection=collection.value,
            metadata_filter=metadata_filter,
            chat_history=_chat_history,
        )

        # retrieve() restituisce 3 valori con multiquery, 2 senza
        if len(result) == 3:
            documents, rewritten_query, multi_queries = result
        else:
            documents, rewritten_query = result
            multi_queries = [rewritten_query]

        # Aggiorna i metadati per il logging
        top_links = []
        for doc in documents[:5]:
            link = doc.metadata.get("source_url", "N/D")
            if link not in top_links:
                top_links.append(link)

        _last_search_meta = {
            "rewritten_query": rewritten_query,
            "multi_queries": multi_queries,
            "tool_name": tool_name,
            "collection": collection.value,
            "metadata_filter": metadata_filter,
            "top_links": top_links[:5],
        }

        # Reset contatore errori su successo
        _tool_error_counts.pop(tool_key, None)

        if not documents:
            return (
                f"Non ho trovato informazioni pertinenti per: '{query}'. "
                "Prova a riformulare la domanda."
            )
        return _format_results(documents)

    except Exception as e:
        logger.error(f"Errore ricerca {collection.value}: {e}", exc_info=True)

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
    """Formatta i documenti recuperati per l'output del tool."""
    context_parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source_url", "fonte non disponibile")
        doc_type = doc.metadata.get("doc_type", "sconosciuto")
        category = doc.metadata.get("doc_category", "")
        score = doc.metadata.get("relevance_score", None)
        score_str = f" — score: {score:.4f}" if score is not None else ""

        standard_keys = {
            "source_url", "doc_type", "doc_category",
            "relevance_score", "start_index",
        }
        extra_meta = {
            k: v for k, v in doc.metadata.items()
            if k not in standard_keys and v is not None and v != ""
        }
        extra_meta_str = ""
        if extra_meta:
            pairs = " | ".join(f"{k}={v}" for k, v in extra_meta.items())
            extra_meta_str = f"\n[meta: {pairs}]"

        context_parts.append(
            f"[Documento {i} — {doc_type} — {category} — "
            f"{source}{score_str}]{extra_meta_str}\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(context_parts)


# ============================================================
# TOOL 1: Docenti (con sezione)
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
    """
    if sezione and sezione.lower().strip() not in _VALID_SEZIONI:
        logger.warning(
            f"Parametro sezione non valido: '{sezione}'. Ignoro il filtro."
        )
        sezione = None
    elif sezione:
        sezione = sezione.lower().strip()

    metadata_filter = None
    if sezione:
        metadata_filter = {"docente_sezione": sezione}

    return _search_collection(
        query, CollectionTarget.DOCENTI_DIDATTICA,
        "search_docenti", metadata_filter
    )


# ============================================================
# TOOL 2-3: Offerta Formativa e Bandi
# ============================================================

@tool("search_offerta_formativa")
def search_offerta_formativa(query: str) -> str:
    """Cerca informazioni su corsi di laurea, piani di studio e regolamenti.

    USA QUESTO TOOL per: corsi di laurea del DIEM, piani di studio,
    regolamenti didattici, requisiti di ammissione, crediti, programmi,
    OFA, tesi, statistiche corsi.

    Args:
        query: Domanda su corsi e offerta formativa.
    """
    return _search_collection(
        query, CollectionTarget.OFFERTA_FORMATIVA,
        "search_offerta_formativa"
    )


@tool("search_bandi")
def search_bandi(query: str) -> str:
    """Cerca bandi, borse di studio, assegni di ricerca e avvisi del DIEM.

    USA QUESTO TOOL SOLO per: bandi di concorso, borse di studio,
    assegni di ricerca, avvisi amministrativi, dottorato.

    Args:
        query: Domanda su bandi e avvisi.
    """
    return _search_collection(
        query, CollectionTarget.BANDI_AMMINISTRAZIONE,
        "search_bandi"
    )


# ============================================================
# TOOL 4: Dipartimento e Ricerca
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
    return _search_collection(
        query, CollectionTarget.DIPARTIMENTO_RICERCA,
        "search_dipartimento"
    )


# ============================================================
# TOOL 5: Strutture Fisiche
# ============================================================

@tool("search_strutture_fisiche")
def search_strutture_fisiche(query: str, tipo_struttura: Optional[str] = None) -> str:
    """Cerca informazioni su AULE, LABORATORI e SEDI del DIEM.

    USA QUESTO TOOL per: dove si trova un'aula, quali laboratori sono
    disponibili, attrezzature, sedi del dipartimento, mappa campus.

    Args:
        query: Domanda su strutture fisiche del DIEM.
        tipo_struttura: Filtro opzionale. Valori ammessi: "aula", "laboratorio", "sede".
    """
    if tipo_struttura is None:
        query_lower = query.lower()
        if "aula" in query_lower:
            tipo_struttura = "aula"
        elif any(kw in query_lower for kw in ["laboratorio", "lab "]):
            tipo_struttura = "laboratorio"
        elif any(kw in query_lower for kw in ["sede", "edificio", "campus"]):
            tipo_struttura = "sede"

    if tipo_struttura and tipo_struttura in ("aula", "laboratorio", "sede"):
        metadata_filter = {"doc_category": tipo_struttura}
    else:
        metadata_filter = {
            "doc_category": ["aula", "laboratorio", "sede"]
        }

    return _search_collection(
        query, CollectionTarget.DIPARTIMENTO_RICERCA,
        "search_strutture_fisiche", metadata_filter
    )


# ============================================================
# TOOL 6: Ricerca Trasversale
# ============================================================

@tool("search_all")
def search_all(query: str) -> str:
    """Ricerca trasversale su TUTTA la knowledge base del DIEM.

    Usa SOLO quando: la domanda è ambigua, copre più aree, o gli
    altri tool non hanno dato risultati.

    Args:
        query: Domanda generica o cross-dominio.
    """
    global _last_search_meta

    if _retrieval_engine is None:
        return "Errore interno: motore di ricerca non inizializzato."
    try:
        documents, used_query = _retrieval_engine.retrieve_from_all(query)

        top_links = []
        for doc in documents[:5]:
            link = doc.metadata.get("source_url", "N/D")
            if link not in top_links:
                top_links.append(link)

        _last_search_meta = {
            "rewritten_query": used_query,
            "multi_queries": [used_query],
            "tool_name": "search_all",
            "collection": "ALL (cross-collection)",
            "metadata_filter": None,
            "top_links": top_links[:5],
        }

        if not documents:
            return f"Non ho trovato informazioni per: '{query}'."
        return _format_results(documents)
    except Exception as e:
        logger.error(f"Errore ricerca all: {e}")
        return f"Errore durante la ricerca: {str(e)}"


# ============================================================
# TOOL 7: EasyCourse
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
