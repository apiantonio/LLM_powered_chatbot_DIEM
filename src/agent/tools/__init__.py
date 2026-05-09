"""
agent/tools/__init__.py — Tool multi-collection per l'agente RAG.

Ogni tool interroga una collection specifica, eliminando
l'inquinamento cross-dominio che causava i falsi positivi.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

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


def _search_collection(query: str, collection: CollectionTarget) -> str:
    """Helper condiviso per la ricerca in una singola collection."""
    if _retrieval_engine is None:
        return "Errore interno: motore di ricerca non inizializzato."
    try:
        documents, used_query = _retrieval_engine.retrieve(
            query, collection=collection.value
        )
        if not documents:
            return (
                f"Non ho trovato informazioni pertinenti per: '{query}'. "
                "Prova a riformulare la domanda."
            )
        return _format_results(documents)
    except Exception as e:
        logger.error(f"Errore ricerca {collection.value}: {e}")
        return f"Errore durante la ricerca: {str(e)}"


def _format_results(documents) -> str:
    """Formatta i documenti recuperati con citazione fonti e score."""
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
# TOOL 1: Docenti e Didattica
# ============================================================

@tool("search_docenti_didattica")
def search_docenti_didattica(query: str) -> str:
    """
    Cerca informazioni su docenti e personale del DIEM.

    Usa questo strumento per domande su: curriculum di un docente,
    qualifica accademica, contatti e email istituzionale, orario di
    ricevimento, corsi insegnati, aree di ricerca personali,
    pubblicazioni, ufficio e laboratorio.

    Esempi di query appropriate:
    - "Chi è Mario Vento?"
    - "Corsi insegnati dal prof. Capuano"
    - "Ricevimento del prof. Greco"
    - "Email del prof. Napoli"

    Args:
        query: Domanda o termini di ricerca sul docente.
    """
    return _search_collection(query, CollectionTarget.DOCENTI_DIDATTICA)


# ============================================================
# TOOL 2: Offerta Formativa e Corsi
# ============================================================

@tool("search_offerta_formativa")
def search_offerta_formativa(query: str) -> str:
    """
    Cerca informazioni su corsi di laurea, piani di studio e regolamenti.

    Usa questo strumento per domande su: corsi di laurea offerti dal DIEM,
    piani di studio, regolamenti didattici, requisiti di ammissione,
    crediti formativi, programmi degli insegnamenti, statistiche dei corsi,
    procedure di iscrizione, trasferimenti, OFA, tesi.

    Esempi di query appropriate:
    - "Corsi di laurea del DIEM"
    - "Piano di studi Ingegneria Informatica triennale"
    - "Requisiti ammissione magistrale"
    - "Regolamento didattico corso di laurea"
    - "Punteggio TOLC per iscriversi"

    Args:
        query: Domanda o termini di ricerca su corsi e offerta formativa.
    """
    return _search_collection(query, CollectionTarget.OFFERTA_FORMATIVA)


# ============================================================
# TOOL 3: Bandi e Amministrazione
# ============================================================

@tool("search_bandi_amministrazione")
def search_bandi_amministrazione(query: str) -> str:
    """
    Cerca bandi, borse di studio, assegni di ricerca e avvisi del DIEM.

    Usa questo strumento SOLO per domande specifiche su: bandi di concorso,
    borse di studio, assegni di ricerca, avvisi amministrativi, opportunità
    di finanziamento, contratti di collaborazione.

    NON usare questo strumento per cercare informazioni generali su un
    docente — anche se il docente è menzionato nel bando come responsabile.

    Esempi di query appropriate:
    - "Bandi borse di studio DIEM"
    - "Assegni di ricerca attivi"
    - "Bandi dottorato di ricerca"

    Args:
        query: Domanda o termini di ricerca su bandi e avvisi.
    """
    return _search_collection(query, CollectionTarget.BANDI_AMMINISTRAZIONE)


# ============================================================
# TOOL 4: Dipartimento e Ricerca
# ============================================================

@tool("search_dipartimento_ricerca")
def search_dipartimento_ricerca(query: str) -> str:
    """
    Cerca informazioni istituzionali sul DIEM: strutture, laboratori,
    aree di ricerca, progetti, sedi, internazionalizzazione.

    Usa questo strumento per domande su: dove si trova il DIEM,
    laboratori disponibili, attrezzature, aree di ricerca del dipartimento,
    progetti finanziati, terza missione, mobilità internazionale, Erasmus,
    commissioni, organi dipartimentali.

    Esempi di query appropriate:
    - "Dove si trova il DIEM?"
    - "Laboratori disponibili al DIEM"
    - "Aree di ricerca attive"
    - "Progetti finanziati del dipartimento"
    - "Opportunità Erasmus DIEM"

    Args:
        query: Domanda su strutture, ricerca o servizi dipartimentali.
    """
    return _search_collection(query, CollectionTarget.DIPARTIMENTO_RICERCA)


# ============================================================
# TOOL 5: Ricerca Trasversale (Fallback)
# ============================================================

@tool("search_all_collections")
def search_all_collections(query: str) -> str:
    """
    Ricerca trasversale su TUTTA la knowledge base del DIEM.

    Usa questo strumento SOLO quando:
    - La domanda è ambigua e potrebbe riguardare più domini
    - Gli altri strumenti specifici non hanno dato risultati sufficienti
    - La domanda copre più aspetti (es. "tutto su Mario Vento" include
      sia info personali che eventuali bandi)

    Per domande chiare, preferisci SEMPRE i tool specifici.

    Args:
        query: Domanda generica o cross-dominio.
    """
    if _retrieval_engine is None:
        return "Errore interno: motore di ricerca non inizializzato."
    try:
        documents, used_query = _retrieval_engine.retrieve_from_all(query)
        if not documents:
            return (
                f"Non ho trovato informazioni in nessuna collection per: '{query}'."
            )
        return _format_results(documents)
    except Exception as e:
        logger.error(f"Errore ricerca all: {e}")
        return f"Errore durante la ricerca: {str(e)}"


# ============================================================
# TOOL 6: EasyCourse (invariato)
# ============================================================

@tool("get_course_schedule")
def get_course_schedule(course_or_professor: str) -> str:
    """
    Cerca gli orari delle lezioni e degli esami su EasyCourse UniSA.

    Usa questo strumento SOLO per domande specifiche su orari, calendario
    delle lezioni, o quando degli esami. Per altre informazioni sui corsi
    (programma, crediti, prerequisiti) usa search_offerta_formativa.

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
        search_docenti_didattica,
        search_offerta_formativa,
        search_bandi_amministrazione,
        search_dipartimento_ricerca,
        search_all_collections,
        get_course_schedule,
    ]