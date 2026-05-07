"""
Definizione dei tool esposti all'agente RAG.

Pattern: Agentic RAG — il retrieval e gli orari sono tool che l'agente
         invoca autonomamente nel ciclo ReAct, non step fissi di una chain.

L'agente decide:
- QUANDO chiamare search_knowledge_base (solo se serve contesto dal DIEM).
- QUANDO chiamare get_course_schedule (solo per domande su orari/esami).
- CON QUALI PARAMETRI (la query viene riformulata dall'agente stesso).
- QUANTE VOLTE (può fare follow-up retrieval se il contesto è insufficiente).

KPI Impact:
- Relevance: l'agente sceglie il tool giusto per la query.
- Correctness: EasyCourse fornisce dati real-time, non stale.
- Scope Awareness: se nessun tool è pertinente, l'agente lo dichiara.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain.tools import tool

from agent.tools.easycourse import get_easycourse_client

if TYPE_CHECKING:
    from retrieval.engine import RetrievalEngine

logger = logging.getLogger(__name__)

# Riferimento globale al retrieval engine, iniettato all'avvio
_retrieval_engine: "RetrievalEngine | None" = None


def set_retrieval_engine(engine: "RetrievalEngine") -> None:
    """Inietta il RetrievalEngine nei tool. Chiamato una volta all'avvio."""
    global _retrieval_engine
    _retrieval_engine = engine


@tool("search_knowledge_base")
def search_knowledge_base(query: str) -> str:
    """
    Cerca informazioni nella knowledge base del DIEM (Università di Salerno).
    
    Usa questo strumento per rispondere a domande su: corsi di laurea, docenti,
    regolamenti, procedure amministrative, tesi, borse di studio, laboratori,
    dottorato di ricerca e qualsiasi altra informazione del dipartimento DIEM.
    
    Args:
        query: La domanda o i termini di ricerca da cercare nella knowledge base.
    
    Returns:
        Il contesto testuale recuperato dai documenti del DIEM, con le fonti.
    """
    if _retrieval_engine is None:
        return "Errore interno: il motore di ricerca non è stato inizializzato."
    
    try:
        documents, used_query = _retrieval_engine.retrieve(query)
        
        if not documents:
            return (
                "Non ho trovato informazioni pertinenti nella knowledge base del DIEM "
                f"per la query: '{query}'. Prova a riformulare la domanda."
            )
        
        # Componi il contesto con citazione delle fonti
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source_url", "fonte non disponibile")
            doc_type = doc.metadata.get("doc_type", "sconosciuto")
            
            context_parts.append(
                f"[Documento {i} — {doc_type} — {source}]\n{doc.page_content}"
            )
        
        return "\n\n---\n\n".join(context_parts)
        
    except Exception as e:
        logger.error(f"Errore ricerca KB: {e}")
        return f"Errore durante la ricerca: {str(e)}"


@tool("get_course_schedule")
def get_course_schedule(course_or_professor: str) -> str:
    """
    Cerca gli orari delle lezioni e degli esami su EasyCourse UniSA.
    
    Usa questo strumento SOLO per domande specifiche su orari, calendario
    delle lezioni, o quando degli esami. Per altre informazioni sui corsi
    (programma, crediti, prerequisiti) usa search_knowledge_base.
    
    Args:
        course_or_professor: Nome del corso o cognome del docente.
    
    Returns:
        Gli orari trovati formattati come testo, oppure un messaggio di errore.
    """
    client = get_easycourse_client()
    
    # Euristica: se contiene un cognome (parola singola, capitalizzata), cerca per docente
    words = course_or_professor.strip().split()
    
    if len(words) == 1 and words[0][0].isupper():
        return client.search_schedule(professor_name=course_or_professor)
    else:
        return client.search_schedule(course_name=course_or_professor)


def get_all_tools() -> list:
    """Restituisce la lista di tutti i tool disponibili per l'agente."""
    return [search_knowledge_base, get_course_schedule]
