"""Tool di ricerca per l'agente RAG DIEM.

Definisce i tool LangChain per la ricerca nelle collezioni del knowledge base:
persone, offerta formativa, dipartimento e ricerca trasversale.
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any, Literal, TYPE_CHECKING

from pydantic import BaseModel, Field
from langchain.tools import tool

from ingestion.router import CollectionTarget

if TYPE_CHECKING:
    from retrieval.engine import RetrievalEngine

logger = logging.getLogger(__name__)

_retrieval_engine: "RetrievalEngine | None" = None
_chat_history: list = []

_last_search_meta: Dict[str, Any] = {
    "rewritten_query": "",
    "multi_queries": [],
    "tool_name": "",
    "collection": "",
    "metadata_filter": None,
    "top_links": [],
}


def set_retrieval_engine(engine: "RetrievalEngine") -> None:
    """Inietta il RetrievalEngine nei tool."""
    global _retrieval_engine
    _retrieval_engine = engine


def set_chat_history(history: list) -> None:
    """Inietta la chat history corrente nei tool.

    Mantenuto per retrocompatibilita. La chat_history non viene
    passata a engine.retrieve() poiche il Query Rewriting e' stato
    spostato in agent.py.
    """
    global _chat_history
    _chat_history = history


def get_last_search_meta() -> Dict[str, Any]:
    """Restituisce i metadati dell'ultima ricerca effettuata."""
    return _last_search_meta.copy()


_VALID_SOTTO_AREA_PERSONE = frozenset({
    "profilo", "didattica", "ricerca", "internazionale", "risorse"
})

_VALID_SOTTO_AREA_OFFERTA_FORMATIVA = frozenset({
    "informazioni_corso", "didattica", "aule", "terza_missione",
    "statistiche", "regolamenti", "piani_di_studio", "documentazione_corso",
})

_VALID_SOTTO_AREA_DIPARTIMENTO = frozenset({
    "aule", "laboratori", "bandi", "ricerca_dipartimentale",
    "terza_missione", "internazionale", "organizzazione",
    "alternanza", "eccellenza", "monitoraggio", "personale", "generale",
})

_tool_error_counts: Dict[str, int] = {}
_MAX_TOOL_RETRIES = 2

_DEFINITIVE_NOT_FOUND = (
    "La ricerca su TUTTE le collection (persone, offerta formativa, "
    "dipartimento) non ha prodotto risultati per: '{query}'. "
    "L'informazione richiesta non e' presente nella knowledge base del DIEM. "
    "NON invocare altri tool di ricerca: rispondi all'utente che "
    "l'informazione non e' disponibile e suggerisci di consultare "
    "direttamente il sito web del DIEM o contattare la segreteria."
)


def _fallback_search_all(query: str, original_tool_name: str) -> Optional[str]:
    """Esegue una ricerca trasversale come fallback quando il tool primario non trova risultati.

    Args:
        query: Query di ricerca.
        original_tool_name: Nome del tool che ha originato il fallback.

    Returns:
        Risultati formattati, oppure None se nessun documento trovato.
    """
    global _last_search_meta

    if _retrieval_engine is None:
        return None

    logger.info(
        "FALLBACK INTERNO search_all attivato da %s per query: '%s'",
        original_tool_name,
        query[:80],
    )

    try:
        documents, used_query = _retrieval_engine.retrieve_from_all(query)

        if not documents:
            logger.info(
                "Anche il fallback search_all non ha trovato risultati per: '%s'",
                query[:80],
            )
            return None

        top_links = _extract_top_links(documents)

        _last_search_meta = {
            "rewritten_query": used_query,
            "multi_queries": [used_query],
            "tool_name": f"{original_tool_name} -> fallback:search_all",
            "collection": "ALL (cross-collection, fallback)",
            "metadata_filter": None,
            "top_links": top_links,
        }

        logger.info(
            "Fallback search_all ha trovato %d documenti per: '%s'",
            len(documents),
            query[:80],
        )

        return _format_results(documents)

    except Exception as e:
        logger.error("Errore nel fallback search_all: %s", e, exc_info=True)
        return None


def _search_collection(
    query: str,
    collection: CollectionTarget,
    tool_name: str,
    metadata_filter: Optional[dict] = None,
) -> str:
    """Esegue una ricerca su una collezione specifica con fallback automatico.

    Args:
        query: Query di ricerca.
        collection: Collezione target.
        tool_name: Nome del tool invocante.
        metadata_filter: Filtri opzionali sui metadati.

    Returns:
        Risultati formattati come stringa.
    """
    global _last_search_meta

    if _retrieval_engine is None:
        return "Errore interno: motore di ricerca non inizializzato."

    tool_key = f"{collection.value}:{query[:50]}"

    logger.debug("Ricerca in collezione %s: %s", collection.value, query)

    try:
        result = _retrieval_engine.retrieve(
            query,
            collection=collection.value,
            metadata_filter=metadata_filter,
        )

        if len(result) == 3:
            documents, rewritten_query, multi_queries = result
        else:
            documents, rewritten_query = result
            multi_queries = [rewritten_query]

        top_links = _extract_top_links(documents)

        _last_search_meta = {
            "rewritten_query": rewritten_query,
            "multi_queries": multi_queries,
            "tool_name": tool_name,
            "collection": collection.value,
            "metadata_filter": metadata_filter,
            "top_links": top_links,
        }

        _tool_error_counts.pop(tool_key, None)

        if not documents:
            logger.info(
                "0 risultati da %s (collection=%s). Attivo fallback interno su search_all.",
                tool_name,
                collection.value,
            )

            fallback_result = _fallback_search_all(query, tool_name)

            if fallback_result:
                return fallback_result

            return _DEFINITIVE_NOT_FOUND.format(query=query)

        return _format_results(documents)

    except Exception as e:
        logger.error("Errore ricerca %s: %s", collection.value, e, exc_info=True)

        _tool_error_counts[tool_key] = _tool_error_counts.get(tool_key, 0) + 1
        error_count = _tool_error_counts[tool_key]

        if error_count >= _MAX_TOOL_RETRIES:
            _tool_error_counts.pop(tool_key, None)
            return (
                "La ricerca non e' disponibile al momento. "
                "NON riprovare con questo stesso tool. "
                "Rispondi all'utente che le informazioni non sono al momento "
                "reperibili e suggerisci di consultare il sito web del DIEM."
            )

        return (
            "Problema temporaneo. Prova un tool diverso "
            "(es. search_dipartimento o search_all)."
        )


def _extract_top_links(documents, max_links: int = 5) -> List[str]:
    """Estrae i link principali dai documenti recuperati.

    Args:
        documents: Lista di documenti con metadati.
        max_links: Numero massimo di link da estrarre.

    Returns:
        Lista di URL univoci.
    """
    top_links = []
    for doc in documents[:max_links]:
        link = doc.metadata.get(
            "url_originale",
            doc.metadata.get("source_url", "N/D"),
        )
        if link not in top_links:
            top_links.append(link)
    return top_links[:max_links]


def _format_results(documents) -> str:
    """Formatta i documenti recuperati in una stringa leggibile per il LLM.

    Args:
        documents: Lista di documenti con metadati e contenuto.

    Returns:
        Stringa formattata con i documenti separati da delimitatori.
    """
    standard_keys = {
        "source_url", "url_originale", "doc_type", "formato_sorgente",
        "doc_category", "sotto_area", "relevance_score", "start_index",
        "source_file", "source_domain",
    }

    context_parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get(
            "url_originale",
            doc.metadata.get("source_url", "fonte non disponibile"),
        )
        formato = doc.metadata.get(
            "formato_sorgente",
            doc.metadata.get("doc_type", "sconosciuto"),
        )
        sotto_area = doc.metadata.get("sotto_area", "")
        score = doc.metadata.get("relevance_score", None)
        score_str = f" -- score: {score:.4f}" if score is not None else ""

        extra_meta = {
            k: v for k, v in doc.metadata.items()
            if k not in standard_keys and v is not None and v != ""
        }
        extra_meta_str = ""
        if extra_meta:
            pairs = " | ".join(f"{k}={v}" for k, v in extra_meta.items())
            extra_meta_str = f"\n[meta: {pairs}]"

        context_parts.append(
            f"[Documento {i} -- {formato} -- {sotto_area} -- "
            f"{source}{score_str}]{extra_meta_str}\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(context_parts)


class SearchPersoneInput(BaseModel):
    """Schema di input per il tool search_persone."""

    query: str = Field(
        description=(
            "Domanda COMPLETA dell'utente, ESATTAMENTE come formulata. "
            "NON ridurre a keyword, NON parafrasare. "
            "Esempio corretto: 'Chi è il Professore Rossi?' "
            "Esempio VIETATO: 'Rossi'"
        )
    )
    sotto_area: Optional[Literal[
        "profilo", "didattica", "ricerca", "internazionale", "risorse"
    ]] = Field(
        default=None,
        description=(
            "Filtro sulla sezione della pagina docente. Valori ammessi: "
            "'profilo' (bio, CV, contatti, email, ricevimento), "
            "'didattica' (corsi insegnati, syllabus, programmi), "
            "'ricerca' (pubblicazioni, progetti, laboratori), "
            "'internazionale' (attività internazionali, Erasmus docente), "
            "'risorse' (materiale didattico). "
            "Lascia vuoto se non sei sicuro."
        )
    )
    anno: Optional[int] = Field(
        default=None,
        description=(
            "Anno di riferimento come numero intero (es. 2024, 2025). "
            "Usa il contesto temporale fornito nel messaggio per risolvere "
            "espressioni come 'quest'anno' o 'anno scorso' in un numero. "
            "Lascia vuoto se l'utente non specifica alcun periodo temporale."
        )
    )


class SearchOffertaFormativaInput(BaseModel):
    """Schema di input per il tool search_offerta_formativa."""

    query: str = Field(
        description=(
            "Domanda COMPLETA dell'utente, ESATTAMENTE come formulata. "
            "NON ridurre a keyword, NON parafrasare."
        )
    )
    sotto_area: Optional[Literal[
        "informazioni_corso", "didattica", "aule", "terza_missione",
        "statistiche", "regolamenti", "piani_di_studio", "documentazione_corso"
    ]] = Field(
        default=None,
        description=(
            "Filtro sulla sezione dell'offerta formativa. Valori ammessi: "
            "'informazioni_corso' (info generali sul corso di laurea, presentazione, obiettivi), "
            "'didattica' (piano di studi, insegnamenti del corso, CFU), "
            "'aule' (aule e strutture didattiche del corso), "
            "'terza_missione' (attività di terza missione del corso), "
            "'statistiche' (statistiche sul corso: iscritti, laureati, occupazione), "
            "'regolamenti' (regolamento didattico del CdS, norme, propedeuticità), "
            "'piani_di_studio' (piano di studi ufficiale, curriculum, percorsi), "
            "'documentazione_corso' (altri documenti PDF del corso). "
            "Lascia vuoto se non sei sicuro."
        )
    )
    anno: Optional[int] = Field(
        default=None,
        description=(
            "Anno di riferimento come numero intero (es. 2024, 2025). "
            "Usa il contesto temporale fornito nel messaggio per risolvere "
            "espressioni relative. Lascia vuoto se non specificato."
        )
    )


class SearchDipartimentoInput(BaseModel):
    """Schema di input per il tool search_dipartimento."""

    query: str = Field(
        description=(
            "Domanda COMPLETA dell'utente, ESATTAMENTE come formulata. "
            "NON ridurre a keyword, NON parafrasare."
        )
    )
    sotto_area: Optional[Literal[
        "aule", "laboratori", "bandi",
        "ricerca_dipartimentale", "terza_missione",
        "internazionale", "organizzazione", "generale"
    ]] = Field(
        default=None,
        description=(
            "Filtro sulla sezione del dipartimento. Valori ammessi: "
            "'aule' (aule, strutture didattiche, capienza), "
            "'laboratori' (laboratori di ricerca), "
            "'bandi' (bandi, borse, concorsi, dottorato, avvisi), "
            "'ricerca_dipartimentale' (ricerca del dipartimento), "
            "'terza_missione' (terza missione, impatto sociale), "
            "'internazionale' (Erasmus, mobilita, accordi internazionali), "
            "'organizzazione' (organigramma, commissioni, personale), "
            "'generale' (informazioni generiche dipartimento). "
            "ATTENZIONE: 'strutture' NON esiste. Usa 'aule' o 'laboratori'. "
            "Lascia vuoto se non sei sicuro."
        )
    )
    anno: Optional[int] = Field(
        default=None,
        description=(
            "Anno di riferimento come numero intero (es. 2024, 2025). "
            "Utile per filtrare bandi per anno. "
            "Lascia vuoto se non specificato."
        )
    )


class SearchAllInput(BaseModel):
    """Schema di input per il tool search_all."""

    query: str = Field(
        description=(
            "Domanda COMPLETA dell'utente, ESATTAMENTE come formulata. "
            "NON ridurre a keyword, NON parafrasare."
        )
    )


@tool("search_persone", args_schema=SearchPersoneInput)
def search_persone(
    query: str,
    sotto_area: Optional[str] = None,
    anno: Optional[int] = None,
) -> str:
    """Cerca informazioni su DOCENTI del DIEM: profilo, email, ricevimento, corsi insegnati, syllabus/insegnamenti, ricerca, attività internazionali.

USA QUESTO TOOL per:
- "Chi è il prof. X?" → profilo docente
- "Cosa insegna il prof. X?" → corsi del docente
- "Chi insegna Machine Learning?"
- "Syllabus/programma di Algoritmi" → didattica dell'insegnamento (sotto_area="didattica")
- "Email/ricevimento del prof. X"  → contatti del docente (sotto_area="profilo")

DIRETTIVA: Passa la query INTEGRA nel campo `query`. NON ridurre a keyword."""
    if sotto_area and sotto_area.lower().strip() not in _VALID_SOTTO_AREA_PERSONE:
        logger.warning("sotto_area non valido per PERSONE: '%s'. Ignoro.", sotto_area)
        sotto_area = None
    elif sotto_area:
        sotto_area = sotto_area.lower().strip()

    metadata_filter = {}
    if sotto_area:
        metadata_filter["sotto_area"] = sotto_area
    if anno is not None:
        metadata_filter["anno"] = str(anno)

    metadata_filter = metadata_filter if metadata_filter else None

    logger.debug("search_persone: query='%s', sotto_area=%s, anno=%s", query, sotto_area, anno)

    return _search_collection(
        query, CollectionTarget.PERSONE,
        "search_persone", metadata_filter,
    )


@tool("search_offerta_formativa", args_schema=SearchOffertaFormativaInput)
def search_offerta_formativa(
    query: str,
    sotto_area: Optional[str] = None,
    anno: Optional[int] = None,
) -> str:
    """Cerca informazioni su CORSI DI LAUREA del DIEM: piani di studio, regolamenti, requisiti ammissione, CFU, OFA, tesi, statistiche.

USA QUESTO TOOL per:
- "Piano di studi Informatica triennale" → sotto_area="piani_di_studio"
- "Regolamento Ingegneria Informatica magistrale" → sotto_area="regolamenti"
- "Quali esami ci sono al primo anno?" → sotto_area="didattica"
- "Statistiche occupazionali del corso" → sotto_area="statistiche"
- "Informazioni sul corso di Informatica" → sotto_area="informazioni_corso"

NON contiene info su chi insegna o syllabus di insegnamenti specifici → usa search_persone.

DIRETTIVA: Passa la query INTEGRA nel campo `query`. NON ridurre a keyword."""
    if sotto_area and sotto_area.lower().strip() not in _VALID_SOTTO_AREA_OFFERTA_FORMATIVA:
        logger.warning("sotto_area non valido per OFFERTA_FORMATIVA: '%s'. Ignoro.", sotto_area)
        sotto_area = None
    elif sotto_area:
        sotto_area = sotto_area.lower().strip()

    if sotto_area is None:
        query_lower = query.lower()
        if any(kw in query_lower for kw in [
            "regolamento", "regolamenti", "propedeuticità", "propedeuticita",
            "norme", "regole"
        ]):
            sotto_area = "regolamenti"
        elif any(kw in query_lower for kw in [
            "piano di studi", "piano di studio", "curriculum", "percorso",
            "percorsi", "insegnamenti del corso"
        ]):
            sotto_area = "piani_di_studio"
        elif any(kw in query_lower for kw in [
            "statistica", "statistiche", "occupazione", "occupazionali",
            "laureati", "iscritti"
        ]):
            sotto_area = "statistiche"
        elif any(kw in query_lower for kw in [
            "aula", "aule", "strutture didattiche"
        ]):
            sotto_area = "aule"

    metadata_filter = {}
    if sotto_area:
        metadata_filter["sotto_area"] = sotto_area
    if anno is not None:
        metadata_filter["anno"] = str(anno)

    metadata_filter = metadata_filter if metadata_filter else None

    logger.debug("search_offerta_formativa: query='%s', sotto_area=%s, anno=%s", query, sotto_area, anno)

    return _search_collection(
        query, CollectionTarget.OFFERTA_FORMATIVA,
        "search_offerta_formativa", metadata_filter,
    )


@tool("search_dipartimento", args_schema=SearchDipartimentoInput)
def search_dipartimento(
    query: str,
    sotto_area: Optional[str] = None,
    anno: Optional[int] = None,
) -> str:
    """Cerca informazioni istituzionali del DIEM: bandi, borse, dottorato, aule, laboratori, sedi, Erasmus, ricerca dipartimentale, terza missione.

ATTENZIONE: sotto_area="strutture" NON esiste. Usare "aule", "laboratori" o "sedi".

DIRETTIVA: Passa la query INTEGRA nel campo `query`. NON ridurre a keyword."""
    if sotto_area and sotto_area.lower().strip() not in _VALID_SOTTO_AREA_DIPARTIMENTO:
        logger.warning("sotto_area non valido per DIPARTIMENTO: '%s'. Ignoro.", sotto_area)
        sotto_area = None
    elif sotto_area:
        sotto_area = sotto_area.lower().strip()

    if sotto_area is None:
        query_lower = query.lower()
        if any(kw in query_lower for kw in [
            "bando", "bandi", "borsa", "borse", "assegno", "assegni",
            "concorso", "concorsi", "dottorato", "avviso", "avvisi"
        ]):
            sotto_area = "bandi"
        elif any(kw in query_lower for kw in [
            "laboratorio", "laboratori", "lab "
        ]):
            sotto_area = "laboratori"
        elif any(kw in query_lower for kw in [
            "aula", "aule", "strutture didattiche", "strutture-didattiche"
        ]):
            sotto_area = "aule"
        elif any(kw in query_lower for kw in [
            "terza missione", "terza-missione"
        ]):
            sotto_area = "terza_missione"
        elif any(kw in query_lower for kw in [
            "erasmus", "internazionale", "international", "mobilità"
        ]):
            sotto_area = "internazionale"

    metadata_filter = {}
    if sotto_area:
        metadata_filter["sotto_area"] = sotto_area
    if anno is not None:
        metadata_filter["anno"] = str(anno)

    metadata_filter = metadata_filter if metadata_filter else None

    logger.debug("search_dipartimento: query='%s', sotto_area=%s, anno=%s", query, sotto_area, anno)

    return _search_collection(
        query, CollectionTarget.DIPARTIMENTO,
        "search_dipartimento", metadata_filter,
    )


@tool("search_all", args_schema=SearchAllInput)
def search_all(query: str) -> str:
    """Ricerca trasversale su TUTTA la knowledge base del DIEM. Usa SOLO quando gli altri tool falliscono o la domanda è ambigua/cross-dominio.

DIRETTIVA: Passa la query INTEGRA nel campo `query`."""
    global _last_search_meta

    if _retrieval_engine is None:
        return "Errore interno: motore di ricerca non inizializzato."

    try:
        documents, used_query = _retrieval_engine.retrieve_from_all(query)

        top_links = _extract_top_links(documents)

        _last_search_meta = {
            "rewritten_query": used_query,
            "multi_queries": [used_query],
            "tool_name": "search_all",
            "collection": "ALL (cross-collection)",
            "metadata_filter": None,
            "top_links": top_links,
        }

        if not documents:
            return f"Non ho trovato informazioni per: '{query}'."

        return _format_results(documents)

    except Exception as e:
        logger.error("Errore ricerca all: %s", e)
        return f"Errore durante la ricerca: {str(e)}"


def get_all_tools() -> list:
    """Restituisce tutti i tool disponibili per l'agente."""
    return [
        search_persone,
        search_offerta_formativa,
        search_dipartimento,
        search_all,
    ]