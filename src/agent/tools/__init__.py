"""
agent/tools/__init__.py — Tool RIVISTI per 3 Vector Store.

REFACTORING secondo audit_fattibilita_metadati.md §7-§8:
  ELIMINATI:
    - search_docenti → sostituito da search_persone
    - search_bandi → assorbito in search_dipartimento (sotto_area="bandi")
    - search_strutture_fisiche → assorbito in search_dipartimento (sotto_area="laboratori")
    - get_course_schedule → EasyCourse ESCLUSO da questa implementazione

  NUOVI/RINOMINATI:
    - search_persone: collection PERSONE (ex DOCENTI_DIDATTICA), filtro sotto_area
    - search_offerta_formativa: collection OFFERTA_FORMATIVA (invariata)
    - search_dipartimento: collection DIPARTIMENTO (assorbe bandi + strutture),
                           filtro sotto_area per bandi/laboratori/strutture
    - search_all: cross-collection su tutte e 3 le collection

  FIX APPLICATI:
    - Tool description con direttiva ANTI-COMPRESSIONE: "Passa la query
      dell'utente INTEGRA e COMPLETA nel campo query."
    - Segnale di FALLBACK esplicito nell'output quando non si trovano
      risultati pertinenti, per guidare l'Agente verso il tool alternativo.
    - Routing migliorato per "Chi insegna X?" → search_persone

  FIX MISMATCH sotto_area (BUG AULE):
    Il router (router.py) indicizza i chunk con sotto_area granulari:
      "aule", "laboratori", "sedi", "bandi", "ricerca_dipartimentale",
      "terza_missione", "internazionale", "organizzazione", "generale",
      "alternanza", "eccellenza", "monitoraggio", "personale"
    I valori ammessi e l'inferenza automatica nel tool DEVONO usare
    gli STESSI valori. Il vecchio valore "strutture" NON esiste in Chroma
    ed è stato eliminato.
"""

from __future__ import annotations
import logging
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from langchain.tools import tool
from ingestion.router import CollectionTarget

if TYPE_CHECKING:
    from retrieval.engine import RetrievalEngine

logger = logging.getLogger(__name__)
_retrieval_engine: "RetrievalEngine | None" = None

# Chat history globale per il query rewriting contestuale
_chat_history: list = []

# Metadati dell'ultima ricerca per il sistema di logging
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


# ── Valori ammessi per sotto_area di PERSONE (audit §6) ──
_VALID_SOTTO_AREA_PERSONE = frozenset({
    "profilo", "didattica", "ricerca", "internazionale", "risorse"
})

# ── Valori ammessi per sotto_area di DIPARTIMENTO (audit §6) ──
# FIX MISMATCH AULE: allineati ai valori REALI prodotti dal router
# (router.py → _classify_dipartimento_sottoarea).
# Il vecchio valore "strutture" NON esiste nei metadati Chroma.
# I valori reali per le strutture fisiche sono: "aule", "laboratori", "sedi".
_VALID_SOTTO_AREA_DIPARTIMENTO = frozenset({
    "aule",                     # ← strutture-didattiche, aula, aule
    "laboratori",               # ← laboratorio, laboratori, lab
    "sedi",                     # ← sede, sedi, edificio, campus
    "bandi",                    # ← bando, bandi, borsa, assegno, dottorato
    "ricerca_dipartimentale",   # ← /ricerca/
    "terza_missione",           # ← /terza-missione/
    "internazionale",           # ← /international/
    "organizzazione",           # ← organi, commissioni
    "alternanza",               # ← /didattica/alternanza
    "eccellenza",               # ← /dipartimento/eccellenza
    "monitoraggio",             # ← /dati-di-monitoraggio
    "personale",                # ← /dipartimento/personale
    "generale",                 # ← fallback generico
})

# Contatore errori per tool (anti-loop)
_tool_error_counts: Dict[str, int] = {}
_MAX_TOOL_RETRIES = 2


# ── Segnale di FALLBACK per l'Agente ──
_FALLBACK_SIGNAL_PERSONE = (
    "Non ho trovato informazioni pertinenti per: '{query}' nel Vector Store PERSONE. "
    "SUGGERIMENTO: prova a cercare con search_offerta_formativa se la domanda "
    "riguarda un corso di laurea o un piano di studi, oppure con search_dipartimento "
    "se riguarda strutture, bandi o informazioni istituzionali."
)

_FALLBACK_SIGNAL_OFFERTA = (
    "Non ho trovato informazioni pertinenti per: '{query}' nel Vector Store OFFERTA_FORMATIVA. "
    "SUGGERIMENTO: se la domanda riguarda un docente specifico o chi insegna un "
    "certo insegnamento, prova con search_persone. Se riguarda bandi, strutture "
    "o informazioni istituzionali, prova con search_dipartimento."
)

_FALLBACK_SIGNAL_DIPARTIMENTO = (
    "Non ho trovato informazioni pertinenti per: '{query}' nel Vector Store DIPARTIMENTO. "
    "SUGGERIMENTO: prova con search_persone se la domanda riguarda un docente, "
    "o con search_offerta_formativa se riguarda un corso di laurea."
)

_FALLBACK_SIGNAL_GENERIC = (
    "Non ho trovato informazioni pertinenti per: '{query}'. "
    "Prova a riformulare la domanda."
)


def _search_collection(
    query: str,
    collection: CollectionTarget,
    tool_name: str,
    metadata_filter: Optional[dict] = None,
) -> str:
    """
    Helper condiviso per la ricerca in una singola collection.

    Flusso: retrieve() → (documents, rewritten_query, multi_queries)
    I metadati vengono salvati in _last_search_meta per il logging.

    FIX: Quando non si trovano risultati, il messaggio di ritorno
    include un SEGNALE DI FALLBACK esplicito che indica all'Agente
    quale tool alternativo provare.
    """
    global _last_search_meta

    if _retrieval_engine is None:
        return "Errore interno: motore di ricerca non inizializzato."

    tool_key = f"{collection.value}:{query[:50]}"

    print(f"SEARCH COLLECTION: {query}")

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
            link = doc.metadata.get("url_originale",
                   doc.metadata.get("source_url", "N/D"))
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
            # FIX: Segnale di fallback specifico per collection
            fallback_map = {
                CollectionTarget.PERSONE: _FALLBACK_SIGNAL_PERSONE,
                CollectionTarget.OFFERTA_FORMATIVA: _FALLBACK_SIGNAL_OFFERTA,
                CollectionTarget.DIPARTIMENTO: _FALLBACK_SIGNAL_DIPARTIMENTO,
            }
            signal = fallback_map.get(collection, _FALLBACK_SIGNAL_GENERIC)
            return signal.format(query=query)

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
        # Usa url_originale (nuovo schema audit §6), fallback a source_url
        source = doc.metadata.get("url_originale",
                 doc.metadata.get("source_url", "fonte non disponibile"))
        formato = doc.metadata.get("formato_sorgente",
                  doc.metadata.get("doc_type", "sconosciuto"))
        sotto_area = doc.metadata.get("sotto_area", "")
        score = doc.metadata.get("relevance_score", None)
        score_str = f" — score: {score:.4f}" if score is not None else ""

        # Metadati extra significativi (esclusi quelli standard)
        standard_keys = {
            "source_url", "url_originale", "doc_type", "formato_sorgente",
            "doc_category", "sotto_area", "relevance_score", "start_index",
            "source_file", "source_domain",
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
            f"[Documento {i} — {formato} — {sotto_area} — "
            f"{source}{score_str}]{extra_meta_str}\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(context_parts)


# ============================================================
# TOOL 1: PERSONE (ex search_docenti) — audit §7-§8
# ============================================================

@tool("search_persone")
def search_persone(query: str, sotto_area: Optional[str] = None) -> str:
    """Cerca informazioni su PERSONE (docenti) del DIEM.

    <description>
    ## Quando usare questo tool
    Usa per domande su PERSONE: profilo, curriculum, email, ricevimento,
    corsi insegnati da un docente, aree di ricerca personali, attività
    internazionali di un docente specifico.

    ## REGOLA CRITICA — "Chi insegna X?"
    Se l'utente chiede "Chi insegna [insegnamento]?" o "Chi è il docente
    di [insegnamento]?", questo è IL TOOL CORRETTO (non search_offerta_formativa).
    Il VS PERSONE contiene il campo nomi_insegnamenti con i nomi degli
    insegnamenti tenuti da ciascun docente.

    ## Routing (audit §7):
    - "Chi è il prof. X?" → search_persone
    - "Cosa insegna il prof. X?" → search_persone
    - "Email/ricevimento del prof. X" → search_persone
    - "Chi insegna Machine Learning?" → search_persone (NON offerta_formativa!)
    - Ponte verso OFFERTA: se il docente menziona un insegnamento, il
      campo nomi_insegnamenti consente di rintracciare il corso in
      search_offerta_formativa.

    ## NON usare per:
    Corsi di laurea senza riferimento a un docente (usa search_offerta_formativa),
    bandi (usa search_dipartimento), strutture fisiche (usa search_dipartimento).

    ## DIRETTIVA ANTI-COMPRESSIONE:
    Passa la query dell'utente INTEGRA e COMPLETA nel campo `query`.
    NON ridurre la query a sole keyword o nomi propri.
    Esempio CORRETTO: query="Chi è il Professore Francesco Basile?"
    Esempio SBAGLIATO: query="Francesco Basile"
    </description>

    Args:
        query: La domanda COMPLETA dell'utente sul docente, passata INTEGRA.
               Esempi: "Chi è Mario Vento?", "Corsi insegnati dal prof. Capuano",
               "Email e ricevimento del prof. Greco", "Chi insegna Machine Learning?".
        sotto_area: Filtro opzionale sulla sottosezione del docente.
                    Valori ammessi: "profilo", "didattica", "ricerca",
                    "internazionale", "risorse".
    """
    if sotto_area and sotto_area.lower().strip() not in _VALID_SOTTO_AREA_PERSONE:
        logger.warning(
            f"Parametro sotto_area non valido per PERSONE: '{sotto_area}'. "
            f"Ignoro il filtro."
        )
        sotto_area = None
    elif sotto_area:
        sotto_area = sotto_area.lower().strip()

    metadata_filter = None
    if sotto_area:
        metadata_filter = {"sotto_area": sotto_area}

    print(f"QUERY ARRIVATA A SEARCH PERSONE: {query}")

    return _search_collection(
        query, CollectionTarget.PERSONE,
        "search_persone", metadata_filter
    )


# ============================================================
# TOOL 2: OFFERTA FORMATIVA — audit §7-§8
# ============================================================

@tool("search_offerta_formativa")
def search_offerta_formativa(query: str) -> str:
    """Cerca informazioni su corsi di laurea, piani di studio e regolamenti.

    <description>
    ## Quando usare questo tool
    Usa per: corsi di laurea del DIEM, piani di studio, regolamenti didattici,
    requisiti di ammissione, crediti, programmi, OFA, tesi, statistiche corsi,
    insegnamenti (quando NON si cerca un docente specifico e NON si chiede
    "chi insegna").

    ## Routing (audit §7):
    - "Piano di studi di Informatica triennale" → search_offerta_formativa
    - "Regolamento Ingegneria Informatica magistrale" → search_offerta_formativa
    - "Quali esami ci sono al primo anno?" → search_offerta_formativa
    - Se serve info su un DOCENTE → usa search_persone
    - Se si chiede "Chi insegna X?" → usa search_persone (NON questo tool!)

    ## IMPORTANTE:
    Questo Vector Store NON contiene informazioni su chi insegna i singoli
    insegnamenti. Per domande tipo "Chi insegna Machine Learning?" usa
    search_persone.

    ## DIRETTIVA ANTI-COMPRESSIONE:
    Passa la query dell'utente INTEGRA e COMPLETA nel campo `query`.
    NON ridurre la query a sole keyword o nomi propri.
    Esempio CORRETTO: query="Parlami della didattica di Analisi Matematica 1 di Vittorio Zampoli"
    Esempio SBAGLIATO: query="Analisi Matematica 1"
    </description>

    Args:
        query: La domanda COMPLETA dell'utente su corsi e offerta formativa,
               passata INTEGRA senza compressione.
    """
    return _search_collection(
        query, CollectionTarget.OFFERTA_FORMATIVA,
        "search_offerta_formativa"
    )


# ============================================================
# TOOL 3: DIPARTIMENTO (assorbe bandi + strutture) — audit §7-§8
# ============================================================

@tool("search_dipartimento")
def search_dipartimento(query: str, sotto_area: Optional[str] = None) -> str:
    """Cerca informazioni istituzionali, bandi, laboratori e strutture del DIEM.

    <description>
    ## Quando usare questo tool
    Usa per TUTTO ciò che riguarda il dipartimento DIEM come istituzione:
    aree di ricerca dipartimentali, progetti finanziati, Erasmus e
    internazionalizzazione, terza missione, organi, commissioni, bandi,
    borse di studio, assegni di ricerca, avvisi amministrativi, dottorato,
    aule, laboratori, sedi, strutture fisiche.

    ## Routing — VALORI sotto_area CORRETTI:
    - "Dove si trova l'aula 126?" → search_dipartimento (sotto_area="aule")
    - "Dove si trova l'aula 152?" → search_dipartimento (sotto_area="aule")
    - "Quali aule ci sono?" → search_dipartimento (sotto_area="aule")
    - "Laboratorio ICAR" → search_dipartimento (sotto_area="laboratori")
    - "Dove si trova la sede?" → search_dipartimento (sotto_area="sedi")
    - "Bandi di dottorato DIEM" → search_dipartimento (sotto_area="bandi")
    - "Borsa di studio DIEM" → search_dipartimento (sotto_area="bandi")
    - "Progetti di ricerca DIEM" → search_dipartimento (sotto_area="ricerca_dipartimentale")
    - "Erasmus DIEM" → search_dipartimento (sotto_area="internazionale")
    - "Terza missione DIEM" → search_dipartimento (sotto_area="terza_missione")
    - "Commissione paritetica" → search_dipartimento (sotto_area="generale")

    ## ATTENZIONE — VALORI sotto_area:
    NON usare "strutture" come sotto_area — questo valore NON ESISTE
    nella knowledge base. Usare invece i valori specifici:
    - "aule" per aule e strutture didattiche
    - "laboratori" per laboratori
    - "sedi" per sedi, edifici, campus

    ## DIRETTIVA ANTI-COMPRESSIONE:
    Passa la query dell'utente INTEGRA e COMPLETA nel campo `query`.
    NON ridurre la query a sole keyword o nomi propri.
    </description>

    Args:
        query: La domanda COMPLETA dell'utente su dipartimento, bandi, strutture,
               passata INTEGRA senza compressione.
        sotto_area: Filtro opzionale. Valori ammessi: "aule", "laboratori",
                    "sedi", "bandi", "ricerca_dipartimentale", "terza_missione",
                    "internazionale", "organizzazione", "generale".
                    ATTENZIONE: NON usare "strutture" — usare "aule",
                    "laboratori" o "sedi" a seconda del contesto.
    """
    if sotto_area and sotto_area.lower().strip() not in _VALID_SOTTO_AREA_DIPARTIMENTO:
        logger.warning(
            f"Parametro sotto_area non valido per DIPARTIMENTO: '{sotto_area}'. "
            f"Ignoro il filtro."
        )
        sotto_area = None
    elif sotto_area:
        sotto_area = sotto_area.lower().strip()

    # ── FIX MISMATCH sotto_area: inferenza allineata ai valori reali in Chroma ──
    # Il router (router.py → _classify_dipartimento_sottoarea) assegna:
    #   "aule" per strutture-didattiche, aula, aule
    #   "laboratori" per laboratorio, laboratori, lab
    #   "sedi" per sede, sedi, edificio, campus
    #   "bandi" per bando, bandi, borsa, assegno, dottorato
    # L'inferenza qui DEVE usare gli STESSI valori.
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
            # FIX: era "strutture" → ora "aule" (allineato a Chroma)
            sotto_area = "aule"
        elif any(kw in query_lower for kw in [
            "sede", "sedi", "edificio", "campus"
        ]):
            # FIX: era raggruppato in "strutture" → ora "sedi" (allineato a Chroma)
            sotto_area = "sedi"
        elif any(kw in query_lower for kw in [
            "terza missione", "terza-missione"
        ]):
            sotto_area = "terza_missione"
        elif any(kw in query_lower for kw in [
            "erasmus", "internazionale", "international", "mobilità"
        ]):
            sotto_area = "internazionale"

    metadata_filter = None
    if sotto_area:
        metadata_filter = {"sotto_area": sotto_area}

    return _search_collection(
        query, CollectionTarget.DIPARTIMENTO,
        "search_dipartimento", metadata_filter
    )


# ============================================================
# TOOL 4: Ricerca Trasversale — audit §7
# ============================================================

@tool("search_all")
def search_all(query: str) -> str:
    """Ricerca trasversale su TUTTA la knowledge base del DIEM.

    <description>
    ## Quando usare questo tool
    Usa SOLO quando: la domanda è ambigua, copre più aree (es. docente
    + corso + struttura), o gli altri tool non hanno dato risultati.

    ## NON usare come prima scelta:
    Prova prima il tool specifico (search_persone, search_offerta_formativa,
    search_dipartimento).

    ## DIRETTIVA ANTI-COMPRESSIONE:
    Passa la query dell'utente INTEGRA e COMPLETA nel campo `query`.
    </description>

    Args:
        query: La domanda COMPLETA dell'utente, generica o cross-dominio,
               passata INTEGRA senza compressione.
    """
    global _last_search_meta

    if _retrieval_engine is None:
        return "Errore interno: motore di ricerca non inizializzato."
    try:
        documents, used_query = _retrieval_engine.retrieve_from_all(query)

        top_links = []
        for doc in documents[:5]:
            link = doc.metadata.get("url_originale",
                   doc.metadata.get("source_url", "N/D"))
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
# REGISTRY: lista di tutti i tool disponibili
# ============================================================

def get_all_tools() -> list:
    """
    Restituisce tutti i tool disponibili per l'agente.

    AGGIORNATO per 3 Vector Store (audit §8):
      - search_persone (ex search_docenti)
      - search_offerta_formativa
      - search_dipartimento (assorbe ex search_bandi + search_strutture_fisiche)
      - search_all

    RIMOSSI:
      - search_docenti → rinominato in search_persone
      - search_bandi → assorbito in search_dipartimento
      - search_strutture_fisiche → assorbito in search_dipartimento
      - get_course_schedule → EasyCourse escluso da questa implementazione
    """
    return [
        search_persone,
        search_offerta_formativa,
        search_dipartimento,
        search_all,
    ]