"""
retrieval/engine.py — QueryOptimizer con Domain-Aware Rewriting + Multiquery.

REFACTORING APPLICATO (4 requisiti):
  1. MULTIQUERY: il metodo expand() è ora richiamato nel flusso di retrieve().
     Nuovo prompt estensivo per generare 3 varianti semantiche coerenti
     con il contesto universitario DIEM.
  2. QUERY REWRITING INTELLIGENTE: il prompt di rewrite ora include uno
     step di valutazione della rilevanza della history. Se la domanda
     corrente NON è correlata alla history, la history viene ignorata
     completamente per evitare il "context bleed" (es. "Chi è Marcelli?"
     seguito da "Dove si trova l'aula 126?" non deve fondere i due topic).
  3. CrossEncoderReranker con score_threshold per filtrare falsi positivi.
  4. retrieve() ora esegue: rewrite → expand (multiquery) → retrieval
     parallelo → merge → rerank.
"""

import logging
from datetime import datetime
from typing import List, Optional, Set

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from config.settings import RerankerConfig
from ingestion.router import CollectionTarget

logger = logging.getLogger(__name__)


class QueryOptimizer:
    """
    Pre-Retrieval: Domain-Aware Query Rewriting + Multiquery Expansion.

    Due fasi:
      1. rewrite(): Riscrive la query con espansione di dominio e,
         SOLO se la domanda è correlata alla history, integra il contesto
         conversazionale per risolvere coreferenze anaforiche.
      2. expand(): Genera 3 varianti semantiche della query riscritta
         per massimizzare la copertura nel vector database.
    """

    # ================================================================
    # PROMPT 1: QUERY REWRITING CON VALUTAZIONE RILEVANZA HISTORY
    # ================================================================

    REWRITE_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         """Sei un ottimizzatore di query per un sistema di ricerca semantica \
del Dipartimento DIEM dell'Università degli Studi di Salerno.

Data e ora correnti: {current_datetime}

Il tuo compito è RISCRIVERE la domanda dell'utente in una query ottimizzata \
per la ricerca in un database vettoriale.

═══════════════════════════════════════════════════════════════
STEP A — VALUTAZIONE DELLA RILEVANZA DELLA HISTORY
═══════════════════════════════════════════════════════════════

Prima di tutto, analizza la CRONOLOGIA della conversazione (se presente) \
e la DOMANDA CORRENTE dell'utente. Determina se la domanda corrente è \
semanticamente correlata alla cronologia:

• CORRELATA: la domanda corrente fa riferimento (anche implicito) a \
  entità, persone, corsi, aule, argomenti menzionati nella cronologia. \
  Esempi: "e dove insegna?" (dopo aver parlato di un docente), \
  "quali sono i prerequisiti?" (dopo aver parlato di un corso).

• NON CORRELATA: la domanda corrente introduce un argomento completamente \
  nuovo, senza alcun legame con la cronologia. \
  Esempi: si parlava di "Angelo Marcelli" e ora si chiede "Dove si trova \
  l'aula 126?". L'aula 126 NON ha relazione con Marcelli. \
  Altro esempio: si parlava di "borse di studio" e ora si chiede \
  "Chi è il prof. Vento?".

SE LA DOMANDA È CORRELATA alla history:
  → Integra le informazioni dalla history per risolvere pronomi, \
    riferimenti impliciti e coreferenze anaforiche.
  → Esempio: History contiene "Chi è Mario Vento?" + Domanda "e dove insegna?" \
    → "In quali corsi insegna il professore Mario Vento del DIEM?"

SE LA DOMANDA NON È CORRELATA alla history:
  → IGNORA COMPLETAMENTE la history. Tratta la domanda come se fosse \
    la prima della conversazione. Non mescolare mai argomenti diversi.

═══════════════════════════════════════════════════════════════
STEP B — RISCRITTURA DELLA QUERY
═══════════════════════════════════════════════════════════════

Dopo aver determinato se usare o meno la history, riscrivi la query \
seguendo TUTTE queste regole:

REGOLA 1 — ESPANSIONE, MAI COMPRESSIONE:
La query riscritta DEVE essere una frase completa e discorsiva, \
PIÙ specifica e dettagliata dell'originale. \
NON ridurre MAI a semplici keyword o frammenti.
VIETATO: "Dove si trova l'aula 126?" → "aula 126" (questo è VIETATO!)
CORRETTO: "Dove si trova l'aula 126?" → "Dove si trova l'aula 126 nel \
campus dell'Università di Salerno, in quale edificio e a quale piano \
del dipartimento DIEM?"
VIETATO: "Chi è Angelo Marcelli?" → "Angelo Marcelli"
CORRETTO: "Chi è Angelo Marcelli?" → "Profilo accademico, qualifica, \
ruolo istituzionale e contatti del professore Angelo Marcelli del \
dipartimento DIEM dell'Università di Salerno"

REGOLA 2 — CONTESTO DI DOMINIO:
Aggiungi "dipartimento DIEM" o "Università di Salerno" se non già presenti.

REGOLA 3 — QUERY SU PERSONE:
Quando si chiede CHI È una persona:
→ Espandi verso: curriculum, qualifica accademica, ruolo, contatti, ricevimento
Quando si chiedono CORSI INSEGNATI:
→ Espandi verso: insegnamenti, corsi di laurea, anno accademico
Quando si chiede la RICERCA di un docente:
→ Espandi verso: aree di ricerca, pubblicazioni, gruppi

REGOLA 4 — QUERY SU STRUTTURE FISICHE:
Quando si chiede DOVE SI TROVA un'aula, laboratorio o sede:
→ Espandi verso: ubicazione, edificio, piano, campus di Fisciano
→ NON mescolare MAI con informazioni su docenti o corsi a meno che \
  la domanda non lo richieda esplicitamente.

REGOLA 5 — RISOLUZIONE TEMPORALE:
Se la domanda contiene riferimenti temporali relativi ("domani", \
"lunedì prossimo"), risolvili con la data corrente fornita sopra.

═══════════════════════════════════════════════════════════════
OUTPUT
═══════════════════════════════════════════════════════════════
Rispondi con SOLO la query riscritta. Una frase completa e discorsiva. \
Nient'altro. Nessuna spiegazione, nessun prefisso."""),
        ("placeholder", "{history}"),
        ("human", "{question}"),
    ])

    # ================================================================
    # PROMPT 2: MULTIQUERY EXPANSION
    # ================================================================

    MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         """Sei un esperto di ricerca semantica specializzato nel dominio \
dell'Università degli Studi di Salerno, in particolare del Dipartimento \
di Ingegneria dell'Informazione ed Elettrica e Matematica applicata (DIEM).

Il tuo compito è generare ESATTAMENTE 3 varianti semantiche della \
domanda fornita dall'utente. Ogni variante deve:

1. PRESERVARE L'INTENTO ORIGINALE: tutte le varianti devono cercare \
   la stessa informazione della domanda originale, ma da angolazioni \
   diverse per massimizzare la copertura nel database vettoriale.

2. VARIARE IL VOCABOLARIO: usa sinonimi, parafrasi e formulazioni \
   alternative. Se la domanda originale usa "docente", una variante \
   può usare "professore" o "insegnante". Se dice "aula", prova \
   "sala", "classe" o "stanza".

3. VARIARE LA STRUTTURA: alterna tra domande dirette, formulazioni \
   descrittive e ricerche per attributi. Esempio per "Chi è Mario Vento?":
   - Variante A (profilo): "Curriculum e qualifica accademica del \
     prof. Mario Vento al DIEM UniSA"
   - Variante B (contatti): "Contatti istituzionali, email e ufficio \
     di Mario Vento Università di Salerno"
   - Variante C (ruolo): "Ruolo e dipartimento di appartenenza del \
     docente Mario Vento DIEM Salerno"

4. MANTENERE IL CONTESTO UNIVERSITARIO: ogni variante deve contenere \
   almeno un riferimento al contesto (DIEM, UniSA, Università di Salerno, \
   Fisciano, dipartimento).

5. ESSERE FRASI COMPLETE: ogni variante deve essere una frase completa \
   e discorsiva, MAI ridotta a sole keyword.

FORMATO OUTPUT:
Restituisci SOLO le 3 varianti, una per riga, senza numerazione, \
senza prefissi, senza spiegazioni. Solo 3 righe di testo."""),
        ("human", "{question}"),
    ])

    def __init__(self, llm_chat_model):
        self._llm = llm_chat_model
        self._rewrite_chain = self.REWRITE_PROMPT | self._llm | RunnableLambda(
            lambda msg: msg.content.strip()
        )
        self._multi_query_chain = self.MULTI_QUERY_PROMPT | self._llm | RunnableLambda(
            lambda msg: [q.strip() for q in msg.content.strip().split("\n") if q.strip()]
        )

    @staticmethod
    def _get_current_datetime_str() -> str:
        """Genera una stringa datetime leggibile in italiano."""
        now = datetime.now()
        giorni = ["lunedì", "martedì", "mercoledì", "giovedì",
                  "venerdì", "sabato", "domenica"]
        mesi = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
                "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"]
        return (
            f"{giorni[now.weekday()]} {now.day} {mesi[now.month - 1]} {now.year}, "
            f"ore {now.strftime('%H:%M')}"
        )

    def rewrite(self, question: str, history: Optional[list] = None) -> str:
        """
        Riscrivi la query con valutazione della rilevanza della history.

        Il prompt include lo Step A (valutazione rilevanza) che impedisce
        il context bleed: se la domanda NON è correlata alla history,
        quest'ultima viene ignorata automaticamente dal modello.

        Args:
            question: La query originale dell'utente.
            history: Lista opzionale di BaseMessage LangChain.

        Returns:
            La query riscritta (espansa e contestualizzata).
        """
        try:
            result = self._rewrite_chain.invoke({
                "question": question,
                "history": history or [],
                "current_datetime": self._get_current_datetime_str(),
            })

            # Sanity check: la query riscritta non deve essere più corta
            # del 50% dell'originale (indicherebbe compressione a keyword).
            if len(result) < len(question) * 0.5:
                logger.warning(
                    f"Query rewriting sospetto (compressione): "
                    f"'{question}' → '{result}'. Uso l'originale."
                )
                return question

            # Sanity check: la query riscritta non deve essere vuota
            if not result.strip():
                logger.warning("Query rewriting ha prodotto stringa vuota.")
                return question

            logger.info(f"Query rewritten: '{question}' → '{result}'")
            return result

        except Exception as e:
            logger.warning(f"Errore rewriting, uso query originale: {e}")
            return question

    def expand(self, question: str) -> List[str]:
        """
        Genera varianti semantiche della query per copertura nel vector DB.

        Restituisce sempre la query originale come primo elemento,
        seguita da max 3 varianti generate dal modello.

        Args:
            question: La query (già riscritta) da espandere.

        Returns:
            Lista di 1-4 query: [originale, variante1, variante2, variante3].
        """
        try:
            variants = self._multi_query_chain.invoke({"question": question})
            # Filtra varianti vuote o duplicate
            seen: Set[str] = {question.strip().lower()}
            unique_variants = []
            for v in variants[:3]:
                v_clean = v.strip()
                if v_clean and v_clean.lower() not in seen:
                    seen.add(v_clean.lower())
                    unique_variants.append(v_clean)
            logger.info(
                f"Multiquery expansion: {len(unique_variants)} varianti generate"
            )
            return [question] + unique_variants
        except Exception as e:
            logger.warning(f"Errore multi-query expansion: {e}")
            return [question]


# ============================================================
# CROSS-ENCODER RERANKER (Post-Retrieval)
# ============================================================

class CrossEncoderReranker:
    """
    Post-Retrieval: ri-ordina i documenti candidati con Cross-Encoder.

    Score threshold a -5.0 per filtrare documenti irrilevanti.
    """

    DEFAULT_SCORE_THRESHOLD = -5.0

    def __init__(self, config: RerankerConfig):
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(config.model_name)
        self._top_n = config.top_n
        self._score_threshold = self.DEFAULT_SCORE_THRESHOLD
        logger.info(f"Cross-Encoder Reranker: {config.model_name}")

    def rerank(
        self, query: str, documents: List[Document], top_n: Optional[int] = None
    ) -> List[Document]:
        """Ri-ordina e filtra i documenti per rilevanza."""
        if not documents:
            return []

        top_n = top_n or self._top_n

        pairs = [[query, doc.page_content] for doc in documents]
        scores = self._model.predict(pairs)

        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

        result = []
        for doc, score in ranked[:top_n]:
            if score < self._score_threshold:
                logger.debug(
                    f"Documento filtrato (score {score:.4f} < threshold "
                    f"{self._score_threshold}): "
                    f"{doc.metadata.get('source_url', 'N/D')[:80]}"
                )
                continue
            doc.metadata["relevance_score"] = float(score)
            result.append(doc)

        if not result and documents:
            logger.warning(
                f"Tutti i {len(documents)} documenti candidati filtrati "
                f"(score < {self._score_threshold}). Query: '{query[:80]}'"
            )

        return result


# ============================================================
# RETRIEVAL ENGINE (con Multiquery integrata)
# ============================================================

class RetrievalEngine:
    """
    Orchestratore retrieval con flusso completo:
      rewrite → expand (multiquery) → retrieval parallelo → merge → rerank.

    La multiquery è ora integrata operativamente: per ogni query,
    il sistema genera varianti semantiche, esegue il retrieval su
    ciascuna variante, deduplica i risultati e poi rerank il tutto.
    """

    def __init__(self, indexer, reranker, query_optimizer=None):
        self._indexer = indexer
        self._reranker = reranker
        self._optimizer = query_optimizer
        self._collection_retrievers = {}
        for target in CollectionTarget:
            self._collection_retrievers[target.value] = (
                indexer.get_collection_retriever(target)
            )
        self._pc_retriever = indexer.get_parent_child_retriever()

    def retrieve(
        self,
        query: str,
        collection: Optional[str] = None,
        metadata_filter: Optional[dict] = None,
        chat_history: Optional[list] = None,
    ) -> tuple:
        """
        Retrieval completo: rewrite → multiquery expand → retrieval → rerank.

        Flusso:
          1. Query Rewriting (espansione di dominio + risoluzione coreferenze)
          2. Multiquery Expansion (3 varianti semantiche)
          3. Retrieval parallelo per ogni variante
          4. Deduplicazione dei candidati
          5. Reranking con Cross-Encoder

        Args:
            query: La query originale dell'utente.
            collection: CollectionTarget.value. Se None, retrieve_from_all().
            metadata_filter: Filtro Chroma per restringere i risultati.
            chat_history: Lista di BaseMessage per contesto conversazionale.

        Returns:
            Tupla (documenti_reranked, query_riscritta, lista_multiqueries).
        """
        # ── STEP 1: QUERY REWRITING ──
        effective_query = query
        if self._optimizer:
            effective_query = self._optimizer.rewrite(query, chat_history)

        # ── STEP 2: MULTIQUERY EXPANSION ──
        multi_queries = [effective_query]
        if self._optimizer:
            multi_queries = self._optimizer.expand(effective_query)

        if collection is None:
            docs, _ = self.retrieve_from_all(effective_query)
            return docs, effective_query, multi_queries

        # ── STEP 3: RETRIEVAL PARALLELO PER OGNI VARIANTE ──
        all_candidates = []
        seen_hashes: Set[int] = set()

        for mq in multi_queries:
            if metadata_filter:
                retriever = self._get_filtered_retriever(collection, metadata_filter)
            else:
                retriever = self._collection_retrievers.get(collection)

            if retriever is None:
                logger.error(f"Collection sconosciuta: {collection}")
                continue

            try:
                candidates = retriever.invoke(mq)
                for doc in candidates:
                    h = hash(doc.page_content[:200])
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        all_candidates.append(doc)
            except Exception as e:
                logger.warning(f"Errore retrieval multiquery '{mq[:60]}': {e}")

        # Parent-Child merge per offerta_formativa
        if collection == CollectionTarget.OFFERTA_FORMATIVA.value and not metadata_filter:
            try:
                pc_docs = self._pc_retriever.invoke(effective_query)
                for doc in pc_docs:
                    h = hash(doc.page_content[:200])
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        all_candidates.append(doc)
            except Exception as e:
                logger.warning(f"Errore retrieval Parent-Child: {e}")

        # ── STEP 4: RERANKING ──
        final_docs = self._reranker.rerank(effective_query, all_candidates)
        return final_docs, effective_query, multi_queries

    def retrieve_from_all(self, query: str) -> tuple:
        """
        Retrieval cross-collection con multiquery.
        """
        # Genera multiquery
        multi_queries = [query]
        if self._optimizer:
            multi_queries = self._optimizer.expand(query)

        all_candidates = []
        seen_hashes: Set[int] = set()

        for mq in multi_queries:
            for collection_name, retriever in self._collection_retrievers.items():
                try:
                    docs = retriever.invoke(mq)
                    for doc in docs:
                        h = hash(doc.page_content[:200])
                        if h not in seen_hashes:
                            seen_hashes.add(h)
                            all_candidates.append(doc)
                except Exception as e:
                    logger.warning(f"Errore retrieval da {collection_name}: {e}")

            # Parent-Child
            try:
                pc_docs = self._pc_retriever.invoke(mq)
                for doc in pc_docs:
                    h = hash(doc.page_content[:200])
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        all_candidates.append(doc)
            except Exception as e:
                logger.warning(f"Errore retrieval Parent-Child: {e}")

        final_docs = self._reranker.rerank(query, all_candidates)
        return final_docs, query

    def _get_filtered_retriever(self, collection_name: str, metadata_filter: dict):
        """Crea un retriever Chroma con filtro metadata."""
        try:
            target = CollectionTarget(collection_name)
        except ValueError:
            return None

        chroma_collection = self._indexer._collections[target]
        chroma_where = self._build_chroma_filter(metadata_filter)

        return chroma_collection.as_retriever(
            search_type=self._indexer._settings.vectorstore.search_type,
            search_kwargs={
                "k": self._indexer._settings.vectorstore.search_k,
                "filter": chroma_where,
            },
        )

    @staticmethod
    def _build_chroma_filter(metadata_filter: dict) -> dict:
        """Converte un dict in formato filtro Chroma."""
        conditions = []

        for key, value in metadata_filter.items():
            if key.startswith("$"):
                return metadata_filter
            elif isinstance(value, dict):
                conditions.append({key: value})
            elif isinstance(value, list):
                conditions.append({key: {"$in": value}})
            else:
                conditions.append({key: {"$eq": value}})

        if len(conditions) == 1:
            return conditions[0]
        elif len(conditions) > 1:
            return {"$and": conditions}

        return {}