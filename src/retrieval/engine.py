"""
retrieval/engine.py — RetrievalEngine RIVISTO per 3 Vector Store.

REFACTORING secondo audit_fattibilita_metadati.md:
  - Aggiornato per 3 collection: PERSONE, OFFERTA_FORMATIVA, DIPARTIMENTO
  - QueryOptimizer: REWRITE PROMPT completamente riscritto per:
    * Conservazione entità (nomi docenti, insegnamenti, corsi di laurea)
    * Anti-generalizzazione (NON sostituire concetti specifici con generici)
    * Anti-compressione (la query riscritta deve essere >= l'originale)
  - MULTI_QUERY PROMPT: riscritto con regole ferree di conservazione entità
  - CrossEncoderReranker invariato
  - RetrievalEngine aggiornato per i nuovi nomi collection

FLUSSO:
  rewrite → expand (multiquery) → retrieval parallelo → merge → rerank
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
      2. expand(): Genera 3 varianti semantiche della query riscritta.

    FIX CRITICO APPLICATO:
      - REWRITE PROMPT riscritto con regole ferree:
        * Regola 1: NON generalizzare (se chiedi "orario di ricevimento",
          la query riscritta DEVE contenere "orario di ricevimento")
        * Regola 2: Conservazione entità (nomi docenti, insegnamenti,
          corsi di laurea INTEGRI nella query riscritta)
        * Anti-compressione: la query riscritta DEVE essere >= originale
      - MULTI_QUERY PROMPT riscritto con le stesse regole
    """

    REWRITE_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         """<role>
## Identità
Sei un ottimizzatore di query per un sistema di ricerca semantica 
del Dipartimento DIEM dell'Università degli Studi di Salerno.
</role>

<temporal_context>
Data e ora correnti: {current_datetime}
</temporal_context>

<task>
## Compito
Il tuo compito è RISCRIVERE la domanda dell'utente in una query ottimizzata 
per la ricerca in un database vettoriale.
</task>

<step_a>
## STEP A — VALUTAZIONE DELLA RILEVANZA DELLA HISTORY

Prima di tutto, analizza la CRONOLOGIA della conversazione (se presente) 
e la DOMANDA CORRENTE dell'utente. Determina se la domanda corrente è 
semanticamente correlata alla cronologia:

### CORRELATA:
La domanda corrente fa riferimento (anche implicito) a entità, persone, 
corsi, aule, argomenti menzionati nella cronologia.
Esempi: "e dove insegna?" (dopo aver parlato di un docente), 
"quali sono i prerequisiti?" (dopo aver parlato di un corso).

### NON CORRELATA:
La domanda corrente introduce un argomento completamente nuovo, senza 
alcun legame con la cronologia.
Esempi: si parlava di "Angelo Marcelli" e ora si chiede "Dove si trova 
l'aula 126?". L'aula 126 NON ha relazione con Marcelli.

### Azione:
- SE CORRELATA → Integra le informazioni dalla history per risolvere 
  pronomi, riferimenti impliciti e coreferenze anaforiche.
- SE NON CORRELATA → IGNORA COMPLETAMENTE la history.
</step_a>

<step_b>
## STEP B — RISCRITTURA DELLA QUERY

### REGOLA 1 — NON GENERALIZZARE MAI:
Se l'utente chiede un concetto SPECIFICO, la query riscritta DEVE 
contenere quello STESSO concetto specifico. 
- "orari di ricevimento" NON deve diventare "informazioni su" o "Chi è?"
- "programma di Analisi Matematica 1" NON deve diventare "insegnamenti di"
- "tesi di laurea" NON deve diventare "informazioni accademiche"

### REGOLA 2 — CONSERVAZIONE ENTITÀ (INVIOLABILE):
Le seguenti entità DEVONO essere mantenute INTEGRE e LETTERALI nella 
query riscritta:
- **Nomi di docenti**: "Nicola Capuano" deve restare "Nicola Capuano"
- **Nomi di insegnamenti**: "Analisi Matematica 1" deve restare "Analisi Matematica 1"
- **Nomi di corsi di laurea**: "Ingegneria Informatica" deve restare "Ingegneria Informatica"
- **Nomi di strutture**: "aula 126", "Laboratorio ICAR"

### REGOLA 3 — ESPANSIONE, MAI COMPRESSIONE:
La query riscritta DEVE essere una frase completa e discorsiva, 
PIÙ specifica e dettagliata dell'originale.
NON ridurre MAI a semplici keyword o frammenti.
Se la query originale è già specifica, mantienila e aggiungi SOLO 
contesto di dominio.

### REGOLA 4 — CONTESTO DI DOMINIO:
Aggiungi "dipartimento DIEM" o "Università di Salerno" se non già presenti.

### REGOLA 5 — QUERY SU PERSONE:
Quando si chiede CHI È una persona:
→ Espandi verso: curriculum, qualifica accademica, ruolo, contatti, ricevimento.
Ma MANTIENI il concetto originale della domanda.

### REGOLA 6 — QUERY SU STRUTTURE FISICHE:
Quando si chiede DOVE SI TROVA un'aula, laboratorio o sede:
→ Espandi verso: ubicazione, edificio, piano, campus di Fisciano.

### REGOLA 7 — RISOLUZIONE TEMPORALE:
Se la domanda contiene riferimenti temporali relativi, risolvili con 
la data corrente.
</step_b>

<examples>
## ESEMPI DI RISCRITTURA CORRETTA E SBAGLIATA:

### Esempio 1:
- Originale: "Quali sono gli orari di ricevimento di Nicola Capuano?"
- CORRETTA: "Orari di ricevimento del docente Nicola Capuano al dipartimento DIEM dell'Università di Salerno"
- SBAGLIATA: "Chi è Nicola Capuano?" ← VIETATO! Concetto "orari di ricevimento" perso!

### Esempio 2:
- Originale: "Parlami della didattica di Analisi Matematica 1 di Vittorio Zampoli"
- CORRETTA: "Informazioni sulla didattica dell'insegnamento Analisi Matematica 1 tenuto dal docente Vittorio Zampoli al DIEM UniSA"
- SBAGLIATA: "Insegnamenti di Vittorio Zampoli" ← VIETATO! Nome insegnamento perso!

### Esempio 3:
- Originale: "Chi insegna Machine Learning a Ingegneria Informatica?"
- CORRETTA: "Docente che insegna l'insegnamento Machine Learning nel corso di Ingegneria Informatica al DIEM UniSA"
- SBAGLIATA: "Machine Learning Ingegneria Informatica" ← VIETATO! Ridotto a keyword!
</examples>

<output>
## Output
Rispondi con SOLO la query riscritta. Una frase completa e discorsiva. 
Nient'altro.
</output>"""),
        ("placeholder", "{history}"),
        ("human", "{question}"),
    ])

    MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         """<role>
## Identità
Sei un esperto di ricerca semantica specializzato nel dominio 
dell'Università degli Studi di Salerno (DIEM).
</role>

<task>
## Compito
Genera ESATTAMENTE 3 varianti semantiche della domanda fornita.
</task>

<rules>
## Regole INVIOLABILI per le varianti:

### Regola 1 — CONSERVAZIONE ENTITÀ:
Ogni variante DEVE contenere TUTTI i nomi propri presenti nella 
domanda originale (nomi di docenti, insegnamenti, corsi di laurea, 
strutture). NESSUNA entità può essere omessa o sostituita.

### Regola 2 — CONSERVAZIONE INTENTO:
Se la domanda chiede "orari di ricevimento", TUTTE le varianti devono 
riguardare "orari di ricevimento" (o sinonimi diretti come "disponibilità 
in ufficio", "quando riceve"). NON generalizzare a "informazioni su" o 
"Chi è?".

### Regola 3 — VARIAZIONE CONTROLLATA:
Varia il vocabolario e la struttura, ma PRESERVA intento e entità.
Alterna tra domande dirette e formulazioni descrittive.

### Regola 4 — FRASI COMPLETE:
Ogni variante deve essere una frase completa. MAI ridurre a keyword.

### Regola 5 — CONTESTO UNIVERSITARIO:
Mantieni riferimento a DIEM/UniSA in almeno 2 varianti su 3.
</rules>

<examples>
## Esempio:
- Domanda: "Quali sono gli orari di ricevimento di Nicola Capuano?"
- Variante 1: "Quando riceve il docente Nicola Capuano del DIEM?"
- Variante 2: "Disponibilità e orari di ricevimento del prof. Nicola Capuano all'Università di Salerno"
- Variante 3: "Orario di ricevimento studenti del professore Capuano al dipartimento DIEM UniSA"
- SBAGLIATA: "Chi è Nicola Capuano?" ← VIETATO! Intento completamente diverso!
</examples>

<output>
## Formato Output
Restituisci SOLO le 3 varianti, una per riga, senza numerazione.
</output>"""),
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

        FIX: Validazione post-rewriting per garantire che le entità
        principali (nomi propri) siano preservate nella query riscritta.
        """
        try:
            result = self._rewrite_chain.invoke({
                "question": question,
                "history": history or [],
                "current_datetime": self._get_current_datetime_str(),
            })

            # FIX: Validazione anti-compressione
            if len(result) < len(question) * 0.5:
                logger.warning(
                    f"Query rewriting sospetto (compressione): "
                    f"'{question}' → '{result}'. Uso l'originale."
                )
                return question

            if not result.strip():
                logger.warning("Query rewriting ha prodotto stringa vuota.")
                return question

            # FIX: Validazione conservazione entità
            # Estrae nomi propri dalla query originale (parole con maiuscola
            # che non sono a inizio frase) e verifica che siano nella riscritta
            original_words = set(question.split())
            # Cerca parole con lettera maiuscola (potenziali nomi propri)
            capitalized_words = {
                w.strip("?.,!;:'\"()[]") for w in original_words
                if w and w[0].isupper() and len(w) > 2
                # Escludi parole a inizio frase (posizione 0) e articoli
                and w.lower() not in {
                    "chi", "cosa", "come", "dove", "quando", "quale", "quali",
                    "parlami", "dimmi", "spiegami", "cercami", "trovami",
                    "il", "la", "lo", "le", "gli", "un", "una", "del", "della",
                    "dei", "delle", "nel", "nella", "sul", "sulla",
                    "per", "con", "tra", "fra",
                }
            }

            if capitalized_words:
                result_lower = result.lower()
                missing_entities = [
                    w for w in capitalized_words
                    if w.lower() not in result_lower
                ]
                if missing_entities:
                    logger.warning(
                        f"Query rewriting ha perso entità: {missing_entities}. "
                        f"Originale: '{question}' → Riscritta: '{result}'. "
                        f"Uso l'originale."
                    )
                    return question

            logger.info(f"Query rewritten: '{question}' → '{result}'")
            return result

        except Exception as e:
            logger.warning(f"Errore rewriting, uso query originale: {e}")
            return question

    def expand(self, question: str) -> List[str]:
        """Genera varianti semantiche della query."""
        try:
            variants = self._multi_query_chain.invoke({"question": question})
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
# CROSS-ENCODER RERANKER
# ============================================================

class CrossEncoderReranker:
    """Post-Retrieval: ri-ordina i documenti candidati con Cross-Encoder."""

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
        if not documents:
            return []

        top_n = top_n or self._top_n
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

        # ------------------------------------------------------------------
        # STAMPA DI TUTTI I CANDIDATI PRIMA DEI TAGLI (TOP_N/THRESHOLD)
        # ------------------------------------------------------------------
        print(f"\n--- CLASSIFICA COMPLETA ({len(ranked)} CANDIDATI) PRIMA DEL FILTRAGGIO ---")
        for i, (doc, score) in enumerate(ranked):
            print(f"[{i+1}] Score: {score:.4f} | Fonte: {doc.metadata.get('source_url', 'N/D')}")
            print(f"    Testo: {doc.page_content[:150]}...")
        print("-" * 65)
        # ------------------------------------------------------------------

        result = []
        for doc, score in ranked[:top_n]:
            if score < self._score_threshold:
                logger.debug(
                    f"Documento filtrato (score {score:.4f} < threshold "
                    f"{self._score_threshold}): "
                    f"{doc.metadata.get('url_originale', doc.metadata.get('source_url', 'N/D'))[:80]}"
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
# RETRIEVAL ENGINE — aggiornato per 3 Vector Store
# ============================================================

class RetrievalEngine:
    """
    Orchestratore retrieval per 3 Vector Store (audit §8).

    Flusso: rewrite → expand (multiquery) → retrieval parallelo → merge → rerank.
    """

    def __init__(self, indexer, reranker, query_optimizer=None):
        self._indexer = indexer
        self._reranker = reranker
        self._optimizer = query_optimizer

        # Costruisce retriever per ciascuna delle 3 collection
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
        Retrieval completo: rewrite → multiquery → retrieval → rerank.

        Args:
            query: La query originale dell'utente.
            collection: CollectionTarget.value (es. "persone"). Se None → all.
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
        """Retrieval cross-collection con multiquery."""
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