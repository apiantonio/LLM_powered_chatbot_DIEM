"""
retrieval/engine.py — RetrievalEngine RIVISTO per 3 Vector Store.

REFACTORING v2:
  - QueryOptimizer.rewrite(): prompt RISCRITTO DA ZERO.
    * Unico compito: risolvere coreferenze (pronomi → nomi propri dalla history)
    * ZERO logiche hardcoded (if/else per pronomi ELIMINATI)
    * NON risponde alla domanda, NON aggiunge info dalla risposta precedente
    * Se non ci sono riferimenti da svelare, mantiene la query identica
  - MULTI_QUERY_PROMPT: semplificato, stesse regole di conservazione entità
  - Iniezione temporale dinamica: middleware pre-agente (vedi agent.py)
  - CrossEncoderReranker: invariato
  - RetrievalEngine: invariato

FLUSSO:
  rewrite → expand (multiquery) → retrieval parallelo → merge → rerank
"""

import logging
from typing import List, Optional, Set

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from config.settings import RerankerConfig
from ingestion.router import CollectionTarget

logger = logging.getLogger(__name__)


class QueryOptimizer:
    """
    Pre-Retrieval: Query Rewriting minimale + Multiquery Expansion.

    REWRITE (v2 — riscritto da zero):
      Unico compito: prendere la query dell'utente e sostituire pronomi
      o riferimenti impliciti con i nomi propri estratti dalla history.
      NON rispondere alla domanda.
      NON aggiungere informazioni dalla risposta precedente.
      Se non ci sono riferimenti da svelare, restituire la query IDENTICA.

    EXPAND:
      Genera 3 varianti semantiche preservando entità e intento.
    """

    REWRITE_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         """Sei un risolutore di coreferenze. Il tuo UNICO compito:

REGOLE:
1. Se la query contiene pronomi o riferimenti impliciti (lui, lei, questo corso, quali corsi insegna, ecc.) che si riferiscono a entità nominate nella history, SOSTITUISCI quei pronomi con i nomi propri corrispondenti.
2. Se la query è autosufficiente (non ha pronomi da risolvere), restituiscila IDENTICA.
3. NON rispondere MAI alla domanda.
4. NON aggiungere MAI informazioni prese dalle risposte precedenti.
5. NON espandere, arricchire o riformulare la query oltre la sostituzione dei pronomi.
6. Output: SOLO la query risultante, nient'altro.

ESEMPI:
- History: "Chi è X?" / Query: "Quali corsi insegna?"
  → "Quali corsi insegna X?"

- History: "Parlami di Ingegneria Informatica" / Query: "Quali sono i requisiti?"
  → "Quali sono i requisiti di Ingegneria Informatica?"

- History: "Chi è X?" / Query: "Dove si trova l'aula Y?"
  → "Dove si trova l'aula Y?"
  (Nessun pronome da risolvere, query restituita identica)

- Query senza history: "Quali laboratori ha il DIEM?"
  → "Quali laboratori ha il DIEM?"
  (Nessuna history, query restituita identica)"""),
        ("placeholder", "{history}"),
        ("human", "{question}"),
    ])

    MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         """Genera ESATTAMENTE 3 varianti della domanda fornita per la ricerca semantica.

REGOLE:
1. Ogni variante DEVE contenere TUTTI i nomi propri della domanda originale.
2. Ogni variante DEVE mantenere lo STESSO intento (se chiede "ricevimento", tutte chiedono "ricevimento").
3. Varia vocabolario e struttura, ma conserva entità e intento.
4. Frasi complete, MAI keyword isolate.

Output: 3 varianti, una per riga, senza numerazione."""),
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

    def rewrite(self, question: str, history: Optional[list] = None) -> str:
        """
        Riscrivi la query risolvendo SOLO coreferenze.

        Se non c'è history o non ci sono pronomi da risolvere,
        restituisce la query originale.
        """
        # Se non c'è history, non c'è nulla da risolvere
        if not history or len(history) == 0:
            logger.info(f"Nessuna history, query invariata: '{question}'")
            return question

        try:
            result = self._rewrite_chain.invoke({
                "question": question,
                "history": history,
            })

            # Sanity check: il modello potrebbe aver generato una risposta
            # invece di una query. Segnali di allucinazione:
            if any(marker in result for marker in [
                "**", "1.", "2.", "3.", "Ecco", "In base", "Basandomi",
                "La risposta", "Il docente", "Il corso",
            ]):
                logger.warning(
                    f"Rewriting ha prodotto una risposta invece di una query. "
                    f"Originale: '{question}' → Riscritta: '{result[:100]}...'. "
                    f"Uso l'originale."
                )
                return question

            # Se il risultato è troppo lungo rispetto all'originale (>3x),
            # probabilmente il modello ha risposto alla domanda
            if len(result) > len(question) * 3:
                logger.warning(
                    f"Rewriting troppo lungo ({len(result)} vs {len(question)} chars). "
                    f"Uso l'originale."
                )
                return question

            # Se il risultato è vuoto
            if not result.strip():
                logger.warning("Rewriting ha prodotto stringa vuota.")
                return question

            # Validazione conservazione entità: i nomi propri della query
            # originale devono essere presenti nella riscritta
            original_entities = self._extract_proper_nouns(question)
            if original_entities:
                result_lower = result.lower()
                missing = [e for e in original_entities if e.lower() not in result_lower]
                if missing:
                    logger.warning(
                        f"Rewriting ha perso entità: {missing}. Uso l'originale."
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

    @staticmethod
    def _extract_proper_nouns(text: str) -> list:
        """
        Estrae probabili nomi propri da un testo.
        Cerca parole con maiuscola iniziale che non siano a inizio frase
        o parole funzionali italiane.
        """
        skip_words = {
            "chi", "cosa", "come", "dove", "quando", "quale", "quali",
            "parlami", "dimmi", "spiegami", "cercami", "trovami",
            "il", "la", "lo", "le", "gli", "un", "una", "del", "della",
            "dei", "delle", "nel", "nella", "sul", "sulla",
            "per", "con", "tra", "fra", "che", "non", "sono",
            "qual", "quanto", "quanti", "quante",
        }
        words = text.split()
        entities = []
        for w in words:
            clean = w.strip("?.,!;:'\"()[]")
            if clean and clean[0].isupper() and len(clean) > 2 and clean.lower() not in skip_words:
                entities.append(clean)
        return entities


# ============================================================
# CROSS-ENCODER RERANKER
# ============================================================

class CrossEncoderReranker:
    """Post-Retrieval: ri-ordina i documenti candidati con Cross-Encoder."""

    def __init__(self, config: RerankerConfig):
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(config.model_name)
        self._top_n = config.top_n
        self._score_threshold = config.score_treshold
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

        # Stampa classifica per debug
        print(f"\n--- TOP 5 CLASSIFICA (su {len(ranked)} CANDIDATI) ---")
        for i, (doc, score) in enumerate(ranked[:5]):  # <-- Modificato qui: aggiunto [:5]
            print(f"[{i+1}] Score: {score:.4f} | Fonte: {doc.metadata.get('source_url', 'N/D')}")
            print(f"    Testo: {doc.page_content[:150]}...")
        print("-" * 65)

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

    def retrieve_from_all(self, query: str, metadata_filter: Optional[dict] = None) -> tuple:
        """Retrieval cross-collection con multiquery e filtro opzionale."""
        multi_queries = [query]
        if self._optimizer:
            multi_queries = self._optimizer.expand(query)

        all_candidates = []
        seen_hashes: Set[int] = set()

        for mq in multi_queries:
            for collection_name, default_retriever in self._collection_retrievers.items():
                try:
                    retriever = default_retriever
                    if metadata_filter:
                        filtered = self._get_filtered_retriever(collection_name, metadata_filter)
                        if filtered:
                            retriever = filtered
                            
                    docs = retriever.invoke(mq)
                    for doc in docs:
                        h = hash(doc.page_content[:200])
                        if h not in seen_hashes:
                            seen_hashes.add(h)
                            all_candidates.append(doc)
                except Exception as e:
                    logger.warning(f"Errore retrieval da {collection_name}: {e}")

            try:
                if not metadata_filter:
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