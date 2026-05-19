"""Motore di retrieval multi-collezione con reranking e ottimizzazione query.

Contiene il QueryOptimizer per riscrittura e espansione delle query,
il CrossEncoderReranker per il riordinamento dei risultati e il
RetrievalEngine che orchestra il flusso completo di recupero documenti.
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
    """Ottimizzatore di query con risoluzione di coreferenze ed espansione multi-query.

    Utilizza un LLM per riscrivere query contenenti riferimenti anaforici
    e per generare varianti semantiche utili al retrieval.
    """

    REWRITE_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "You are a coreference resolver for an Italian university Q&A system. "
         "You receive the last interaction (user question + assistant answer) and a new query. "
         "Replace any pronouns or implicit references (lui, lei, suo, suoi, questo, quella, "
         "lì, ci, ne, quali sono i suoi, ecc.) with the explicit entity from the last interaction. "
         "The entity can be anything: a person, a course, a classroom, a lab, a scholarship, etc. "
         "If the new query is already self-contained, return it unchanged. "
         "Do not answer the question. Output only the rewritten query.\n\n"
         "Last Q: \"Chi è il prof. Rossi?\" Last A: \"Il prof. Rossi insegna...\" → "
         "New query: \"Qual è il suo ricevimento?\" → "
         "Output: Qual è il ricevimento del prof. Rossi?\n\n"
         "Last Q: \"Parlami del corso di Informatica triennale\" Last A: \"Il corso prevede...\" → "
         "New query: \"Quali sono i suoi contenuti?\" → "
         "Output: Quali sono i contenuti del corso di Informatica triennale?\n\n"
         "Last Q: \"Dove si trova l'aula 10?\" Last A: \"L'aula 10 è nel campus...\" → "
         "New query: \"Quanti posti ha?\" → "
         "Output: Quanti posti ha l'aula 10?\n\n"
         "Last Q: \"Parlami di Ingegneria Informatica\" Last A: \"...\" → "
         "New query: \"Dove si trova l'aula 10?\" → "
         "Output: Dove si trova l'aula 10?"),
        ("human",
         "Last Q: \"{last_user_query}\"\nLast A: \"{last_assistant_answer}\"\n\n"
         "New query: \"{question}\""),
    ])

    MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "Generate exactly 3 Italian rephrasings of the given question for semantic search. "
         "Preserve all proper nouns and the original intent. "
         "Output one variant per line, no numbering, no explanations."),
        ("human", "{question}"),
    ])

    def __init__(self, llm_chat_model):
        """Inizializza l'ottimizzatore con il modello LLM fornito.

        Args:
            llm_chat_model: Modello di chat LangChain per la riscrittura e l'espansione.
        """
        self._llm = llm_chat_model

        self._rewrite_chain = self.REWRITE_PROMPT | self._llm | RunnableLambda(
            lambda msg: msg.content.strip().split('\n')[0].strip()
        )

        self._multi_query_chain = self.MULTI_QUERY_PROMPT | self._llm | RunnableLambda(
            lambda msg: [q.strip() for q in msg.content.strip().split("\n") if q.strip()]
        )

    def rewrite(
        self,
        question: str,
        last_user_query: str = "",
        last_assistant_answer: str = "",
    ) -> str:
        """Riscrive la query risolvendo coreferenze dall'ultimo turno di conversazione.

        Args:
            question: La query corrente dell'utente.
            last_user_query: La domanda dell'utente nel turno precedente.
            last_assistant_answer: La risposta dell'assistente nel turno precedente.

        Returns:
            La query riscritta con coreferenze risolte, oppure
            la query originale se non c'e' contesto o e' autosufficiente.
        """
        if not last_user_query:
            logger.info("Nessun turno precedente, query invariata: '%s'", question)
            return question

        truncated_answer = last_assistant_answer[:300]
        if len(last_assistant_answer) > 300:
            truncated_answer += "..."

        try:
            result = self._rewrite_chain.invoke({
                "question": question,
                "last_user_query": last_user_query,
                "last_assistant_answer": truncated_answer,
            })

            if not result.strip():
                logger.warning("Rewriting ha prodotto stringa vuota, uso l'originale")
                return question

            if len(result) > len(question) * 8:
                logger.warning(
                    "Rewriting sospetto (%d chars vs %d), uso l'originale",
                    len(result), len(question),
                )
                return question

            original_entities = self._extract_proper_nouns(question)
            if original_entities:
                result_lower = result.lower()
                missing = [e for e in original_entities if e.lower() not in result_lower]
                if missing:
                    logger.warning(
                        "Rewriting ha perso entita: %s, uso l'originale", missing
                    )
                    return question

            logger.info("Query rewritten: '%s' -> '%s'", question, result)
            return result

        except Exception as e:
            logger.warning("Errore rewriting, uso query originale: %s", e)
            return question

    def expand(self, question: str) -> List[str]:
        """Genera varianti semantiche della query per multi-query retrieval.

        Args:
            question: La query originale dell'utente.

        Returns:
            Lista contenente la query originale seguita dalle varianti generate.
        """
        try:
            variants = self._multi_query_chain.invoke({"question": question})
            seen: Set[str] = {question.strip().lower()}
            unique_variants = []
            for v in variants[:3]:
                v_clean = v.strip().lstrip('0123456789.-) ')
                if v_clean and v_clean.lower() not in seen:
                    seen.add(v_clean.lower())
                    unique_variants.append(v_clean)
            logger.info(
                "Multiquery expansion: %d varianti generate", len(unique_variants)
            )
            return [question] + unique_variants
        except Exception as e:
            logger.warning("Errore multi-query expansion: %s", e)
            return [question]

    @staticmethod
    def _extract_proper_nouns(text: str) -> list:
        """Estrae nomi propri dal testo basandosi sulla capitalizzazione.

        Args:
            text: Testo da cui estrarre i nomi propri.

        Returns:
            Lista di stringhe identificate come nomi propri.
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
            if (
                clean
                and clean[0].isupper()
                and len(clean) > 2
                and clean.lower() not in skip_words
            ):
                entities.append(clean)
        return entities


class CrossEncoderReranker:
    """Reranker basato su Cross-Encoder per il riordinamento dei documenti candidati.

    Utilizza un modello sentence-transformers CrossEncoder per assegnare
    uno score di rilevanza a ciascun documento rispetto alla query,
    filtrando quelli sotto la soglia configurata.
    """

    def __init__(self, config: RerankerConfig):
        """Inizializza il reranker con il modello e i parametri configurati.

        Args:
            config: Configurazione del reranker (modello, top_n, soglia).
        """
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(config.model_name)
        self._top_n = config.top_n
        self._score_threshold = config.score_treshold
        logger.info("Cross-Encoder Reranker: %s", config.model_name)

    def rerank(
        self, query: str, documents: List[Document], top_n: Optional[int] = None
    ) -> List[Document]:
        """Riordina i documenti candidati per rilevanza rispetto alla query.

        Args:
            query: Query di ricerca dell'utente.
            documents: Lista di documenti candidati da riordinare.
            top_n: Numero massimo di risultati da restituire (default da config).

        Returns:
            Lista di documenti riordinati e filtrati per soglia di rilevanza.
        """
        if not documents:
            return []

        top_n = top_n or self._top_n
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

        logger.info(
            "--- TOP 5 CLASSIFICA (su %d CANDIDATI) ---", len(ranked)
        )
        for i, (doc, score) in enumerate(ranked[:5]):
            logger.info(
                "[%d] Score: %.4f | Fonte: %s",
                i + 1, score, doc.metadata.get("source_url", "N/D"),
            )
            logger.debug(
                "    Testo: %s...", doc.page_content[:150]
            )

        result = []
        for doc, score in ranked[:top_n]:
            if score < self._score_threshold:
                logger.debug(
                    "Documento filtrato (score %.4f < threshold %.4f): %s",
                    score, self._score_threshold,
                    doc.metadata.get(
                        "url_originale", doc.metadata.get("source_url", "N/D")
                    )[:80],
                )
                continue
            doc.metadata["relevance_score"] = float(score)
            result.append(doc)

        if not result and documents:
            logger.warning(
                "Tutti i %d documenti candidati filtrati (score < %.4f). Query: '%s'",
                len(documents), self._score_threshold, query[:80],
            )

        return result


class RetrievalEngine:
    """Motore di retrieval che orchestra ricerca multi-collezione, espansione e reranking.

    Coordina il flusso completo: espansione multi-query, retrieval da
    collezioni Chroma e Parent-Child, deduplicazione e reranking finale.
    """

    def __init__(self, indexer, reranker, query_optimizer=None):
        """Inizializza il motore di retrieval con le componenti necessarie.

        Args:
            indexer: Istanza di KnowledgeBaseIndexer per accesso ai retriever.
            reranker: Istanza di CrossEncoderReranker per il riordinamento.
            query_optimizer: Istanza opzionale di QueryOptimizer per l'espansione.
        """
        self._indexer = indexer
        self._reranker = reranker
        self._optimizer = query_optimizer

        self._collection_retrievers = {}
        for target in CollectionTarget:
            self._collection_retrievers[target.value] = (
                indexer.get_collection_retriever(target)
            )
        self._pc_retriever = indexer.get_parent_child_retriever()
        self._pc_child_vectorstore = indexer._pc_child_vectorstore

    def retrieve(
        self,
        query: str,
        collection: Optional[str] = None,
        metadata_filter: Optional[dict] = None,
    ) -> tuple:
        """Esegue il retrieval con espansione multi-query e reranking.

        Args:
            query: Query di ricerca dell'utente.
            collection: Nome della collezione specifica o None per tutte.
            metadata_filter: Filtro opzionale sui metadati dei documenti.

        Returns:
            Tupla (documenti_finali, query_effettiva, multi_queries).
        """
        effective_query = query

        multi_queries = [effective_query]
        if self._optimizer:
            multi_queries = self._optimizer.expand(effective_query)

        if collection is None:
            docs, _ = self.retrieve_from_all(effective_query)
            return docs, effective_query, multi_queries

        all_candidates = []
        seen_hashes: Set[int] = set()

        for mq in multi_queries:
            if metadata_filter:
                retriever = self._get_filtered_retriever(collection, metadata_filter)
            else:
                retriever = self._collection_retrievers.get(collection)

            if retriever is None:
                logger.error("Collection sconosciuta: %s", collection)
                continue

            try:
                candidates = retriever.invoke(mq)
                for doc in candidates:
                    h = hash(doc.page_content[:200])
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        all_candidates.append(doc)
            except Exception as e:
                logger.warning(
                    "Errore retrieval multiquery '%s': %s", mq[:60], e
                )

            if collection == CollectionTarget.OFFERTA_FORMATIVA.value:
                try:
                    pc_docs = self._retrieve_from_parent_child(mq, metadata_filter)
                    for doc in pc_docs:
                        h = hash(doc.page_content[:200])
                        if h not in seen_hashes:
                            seen_hashes.add(h)
                            all_candidates.append(doc)
                except Exception as e:
                    logger.warning(
                        "Errore retrieval Parent-Child per '%s': %s", mq[:60], e
                    )

        logger.info(
            "Retrieval completato: %d candidati unici (collection=%s, filter=%s)",
            len(all_candidates), collection, metadata_filter,
        )

        final_docs = self._reranker.rerank(effective_query, all_candidates)
        return final_docs, effective_query, multi_queries

    def retrieve_from_all(
        self, query: str, metadata_filter: Optional[dict] = None
    ) -> tuple:
        """Esegue il retrieval da tutte le collezioni con reranking.

        Args:
            query: Query di ricerca dell'utente.
            metadata_filter: Filtro opzionale sui metadati dei documenti.

        Returns:
            Tupla (documenti_finali, query).
        """
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
                        filtered = self._get_filtered_retriever(
                            collection_name, metadata_filter
                        )
                        if filtered:
                            retriever = filtered

                    docs = retriever.invoke(mq)
                    for doc in docs:
                        h = hash(doc.page_content[:200])
                        if h not in seen_hashes:
                            seen_hashes.add(h)
                            all_candidates.append(doc)
                except Exception as e:
                    logger.warning(
                        "Errore retrieval da %s: %s", collection_name, e
                    )

            try:
                pc_docs = self._retrieve_from_parent_child(mq, metadata_filter)
                for doc in pc_docs:
                    h = hash(doc.page_content[:200])
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        all_candidates.append(doc)
            except Exception as e:
                logger.warning("Errore retrieval Parent-Child: %s", e)

        final_docs = self._reranker.rerank(query, all_candidates)
        return final_docs, query

    def _retrieve_from_parent_child(
        self,
        query: str,
        metadata_filter: Optional[dict] = None,
    ) -> List[Document]:
        """Esegue il retrieval dal vector store Parent-Child.

        Args:
            query: Query di ricerca.
            metadata_filter: Filtro opzionale sui metadati.

        Returns:
            Lista di documenti recuperati dal Parent-Child store.
        """
        if metadata_filter:
            chroma_where = self._build_chroma_filter(metadata_filter)
            try:
                retriever = self._pc_child_vectorstore.as_retriever(
                    search_type=self._indexer._settings.vectorstore.search_type,
                    search_kwargs={
                        "k": self._indexer._settings.vectorstore.search_k,
                        "filter": chroma_where,
                    },
                )
                docs = retriever.invoke(query)
                logger.info(
                    "Parent-Child (filtrato): %d child chunks (filter=%s)",
                    len(docs), metadata_filter,
                )
                return docs
            except Exception as e:
                logger.warning(
                    "Errore Parent-Child con filtro %s: %s. "
                    "Fallback a retrieval senza filtro.",
                    metadata_filter, e,
                )
                try:
                    return self._pc_retriever.invoke(query)
                except Exception as e2:
                    logger.warning(
                        "Anche il fallback Parent-Child e' fallito: %s", e2
                    )
                    return []
        else:
            return self._pc_retriever.invoke(query)

    def _get_filtered_retriever(self, collection_name: str, metadata_filter: dict):
        """Costruisce un retriever con filtro sui metadati per la collezione indicata.

        Args:
            collection_name: Nome della collezione target.
            metadata_filter: Dizionario di filtri sui metadati.

        Returns:
            Retriever LangChain filtrato o None se la collezione non esiste.
        """
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
        """Costruisce un filtro Chroma a partire da un dizionario di metadati.

        Args:
            metadata_filter: Dizionario chiave-valore dei filtri desiderati.

        Returns:
            Dizionario nel formato atteso da Chroma per il filtraggio.
        """
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