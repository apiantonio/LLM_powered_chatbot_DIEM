"""
retrieval/engine.py — Retrieval Engine multi-collection.

Aggiunge il supporto per retrieval collection-specifico
e retrieval trasversale con merge e reranking unificato.
"""

import logging
from typing import List, Optional, Dict, TYPE_CHECKING

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from config.settings import RerankerConfig
from ingestion.router import CollectionTarget

if TYPE_CHECKING:
    from ingestion.indexer import KnowledgeBaseIndexer

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """
    Orchestratore retrieval multi-collection.
    
    Supporta:
    1. retrieve(query, collection) → ricerca in una singola collection
    2. retrieve_from_all(query) → ricerca cross-collection con merge
    """

    def __init__(
        self,
        indexer: "KnowledgeBaseIndexer",
        reranker: "CrossEncoderReranker",
        query_optimizer: Optional["QueryOptimizer"] = None,
    ):
        self._indexer = indexer
        self._reranker = reranker
        self._optimizer = query_optimizer

        # Cache dei retriever per collection
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
        chat_history: Optional[list] = None,
    ) -> tuple[List[Document], str]:
        """
        Retrieval da una singola collection (o tutte se collection=None).
        """
        # Query Optimization
        effective_query = query
        if self._optimizer and chat_history:
            effective_query = self._optimizer.rewrite(query, chat_history)

        if collection is None:
            return self.retrieve_from_all(effective_query)

        # Retrieval dalla collection specifica
        retriever = self._collection_retrievers.get(collection)
        if retriever is None:
            logger.error(f"Collection sconosciuta: {collection}")
            return [], effective_query

        candidates = retriever.invoke(effective_query)

        # Se la collection è offerta_formativa, aggiungi anche i
        # risultati Parent-Child (regolamenti/piani)
        if collection == CollectionTarget.OFFERTA_FORMATIVA.value:
            pc_docs = self._pc_retriever.invoke(effective_query)
            seen = {hash(d.page_content[:200]) for d in candidates}
            for doc in pc_docs:
                h = hash(doc.page_content[:200])
                if h not in seen:
                    seen.add(h)
                    candidates.append(doc)

        # Reranking
        final_docs = self._reranker.rerank(effective_query, candidates)

        logger.info(
            f"Retrieval [{collection}]: '{query[:50]}' → "
            f"{len(candidates)} candidati → {len(final_docs)} dopo reranking"
        )
        return final_docs, effective_query

    def retrieve_from_all(
        self,
        query: str,
        chat_history: Optional[list] = None,
    ) -> tuple[List[Document], str]:
        """
        Retrieval cross-collection: interroga tutte le collection,
        fa merge dei risultati e applica reranking unificato.
        """
        effective_query = query
        if self._optimizer and chat_history:
            effective_query = self._optimizer.rewrite(query, chat_history)

        all_candidates = []
        seen = set()

        for name, retriever in self._collection_retrievers.items():
            docs = retriever.invoke(effective_query)
            for doc in docs:
                h = hash(doc.page_content[:200])
                if h not in seen:
                    seen.add(h)
                    all_candidates.append(doc)

        # Aggiungi Parent-Child
        pc_docs = self._pc_retriever.invoke(effective_query)
        for doc in pc_docs:
            h = hash(doc.page_content[:200])
            if h not in seen:
                seen.add(h)
                all_candidates.append(doc)

        final_docs = self._reranker.rerank(effective_query, all_candidates)

        logger.info(
            f"Retrieval [ALL]: '{query[:50]}' → "
            f"{len(all_candidates)} candidati → {len(final_docs)} dopo reranking"
        )
        return final_docs, effective_query