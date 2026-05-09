"""
retrieval/engine.py — Retrieval Engine multi-collection.

Flusso:
  Query → [Query Optimizer] → Query riscritta
        → [Collection Retriever] → chunk → [Reranker] → top-N
  oppure
  Query → [All Retrievers] → merge → [Reranker] → top-N

Coerenza con gli altri moduli:
  - ingestion/indexer.py: usa get_collection_retriever(CollectionTarget) e
                          get_parent_child_retriever() dall'Indexer.
  - ingestion/router.py: CollectionTarget.value viene usato come chiave
                          nel dict _collection_retrievers.
  - agent/tools/__init__.py: ogni tool chiama retrieve(collection=CollectionTarget.value)
                              o retrieve_from_all().

NOTA: QueryOptimizer e CrossEncoderReranker sono definiti in questo modulo
perché agent_main.py li importa da qui:
  from retrieval.engine import RetrievalEngine, QueryOptimizer, CrossEncoderReranker
"""

import logging
from typing import List, Optional, TYPE_CHECKING

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from config.settings import RerankerConfig
from ingestion.router import CollectionTarget

if TYPE_CHECKING:
    from ingestion.indexer import KnowledgeBaseIndexer

logger = logging.getLogger(__name__)


# ============================================================
# QUERY OPTIMIZER (Pre-Retrieval)
# ============================================================

class QueryOptimizer:
    """
    Pre-Retrieval: riscrittura e espansione query.
    
    1. Conversational Rewriting: risolve coreferenze anaforiche.
    2. Multi-Query: genera varianti per copertura semantica.
    """
    
    REWRITE_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "Sei un ottimizzatore di query di ricerca. "
         "Riscrivi la domanda dell'utente in una query autonoma e specifica "
         "per la ricerca semantica in un database vettoriale universitario. "
         "Risolvi pronomi e riferimenti impliciti usando la cronologia. "
         "Usa terminologia tecnica e accademica dove appropriato. "
         "Restituisci SOLO la query riscritta, nient'altro."),
        ("placeholder", "{history}"),
        ("human", "{question}"),
    ])
    
    MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "Sei un ottimizzatore di ricerca. Genera esattamente 3 varianti diverse "
         "della seguente domanda, ciascuna che esplori un angolo diverso dell'argomento. "
         "Restituisci solo le 3 query, una per riga, senza numerazione."),
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
        """Riscrivi la query risolvendo coreferenze dalla cronologia."""
        if not history:
            return question
        try:
            return self._rewrite_chain.invoke(
                {"question": question, "history": history}
            )
        except Exception as e:
            logger.warning(f"Errore rewriting: {e}")
            return question
    
    def expand(self, question: str) -> List[str]:
        """Genera varianti della query per copertura semantica."""
        try:
            variants = self._multi_query_chain.invoke({"question": question})
            return [question] + variants[:3]
        except Exception as e:
            logger.warning(f"Errore multi-query: {e}")
            return [question]


# ============================================================
# CROSS-ENCODER RERANKER (Post-Retrieval)
# ============================================================

class CrossEncoderReranker:
    """Post-Retrieval: ri-ordina i documenti candidati con Cross-Encoder."""
    
    def __init__(self, config: RerankerConfig):
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(config.model_name)
        self._top_n = config.top_n
        logger.info(f"Cross-Encoder Reranker: {config.model_name}")
    
    def rerank(
        self, query: str, documents: List[Document], top_n: Optional[int] = None
    ) -> List[Document]:
        """Ri-ordina i documenti per rilevanza rispetto alla query."""
        if not documents:
            return []
        
        top_n = top_n or self._top_n
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        
        result = []
        for doc, score in ranked[:top_n]:
            doc.metadata["relevance_score"] = float(score)
            result.append(doc)
        return result


# ============================================================
# RETRIEVAL ENGINE
# ============================================================

class RetrievalEngine:
    """
    Orchestratore retrieval multi-collection.
    
    Supporta:
    1. retrieve(query, collection) → ricerca in una singola collection
    2. retrieve_from_all(query) → ricerca cross-collection con merge + rerank
    
    Il parametro `collection` accetta CollectionTarget.value (stringa),
    coerente con quanto passato dai tool dell'agente.
    """

    def __init__(
        self,
        indexer: "KnowledgeBaseIndexer",
        reranker: CrossEncoderReranker,
        query_optimizer: Optional[QueryOptimizer] = None,
    ):
        self._indexer = indexer
        self._reranker = reranker
        self._optimizer = query_optimizer

        # Cache dei retriever: chiave = CollectionTarget.value (stringa)
        self._collection_retrievers = {}
        for target in CollectionTarget:
            self._collection_retrievers[target.value] = (
                indexer.get_collection_retriever(target)
            )
        self._pc_retriever = indexer.get_parent_child_retriever()
        
        logger.info(
            f"RetrievalEngine inizializzato con "
            f"{len(self._collection_retrievers)} collection retrievers "
            f"+ 1 Parent-Child retriever"
        )

    def retrieve(
        self,
        query: str,
        collection: Optional[str] = None,
        chat_history: Optional[list] = None,
    ) -> tuple[List[Document], str]:
        """
        Retrieval da una singola collection (o tutte se collection=None).
        
        Args:
            query: La query di ricerca.
            collection: CollectionTarget.value (stringa). Se None, usa retrieve_from_all.
            chat_history: Storico conversazionale per query rewriting.
        
        Returns:
            Tupla (documenti_rerankati, query_effettiva).
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
        # risultati Parent-Child (regolamenti/piani di studio)
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

        # Interroga tutte le 4 collection
        for name, retriever in self._collection_retrievers.items():
            docs = retriever.invoke(effective_query)
            for doc in docs:
                h = hash(doc.page_content[:200])
                if h not in seen:
                    seen.add(h)
                    all_candidates.append(doc)

        # Aggiungi Parent-Child (regolamenti/piani)
        pc_docs = self._pc_retriever.invoke(effective_query)
        for doc in pc_docs:
            h = hash(doc.page_content[:200])
            if h not in seen:
                seen.add(h)
                all_candidates.append(doc)

        # Reranking unificato
        final_docs = self._reranker.rerank(effective_query, all_candidates)

        logger.info(
            f"Retrieval [ALL]: '{query[:50]}' → "
            f"{len(all_candidates)} candidati → {len(final_docs)} dopo reranking"
        )
        return final_docs, effective_query