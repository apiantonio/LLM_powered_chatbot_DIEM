"""
Retrieval Engine: Two-Stage con gestione duale HTML/PDF.

Flusso:
  Query → [Query Optimizer] → Query riscritta
        → [HTML Retriever] → chunk diretti → [Reranker] → top-N HTML
        → [PDF  Retriever] → Parent (via ParentDocumentRetriever nativo)
        → Merge + Return contesto finale

Nota architetturale sul reranking dei PDF:
  Il ParentDocumentRetriever restituisce direttamente i Parent (3000 chars).
  Il Cross-Encoder viene applicato sui Parent — perdiamo il vantaggio del
  reranking sui Child focalizzati. Questo è il trade-off accettato per usare
  le classi native LangChain. Mitigazione: il bi-encoder interno al
  ParentDocumentRetriever già seleziona i Child più rilevanti prima della
  risalita, quindi i Parent restituiti sono quelli con i Child migliori.

KPI Impact:
  - Relevance: Query Optimizer colma il gap semantico
  - Context Precision: Reranker filtra i falsi positivi (soprattutto su HTML)
  - Context Recall: Parent chunk ampi garantiscono contesto completo per i PDF
"""

import logging
from typing import List, Optional, TYPE_CHECKING

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from config.settings import RerankerConfig

if TYPE_CHECKING:
    from ingestion.indexer import KnowledgeBaseIndexer

logger = logging.getLogger(__name__)


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
        if not history:
            return question
        try:
            return self._rewrite_chain.invoke({"question": question, "history": history})
        except Exception as e:
            logger.warning(f"Errore rewriting: {e}")
            return question
    
    def expand(self, question: str) -> List[str]:
        try:
            variants = self._multi_query_chain.invoke({"question": question})
            return [question] + variants[:3]
        except Exception as e:
            logger.warning(f"Errore multi-query: {e}")
            return [question]


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


class RetrievalEngine:
    """
    Orchestratore retrieval con gestione duale HTML/PDF.
    
    Due retriever separati vengono interrogati e i risultati
    vengono fusi (merge) e rerankati insieme.
    """
    
    def __init__(
        self,
        indexer: "KnowledgeBaseIndexer",
        reranker: CrossEncoderReranker,
        query_optimizer: Optional[QueryOptimizer] = None,
    ):
        self._indexer = indexer
        self._html_retriever = indexer.get_html_retriever()
        self._pdf_retriever = indexer.get_pdf_retriever()
        self._reranker = reranker
        self._optimizer = query_optimizer
    
    def retrieve(
        self,
        query: str,
        chat_history: Optional[list] = None,
        use_multi_query: bool = False,
    ) -> tuple[List[Document], str]:
        """
        Pipeline completa: optimize → retrieve (HTML + PDF) → rerank → return.
        """
        # Step 1: Query Optimization
        effective_query = query
        if self._optimizer and chat_history:
            effective_query = self._optimizer.rewrite(query, chat_history)
        
        # Step 2: Retrieval da entrambe le pipeline
        if use_multi_query and self._optimizer:
            queries = self._optimizer.expand(effective_query)
        else:
            queries = [effective_query]
        
        all_candidates = []
        seen = set()
        
        for q in queries:
            # HTML: chunk diretti
            html_docs = self._html_retriever.invoke(q)
            for doc in html_docs:
                h = hash(doc.page_content[:200])
                if h not in seen:
                    seen.add(h)
                    all_candidates.append(doc)
            
            # PDF: Parent già risolti dal ParentDocumentRetriever
            pdf_docs = self._pdf_retriever.invoke(q)
            for doc in pdf_docs:
                h = hash(doc.page_content[:200])
                if h not in seen:
                    seen.add(h)
                    all_candidates.append(doc)
        
        # Step 3: Reranking unificato su tutti i candidati
        final_docs = self._reranker.rerank(effective_query, all_candidates)
        
        logger.info(
            f"Retrieval: '{query[:60]}' → "
            f"{len(all_candidates)} candidati → {len(final_docs)} dopo reranking"
        )
        
        return final_docs, effective_query