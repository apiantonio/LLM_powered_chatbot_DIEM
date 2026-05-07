"""
Retrieval Engine: Two-Stage Retrieval con Parent-Child resolution.

Flusso completo:
  Query utente
    → [Query Optimizer] → Query riscritta/espansa
    → [Bi-Encoder Retriever] → Top-K Child candidati (Chroma)
    → [Cross-Encoder Reranker] → Top-N Child filtrati
    → [Parent Resolution] → Child HTML invariati + Child PDF → Parent risaliti
    → Contesto finale per il LLM

Il Parent-Child resolution è trasparente: il RetrievalEngine riceve Child dal
Chroma, ma prima di restituirli li passa attraverso l'Indexer.resolve_parents().
I Child PDF vengono sostituiti dai loro Parent (3000 chars), i Child HTML
passano invariati.

Pattern: Chain of Responsibility + Mediator (l'engine media tra retriever, reranker e indexer).

KPI Impact:
- Query Optimizer → Relevance
- Two-Stage Retrieval → Context Precision 
- Parent Resolution → Context Recall (il LLM riceve sezioni complete, non frammenti)
- Reranker → Faithfulness (contesto pulito → meno allucinazioni)
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
    Pre-Retrieval: riscrittura della query per massimizzare la qualità del retrieval.
    
    Due strategie:
    1. Conversational Rewriting: risolve coreferenze ("quello" → "il regolamento tesi magistrale").
    2. Multi-Query: genera N varianti per copertura semantica più ampia.
    
    KPI Impact: Relevance, Context Recall.
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
        """Riscrive la query risolvendo coreferenze dalla chat history."""
        if not history:
            return question
        try:
            rewritten = self._rewrite_chain.invoke({
                "question": question,
                "history": history,
            })
            logger.debug(f"Query riscritta: '{question}' → '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.warning(f"Errore rewriting, uso query originale: {e}")
            return question
    
    def expand(self, question: str) -> List[str]:
        """Genera varianti multi-query per ampliare la copertura."""
        try:
            variants = self._multi_query_chain.invoke({"question": question})
            all_queries = [question] + variants[:3]
            logger.debug(f"Multi-query: {len(all_queries)} varianti generate")
            return all_queries
        except Exception as e:
            logger.warning(f"Errore multi-query, uso solo query originale: {e}")
            return [question]


class CrossEncoderReranker:
    """
    Post-Retrieval: ri-ordina i Child candidati con un Cross-Encoder.
    
    NOTA: Il reranking avviene sui CHILD (testo corto, focalizzato).
    La risalita ai Parent avviene DOPO il reranking, così il cross-encoder
    valuta la rilevanza sul frammento preciso, non sul blocco diluito.
    
    KPI Impact: Context Precision, Faithfulness.
    """
    
    def __init__(self, config: RerankerConfig):
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(config.model_name)
        self._top_n = config.top_n
        logger.info(f"Cross-Encoder Reranker: {config.model_name}")
    
    def rerank(
        self, query: str, documents: List[Document], top_n: Optional[int] = None
    ) -> List[Document]:
        """Ri-ordina i documenti per rilevanza. Score nei metadata."""
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
        
        logger.debug(
            f"Reranking: {len(documents)} → {len(result)} "
            f"(best: {ranked[0][1]:.4f}, worst kept: {ranked[min(top_n-1, len(ranked)-1)][1]:.4f})"
        )
        return result


class RetrievalEngine:
    """
    Orchestratore del flusso di retrieval completo con Parent-Child resolution.
    
    Sequenza:
    1. Query Optimization (rewrite/expand)
    2. Bi-Encoder search su Chroma (restituisce Child)
    3. Cross-Encoder reranking (sui Child — embedding focalizzato)
    4. Parent Resolution (Child PDF → Parent, Child HTML → invariato)
    5. Return contesto finale per il LLM
    """
    
    def __init__(
        self,
        retriever,
        reranker: CrossEncoderReranker,
        indexer: "KnowledgeBaseIndexer",
        query_optimizer: Optional[QueryOptimizer] = None,
    ):
        self._retriever = retriever
        self._reranker = reranker
        self._indexer = indexer  # Necessario per resolve_parents()
        self._optimizer = query_optimizer
    
    def retrieve(
        self,
        query: str,
        chat_history: Optional[list] = None,
        use_multi_query: bool = False,
    ) -> tuple[List[Document], str]:
        """
        Pipeline completa di retrieval.
        
        Returns:
            (documenti_con_contesto_completo, query_effettivamente_usata)
        """
        # Step 1: Query Optimization
        effective_query = query
        if self._optimizer and chat_history:
            effective_query = self._optimizer.rewrite(query, chat_history)
        
        # Step 2: Retrieval (Child chunks da Chroma)
        if use_multi_query and self._optimizer:
            queries = self._optimizer.expand(effective_query)
            all_docs = []
            seen_contents = set()
            for q in queries:
                docs = self._retriever.invoke(q)
                for doc in docs:
                    content_hash = hash(doc.page_content[:200])
                    if content_hash not in seen_contents:
                        seen_contents.add(content_hash)
                        all_docs.append(doc)
            candidates = all_docs
        else:
            candidates = self._retriever.invoke(effective_query)
        
        # Step 3: Reranking (sui Child — testo focalizzato)
        reranked = self._reranker.rerank(effective_query, candidates)
        
        # Step 4: Parent Resolution
        # Child PDF → Parent (3000 chars con contesto completo)
        # Child HTML → passa invariato
        final_docs = self._indexer.resolve_parents(reranked)
        
        logger.info(
            f"Retrieval: '{query[:60]}...' → "
            f"{len(candidates)} candidati → {len(reranked)} reranked → "
            f"{len(final_docs)} dopo parent resolution"
        )
        
        return final_docs, effective_query
