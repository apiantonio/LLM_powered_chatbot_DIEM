"""
Retrieval Engine: Two-Stage Retrieval con Query Optimization e Re-ranking.

Architettura:
  Query utente → [Query Optimizer] → Query riscritta
                                    → [Bi-Encoder Retriever] → Top-K candidati
                                    → [Cross-Encoder Reranker] → Top-N finali
                                    → Contesto per il LLM

Pattern: Chain of Responsibility (ogni stage processa e passa al successivo).

KPI Impact:
- Query Optimizer → Relevance (colma il gap semantico utente↔documento)
- Two-Stage Retrieval → Context Precision (elimina falsi positivi)
- Reranker → Context Recall (preserva documenti sottilmente rilevanti)
"""

import logging
from typing import List, Optional

from sentence_transformers import CrossEncoder

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from config.settings import AppSettings, RerankerConfig

logger = logging.getLogger(__name__)


class QueryOptimizer:
    """
    Pre-Retrieval: riscrittura della query per massimizzare la qualità del retrieval.
    
    Implementa due strategie:
    1. Conversational Rewriting: risolve coreferenze anaforiche (lui/quello/ecc.)
    2. Multi-Query: genera N varianti per ampliare la copertura semantica.
    
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
        """
        Args:
            llm_chat_model: Modello LangChain chat (es. ChatHuggingFace, ChatOllama).
        """
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
            # Aggiungi sempre la query originale
            all_queries = [question] + variants[:3]
            logger.debug(f"Multi-query: {len(all_queries)} varianti generate")
            return all_queries
        except Exception as e:
            logger.warning(f"Errore multi-query, uso solo query originale: {e}")
            return [question]


class CrossEncoderReranker:
    """
    Post-Retrieval: ri-ordina i documenti candidati con un Cross-Encoder.
    
    Il Cross-Encoder riceve (query, documento) come coppia e produce
    un punteggio di rilevanza tramite cross-attention, catturando
    interazioni lessicali impossibili per il bi-encoder.
    
    KPI Impact: Context Precision (elimina falsi positivi),
    Faithfulness (contesto più pulito → meno allucinazioni).
    """
    
    def __init__(self, config: RerankerConfig):
        self._model = CrossEncoder(config.model_name)
        self._top_n = config.top_n
        logger.info(f"Cross-Encoder Reranker inizializzato: {config.model_name}")
    
    def rerank(self, query: str, documents: List[Document], top_n: Optional[int] = None) -> List[Document]:
        """
        Ri-ordina i documenti per rilevanza effettiva rispetto alla query.
        
        Args:
            query: Query dell'utente (possibilmente già riscritta).
            documents: Lista di documenti candidati dal bi-encoder.
            top_n: Override del numero di documenti da restituire.
            
        Returns:
            Lista ordinata dei top_n documenti più rilevanti,
            con score aggiunto nei metadata.
        """
        if not documents:
            return []
        
        top_n = top_n or self._top_n
        
        # Prepara le coppie (query, document_text) per il cross-encoder
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self._model.predict(pairs)
        
        # Ordina per score decrescente
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        
        result = []
        for doc, score in ranked[:top_n]:
            doc.metadata["relevance_score"] = float(score)
            result.append(doc)
        
        logger.debug(
            f"Reranking: {len(documents)} candidati → {len(result)} selezionati "
            f"(score range: {ranked[0][1]:.4f} → {ranked[-1][1]:.4f})"
        )
        
        return result


class RetrievalEngine:
    """
    Orchestratore del flusso di retrieval completo.
    
    Combina: Query Optimization → Bi-Encoder Search → Cross-Encoder Reranking.
    
    È il componente centrale che alimenta il contesto dell'agente RAG.
    """
    
    def __init__(
        self,
        retriever,  # LangChain VectorStoreRetriever
        reranker: CrossEncoderReranker,
        query_optimizer: Optional[QueryOptimizer] = None,
    ):
        self._retriever = retriever
        self._reranker = reranker
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
            (documenti_rilevanti, query_effettivamente_usata)
        """
        # Step 1: Query Optimization
        effective_query = query
        if self._optimizer and chat_history:
            effective_query = self._optimizer.rewrite(query, chat_history)
        
        # Step 2: Retrieval (singolo o multi-query)
        if use_multi_query and self._optimizer:
            queries = self._optimizer.expand(effective_query)
            all_docs = []
            seen_contents = set()
            for q in queries:
                docs = self._retriever.invoke(q)
                for doc in docs:
                    # Deduplicazione per contenuto
                    content_hash = hash(doc.page_content[:200])
                    if content_hash not in seen_contents:
                        seen_contents.add(content_hash)
                        all_docs.append(doc)
            candidates = all_docs
        else:
            candidates = self._retriever.invoke(effective_query)
        
        # Step 3: Re-ranking
        final_docs = self._reranker.rerank(effective_query, candidates)
        
        logger.info(
            f"Retrieval completato: '{query}' → {len(candidates)} candidati → "
            f"{len(final_docs)} dopo reranking"
        )
        
        return final_docs, effective_query
