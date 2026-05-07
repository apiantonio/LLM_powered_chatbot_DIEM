from abc import ABC, abstractmethod
from interfaces.rag_interfaces import Document
from typing import List

class EmbeddingProvider(ABC):
    """Strategy Interface per il modello di embedding."""
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Genera l'embedding di una singola query."""
        ...
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Genera gli embeddings per un batch di documenti."""
        ...


class VectorStorePort(ABC):
    """
    Port Interface per il Vector Store (Hexagonal Architecture).
    
    Disaccoppia il retrieval engine dall'implementazione specifica
    del database vettoriale (Chroma, FAISS, Qdrant, Pinecone).
    """
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Indicizza una lista di documenti."""
        ...
    
    @abstractmethod
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Esegue una ricerca per similarità semantica."""
        ...
    
    @abstractmethod
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """Restituisce documenti con punteggio di similarità."""
        
        ...
