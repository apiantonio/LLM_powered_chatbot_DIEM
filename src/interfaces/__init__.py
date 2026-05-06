"""
Interfacce astratte (Ports) del sistema RAG.

Design Pattern: Strategy (GoF) + Dependency Inversion (SOLID-D).
Ogni componente dipende da queste astrazioni, mai dalle implementazioni concrete.
Questo permette di:
- Cambiare LLM provider senza toccare la logica dell'agente.
- Sostituire il Vector Store senza modificare il retrieval engine.
- Testare ogni modulo in isolamento con mock/stub.

KPI Impact: Ingegneria del Software (modularità, testabilità, estendibilità).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Document:
    """Rappresentazione unificata di un documento nel sistema."""
    page_content: str
    metadata: dict


@dataclass
class RetrievalResult:
    """Risultato di una query al retrieval engine, con tracciabilità delle fonti."""
    documents: List[Document]
    query_used: str  # La query effettivamente usata (potrebbe essere riscritta)


class LLMProvider(ABC):
    """
    Strategy Interface per il provider LLM.
    
    Permette di intercambiare Ollama, HuggingFace, OpenAI
    senza modificare il codice dell'Agent o della RAG chain.
    """
    
    @abstractmethod
    def invoke(self, prompt: str) -> str:
        """Genera una risposta testuale dato un prompt."""
        ...
    
    @abstractmethod
    def invoke_with_messages(self, messages: List[dict]) -> str:
        """Genera una risposta dato un array di messaggi strutturati."""
        ...
    
    @abstractmethod
    def supports_tool_calling(self) -> bool:
        """Indica se il provider supporta nativamente il tool calling."""
        ...


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


class ScopeClassifier(ABC):
    """
    Interface per il classificatore Out-of-Domain.
    
    KPI Impact: Scope Awareness. Determina se una query appartiene
    al dominio DIEM oppure è fuori contesto (e va bloccata).
    """
    
    @abstractmethod
    def is_in_scope(self, query: str) -> bool:
        """Restituisce True se la query è nel dominio consentito."""
        ...
    
    @abstractmethod
    def get_rejection_message(self) -> str:
        """Messaggio standard di rifiuto per query OOD."""
        ...


class Guardrail(ABC):
    """
    Interface generica per guardrail pre/post processing.
    
    Implementabile come:
    - InputSanitizer (before_agent): pulizia injection
    - OutputValidator (after_agent): anti-hallucination check
    - ScopeGuardrail: verifica dominio
    """
    
    @abstractmethod
    def check(self, text: str, context: Optional[dict] = None) -> tuple[bool, str]:
        """
        Verifica il testo e restituisce:
        - (True, testo_originale_o_modificato) se OK
        - (False, messaggio_di_rifiuto) se bloccato
        """
        ...
