from abc import ABC, abstractmethod
from typing import List, Optional


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