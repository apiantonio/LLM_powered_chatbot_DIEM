"""
Interfacce per il livello Agent del sistema RAG DIEM.

Post-refactoring LCEL: le astrazioni custom sono state sostituite
con i tipi nativi di LangChain. Rimane solo il protocollo Guardrail
come contratto per i componenti pre/post processing.

Le classi rimosse e i rispettivi sostituti LangChain:
- LLMProvider      → langchain_core.language_models.BaseChatModel
- ScopeClassifier  → eliminato (assorbito da ScopeGuardrail)
- Guardrail (ABC)  → GuardrailProtocol (runtime_checkable Protocol)
"""

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class GuardrailProtocol(Protocol):
    """
    Protocollo per guardrail pre/post processing.
    
    Compatibile con RunnableLambda: le implementazioni possono
    essere wrappate in RunnableLambda(guardrail.check) per comporle
    in chain LCEL con l'operatore |.
    
    Returns:
        (passed: bool, text_or_rejection: str)
    """
    
    def check(self, text: str, context: Optional[dict] = None) -> tuple[bool, str]:
        ...