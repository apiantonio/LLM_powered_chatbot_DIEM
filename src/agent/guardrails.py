"""
Guardrails LCEL-nativi per il sistema RAG DIEM.

Post-refactoring: ogni guardrail è un componente LCEL autonomo.
Può essere usato:
  1. Come oggetto standalone: guardrail.check(text)
  2. Come RunnableLambda in una chain LCEL: guardrail.as_runnable() | ...

Tre livelli di protezione:
  1. ScopeGuardrail (before_agent): Blocca domande fuori dominio via LLM.
  2. InputSanitizer (before_model): Rileva prompt injection (deterministico).
  3. OutputValidator (after_agent): Verifica output e blocca PII.

KPI Impact:
  - Scope Awareness: ScopeGuardrail → rilevamento OOD.
  - Robustness: InputSanitizer → resistenza a prompt injection.
  - Correctness: OutputValidator → zero allucinazioni in output.
"""

import re
import logging
from typing import Optional

from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


# ============================================================
# SCOPE GUARDRAIL (Pre-Agent)
# ============================================================

class ScopeGuardrail:
    """
    Classificatore Out-of-Domain basato su LLM, LCEL-nativo.
    
    Usa un ChatPromptTemplate + ChatModel per classificare la query
    come IN_SCOPE o OUT_OF_SCOPE prima che il retrieval parta.
    """
    
    _CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "Sei un classificatore di dominio. Il tuo unico compito è determinare se "
         "la seguente domanda riguarda il Dipartimento DIEM dell'Università degli "
         "Studi di Salerno (corsi, docenti, esami, orari, regolamenti, tesi, "
         "borse di studio, laboratori, servizi, dottorato).\n\n"
         "Rispondi SOLO con 'IN_SCOPE' o 'OUT_OF_SCOPE'. Nient'altro."),
        ("human", "{query}"),
    ])
    
    REJECTION_MSG = (
        "Mi dispiace, posso rispondere solo a domande relative al Dipartimento "
        "DIEM dell'Università degli Studi di Salerno (corsi, docenti, esami, "
        "orari, regolamenti, servizi dipartimentali). "
        "La tua domanda sembra riguardare un altro argomento."
    )
    
    def __init__(self, chat_model: BaseChatModel):
        """
        Args:
            chat_model: Un BaseChatModel LangChain (es. ChatHuggingFace).
                        Sostituisce il vecchio 'llm_provider' custom.
        """
        self._chain = (
            self._CLASSIFICATION_PROMPT
            | chat_model
            | RunnableLambda(lambda msg: msg.content.strip().upper())
        )
    
    def check(self, text: str, context: Optional[dict] = None) -> tuple[bool, str]:
        try:
            response = self._chain.invoke({"query": text})
            
            if "OUT_OF_SCOPE" in response:
                logger.info(f"Query OOD bloccata: '{text[:80]}...'")
                return False, self.REJECTION_MSG
            
            return True, text
            
        except Exception as e:
            logger.warning(f"Errore scope classification, fail-open: {e}")
            return True, text
    
    def as_runnable(self) -> RunnableLambda:
        """Restituisce un Runnable LCEL per inserimento in chain."""
        def _scope_check(input_text: str) -> str:
            passed, result = self.check(input_text)
            if not passed:
                raise ScopeViolationError(result)
            return result
        return RunnableLambda(_scope_check)


class ScopeViolationError(Exception):
    """Eccezione per query fuori dominio. Catturabile dal chiamante."""
    pass


# ============================================================
# INPUT SANITIZER (Pre-Model)
# ============================================================

class InputSanitizer:
    """
    Guardrail deterministico per sanitizzare l'input dell'utente.
    Puramente regex-based, nessuna dipendenza da LLM.
    
    Rileva e neutralizza:
      1. Tentativi di prompt injection.
      2. Tentativi di manipolazione ("Are you sure?" / "Ignore instructions").
      3. Tentativi di estrazione del system prompt.
    """
    
    _INJECTION_PATTERNS = [
        r"ignor[ae]\s+(le\s+)?istruzioni\s+preced",
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"forget\s+(all\s+)?(your\s+)?instructions",
        r"you\s+are\s+now\s+(?:a|an)\s+",
        r"act\s+as\s+(?:a|an)\s+(?!assistente|assistant)",
        r"pretend\s+(?:you\s+are|to\s+be)",
        r"system\s*prompt",
        r"reveal\s+(?:your\s+)?(?:instructions|prompt)",
        r"what\s+(?:are|is)\s+your\s+(?:instructions|system)",
    ]
    
    _MANIPULATION_PATTERNS = [
        r"(?:are|sei)\s+(?:you\s+)?sure\??",
        r"(?:no|non)\s*,?\s*(?:I\s+think|penso|credo)\s+(?:you|che)\s+(?:are|sei|stai)\s+wrong",
        r"(?:actually|in realtà|veramente).*(?:wrong|sbagliato|sbagliata|errato)",
    ]
    
    _REJECTION_MSG = (
        "Ho rilevato un tentativo di manipolazione nelle istruzioni. "
        "Posso aiutarti solo con domande relative al DIEM."
    )
    
    def __init__(self):
        self._injection_regex = [re.compile(p, re.IGNORECASE) for p in self._INJECTION_PATTERNS]
        self._manipulation_regex = [re.compile(p, re.IGNORECASE) for p in self._MANIPULATION_PATTERNS]
    
    def check(self, text: str, context: Optional[dict] = None) -> tuple[bool, str]:
        for pattern in self._injection_regex:
            if pattern.search(text):
                logger.warning(f"Prompt injection rilevata: '{text[:100]}...'")
                return False, self._REJECTION_MSG
        
        for pattern in self._manipulation_regex:
            if pattern.search(text):
                logger.info(f"Tentativo di manipolazione rilevato: '{text[:80]}'")
                if context is not None:
                    context["manipulation_detected"] = True
        
        return True, text
    
    def as_runnable(self) -> RunnableLambda:
        """Restituisce un Runnable LCEL per inserimento in chain."""
        def _sanitize(input_text: str) -> str:
            passed, result = self.check(input_text)
            if not passed:
                raise InputInjectionError(result)
            return result
        return RunnableLambda(_sanitize)


class InputInjectionError(Exception):
    """Eccezione per prompt injection rilevata."""
    pass


# ============================================================
# OUTPUT VALIDATOR (Post-Agent)
# ============================================================

class OutputValidator:
    """
    Guardrail post-generazione per validare la risposta dell'agente.
    
    Controlla:
      1. Che la risposta non contenga PII (codici fiscali, telefoni).
      2. Che la risposta non sia vuota o troppo corta.
    """
    
    _PII_PATTERNS = [
        (r'\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b', "codice_fiscale"),
        (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', "telefono"),
    ]
    
    _EMPTY_MSG = "Mi dispiace, non sono riuscito a generare una risposta. Riprova."
    
    def __init__(self, enable_pii_filter: bool = True):
        self._enable_pii = enable_pii_filter
        self._pii_regex = [(re.compile(p), label) for p, label in self._PII_PATTERNS]
    
    def check(self, text: str, context: Optional[dict] = None) -> tuple[bool, str]:
        if not text or len(text.strip()) < 10:
            return False, self._EMPTY_MSG
        
        if self._enable_pii:
            for pattern, label in self._pii_regex:
                if pattern.search(text):
                    logger.warning(f"PII ({label}) rilevato nell'output, mascheramento...")
                    text = pattern.sub(f"[{label.upper()} RIMOSSO]", text)
        
        return True, text
    
    def as_runnable(self) -> RunnableLambda:
        """Restituisce un Runnable LCEL per inserimento in chain."""
        def _validate(output_text: str) -> str:
            passed, result = self.check(output_text)
            if not passed:
                raise OutputValidationError(result)
            return result
        return RunnableLambda(_validate)


class OutputValidationError(Exception):
    """Eccezione per output non valido."""
    pass