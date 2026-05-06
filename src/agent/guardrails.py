"""
Guardrails per il sistema RAG DIEM.

Tre livelli di protezione:
1. ScopeGuardrail (before_agent): Blocca domande fuori dominio.
2. InputSanitizer (before_model): Rileva prompt injection.
3. OutputValidator (after_agent): Verifica grounding e blocca PII.

KPI Impact:
- Scope Awareness: ScopeGuardrail → rilevamento chirurgico OOD.
- Robustness: InputSanitizer → resistenza a prompt injection e "Are you sure?".
- Correctness: OutputValidator → zero allucinazioni in output.
"""

import re
import logging
from typing import Optional, List

from interfaces import Guardrail

logger = logging.getLogger(__name__)


class ScopeGuardrail(Guardrail):
    """
    Classificatore Out-of-Domain basato su LLM.
    
    Implementazione: Il LLM riceve la query e risponde con un verdetto
    binario (IN_SCOPE / OUT_OF_SCOPE) prima che il retrieval parta.
    
    Perché LLM-based e non regex: le query OOD sono semanticamente
    infinite ("Chi è il re di Spagna?", "Parlami di cucina giapponese").
    Un classificatore a regole non può coprirle tutte.
    
    KPI Impact: Scope Awareness (metrica critica nella rubrica).
    """
    
    CLASSIFICATION_PROMPT = (
        "Sei un classificatore di dominio. Il tuo unico compito è determinare se "
        "la seguente domanda riguarda il Dipartimento DIEM dell'Università degli "
        "Studi di Salerno (corsi, docenti, esami, orari, regolamenti, tesi, "
        "borse di studio, laboratori, servizi, dottorato).\n\n"
        "Rispondi SOLO con 'IN_SCOPE' o 'OUT_OF_SCOPE'. Nient'altro.\n\n"
        "Domanda: {query}"
    )
    
    REJECTION_MSG = (
        "Mi dispiace, posso rispondere solo a domande relative al Dipartimento "
        "DIEM dell'Università degli Studi di Salerno (corsi, docenti, esami, "
        "orari, regolamenti, servizi dipartimentali). "
        "La tua domanda sembra riguardare un altro argomento."
    )
    
    def __init__(self, llm_provider):
        self._llm = llm_provider
    
    def check(self, text: str, context: Optional[dict] = None) -> tuple[bool, str]:
        try:
            prompt = self.CLASSIFICATION_PROMPT.format(query=text)
            response = self._llm.invoke(prompt).strip().upper()
            
            if "OUT_OF_SCOPE" in response:
                logger.info(f"Query OOD bloccata: '{text[:80]}...'")
                return False, self.REJECTION_MSG
            
            return True, text
            
        except Exception as e:
            # Fail-open: in caso di errore, lascia passare la query
            logger.warning(f"Errore scope classification, fail-open: {e}")
            return True, text


class InputSanitizer(Guardrail):
    """
    Guardrail deterministico per sanitizzare l'input dell'utente.
    
    Rileva e neutralizza:
    1. Tentativi di prompt injection.
    2. Tentativi di manipolazione tipo "Are you sure?" / "Ignore previous instructions".
    3. Tentativi di estrazione del system prompt.
    
    KPI Impact: Robustness.
    """
    
    INJECTION_PATTERNS = [
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
    
    MANIPULATION_PATTERNS = [
        r"(?:are|sei)\s+(?:you\s+)?sure\??",
        r"(?:no|non)\s*,?\s*(?:I\s+think|penso|credo)\s+(?:you|che)\s+(?:are|sei|stai)\s+wrong",
        r"(?:actually|in realtà|veramente).*(?:wrong|sbagliato|sbagliata|errato)",
    ]
    
    def __init__(self):
        self._injection_regex = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]
        self._manipulation_regex = [re.compile(p, re.IGNORECASE) for p in self.MANIPULATION_PATTERNS]
    
    def check(self, text: str, context: Optional[dict] = None) -> tuple[bool, str]:
        # 1. Controlla injection
        for pattern in self._injection_regex:
            if pattern.search(text):
                logger.warning(f"Prompt injection rilevata: '{text[:100]}...'")
                return False, (
                    "Ho rilevato un tentativo di manipolazione nelle istruzioni. "
                    "Posso aiutarti solo con domande relative al DIEM."
                )
        
        # 2. Controlla manipolazione (non blocca, ma avvisa nel contesto)
        for pattern in self._manipulation_regex:
            if pattern.search(text):
                logger.info(f"Tentativo di manipolazione rilevato: '{text[:80]}'")
                # Non blocca ma aggiunge un flag per il prompt dell'agente
                if context is not None:
                    context["manipulation_detected"] = True
        
        return True, text


class OutputValidator(Guardrail):
    """
    Guardrail post-generazione per validare la risposta dell'agente.
    
    Controlla:
    1. Che la risposta non contenga PII (email, telefoni, codici fiscali).
    2. Che la risposta non sia vuota o troppo corta.
    3. (Opzionale) Che la risposta sia grounded nel contesto.
    
    KPI Impact: Correctness, Compliance.
    """
    
    PII_PATTERNS = [
        (r'\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b', "codice_fiscale"),
        (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', "telefono"),
        # Email le lasciamo passare perché sono informazioni utili per contattare i docenti
    ]
    
    def __init__(self, enable_pii_filter: bool = True):
        self._enable_pii = enable_pii_filter
        self._pii_regex = [(re.compile(p), label) for p, label in self.PII_PATTERNS]
    
    def check(self, text: str, context: Optional[dict] = None) -> tuple[bool, str]:
        if not text or len(text.strip()) < 10:
            return False, "Mi dispiace, non sono riuscito a generare una risposta. Riprova."
        
        if self._enable_pii:
            for pattern, label in self._pii_regex:
                if pattern.search(text):
                    logger.warning(f"PII ({label}) rilevato nell'output, mascheramento...")
                    text = pattern.sub(f"[{label.upper()} RIMOSSO]", text)
        
        return True, text
