"""
agent/guardrails.py — Sistema Guardrails basato su NVIDIA NeMo Guardrails.

REFACTORING v4.1 — Da middleware a invocazione diretta:

  CAMBIAMENTO CHIAVE:
    Il GuardrailsMiddleware di NeMo NON è compatibile con un agent loop
    che usa tool-calling (LangChain create_agent). Il middleware intercetta
    l'output del PRIMO LLM call, che in un agente con tool è una tool call
    (stringa vuota), e la blocca erroneamente.

    Soluzione: usiamo NeMo Guardrails come ENGINE DI VALUTAZIONE,
    invocando direttamente LLMRails.generate() per i check, ma
    controlliamo NOI quando e cosa checkare, nel metodo RAGAgent.chat().

  ARCHITETTURA:
    RAGAgent.chat():
      1. INPUT CHECK: check_input(user_query_pulita)
         → Se bloccato: ritorna messaggio di rifiuto, STOP
      2. AGENT INVOCATION: agente gira normalmente (tool calls, ecc.)
      3. OUTPUT CHECK: check_output(risposta_finale)
         → Se bloccato: sostituisce risposta con messaggio policy

  LLM PER I CHECK:
    Llama 3.3 70B via Groq (API key: GROQ_GUARDRAILS_API_KEY)
    Invocato tramite langchain_groq.ChatGroq direttamente.

  CONFIGURAZIONE:
    I prompt per i self-check sono in guardrails_config/prompts.yml.
    Vengono caricati e usati direttamente come template.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Path alla directory di configurazione NeMo Guardrails
_GUARDRAILS_CONFIG_DIR = Path(__file__).parent / "guardrails_config"

# ============================================================
# Messaggi di rifiuto
# ============================================================

BLOCKED_INPUT_MESSAGE = (
    "Mi dispiace, non posso elaborare questa richiesta. "
    "Sono l'assistente virtuale del Dipartimento DIEM "
    "dell'Università degli Studi di Salerno. "
    "Posso aiutarti con informazioni su corsi, docenti, esami, "
    "regolamenti, laboratori e servizi universitari."
)

BLOCKED_OUTPUT_MESSAGE = (
    "Mi scuso, non posso fornire questa risposta per motivi di policy. "
    "Posso aiutarti con informazioni sul Dipartimento DIEM."
)


# ============================================================
# GuardrailsChecker: invocazione diretta dei check
# ============================================================

class GuardrailsChecker:
    """
    Esegue i check di input e output usando un LLM dedicato (Groq Llama 70B).

    NON è un middleware. Viene invocato esplicitamente da RAGAgent.chat()
    nei punti corretti del flusso:
      - check_input(): PRIMA dell'invocazione dell'agente, sulla query pulita
      - check_output(): DOPO l'invocazione, sulla risposta finale effettiva

    Questo risolve il problema del middleware NeMo che intercettava le tool
    calls vuote e le bloccava erroneamente.
    """

    def __init__(
        self,
        llm,
        enable_input_check: bool = True,
        enable_output_check: bool = True,
    ):
        self._llm = llm
        self._enable_input = enable_input_check
        self._enable_output = enable_output_check

        # Carica i prompt dai file YAML
        self._input_prompt_template = self._load_prompt("self_check_input")
        self._output_prompt_template = self._load_prompt("self_check_output")

        logger.info(
            f"GuardrailsChecker inizializzato: "
            f"input={'ON' if enable_input_check else 'OFF'}, "
            f"output={'ON' if enable_output_check else 'OFF'}"
        )

    def _load_prompt(self, task_name: str) -> Optional[str]:
        """Carica il template del prompt dal file prompts.yml."""
        prompts_path = _GUARDRAILS_CONFIG_DIR / "prompts.yml"
        if not prompts_path.exists():
            logger.error(f"File prompts.yml non trovato: {prompts_path}")
            return None

        try:
            import yaml
            with open(prompts_path, "r", encoding="utf-8") as f:
                prompts_data = yaml.safe_load(f)

            for prompt_def in prompts_data.get("prompts", []):
                if prompt_def.get("task") == task_name:
                    return prompt_def.get("content", "")

            logger.error(f"Prompt task '{task_name}' non trovato in prompts.yml")
            return None

        except Exception as e:
            logger.error(f"Errore caricamento prompts.yml: {e}")
            return None

    def _call_llm_check(self, prompt: str) -> Optional[bool]:
        """
        Invoca l'LLM per un check e interpreta la risposta Yes/No.

        Returns:
            True se la risposta è consentita (No = non bloccare),
            False se deve essere bloccata (Yes = blocca),
            None se non riesce a determinare.
        """
        try:
            from langchain_core.messages import HumanMessage

            response = self._llm.invoke(
                [HumanMessage(content=prompt)],
            )

            result_text = response.content.strip().lower()

            # Llama 3.3 70B tende a rispondere "No.\n\nThe..." con max_tokens=3
            # Prendiamo solo la prima parola/linea
            first_word = result_text.split()[0].strip(".,;:!?") if result_text else ""

            logger.info(f"Guardrail LLM check result: '{result_text}' → first_word='{first_word}'")

            if first_word == "no":
                return True  # Non bloccare → consentito
            elif first_word == "yes":
                return False  # Blocca
            else:
                # Risposta ambigua: per sicurezza, consenti
                logger.warning(
                    f"Guardrail check: risposta ambigua '{result_text}'. "
                    f"Default: consentito."
                )
                return True

        except Exception as e:
            logger.error(f"Errore durante il guardrail check LLM: {e}")
            # In caso di errore, consenti (fail-open)
            return True

    def check_input(self, user_message: str) -> Tuple[bool, str]:
        """
        Verifica se il messaggio utente è consentito.

        Args:
            user_message: Il messaggio utente PULITO (senza suffissi di sistema).

        Returns:
            (allowed, message): allowed=True se consentito,
            altrimenti (False, BLOCKED_INPUT_MESSAGE).
        """
        if not self._enable_input:
            return (True, "")

        if not self._input_prompt_template:
            logger.warning("Input check template non disponibile, skip.")
            return (True, "")

        # Sostituisci il placeholder nel template
        prompt = self._input_prompt_template.replace("{{ user_input }}", user_message)

        logger.info(f"Guardrail INPUT check per: '{user_message[:80]}...'")
        allowed = self._call_llm_check(prompt)

        if allowed is None or allowed:
            return (True, "")
        else:
            logger.warning(f"🛡️ INPUT BLOCCATO: '{user_message[:80]}...'")
            return (False, BLOCKED_INPUT_MESSAGE)

    def check_output(self, bot_response: str) -> Tuple[bool, str]:
        """
        Verifica se la risposta dell'agente è consentita.

        Args:
            bot_response: La risposta FINALE e COMPLETA dell'agente.

        Returns:
            (allowed, message): allowed=True se consentita,
            altrimenti (False, BLOCKED_OUTPUT_MESSAGE).
        """
        if not self._enable_output:
            return (True, "")

        if not self._output_prompt_template:
            logger.warning("Output check template non disponibile, skip.")
            return (True, "")

        # Non checkare risposte vuote o di errore
        if not bot_response or not bot_response.strip():
            logger.debug("Output check: risposta vuota, skip.")
            return (True, "")

        # Sostituisci il placeholder nel template
        prompt = self._output_prompt_template.replace("{{ bot_response }}", bot_response)

        logger.info(f"Guardrail OUTPUT check per risposta ({len(bot_response)} chars)")
        allowed = self._call_llm_check(prompt)

        if allowed is None or allowed:
            return (True, "")
        else:
            logger.warning(f"🛡️ OUTPUT BLOCCATO ({len(bot_response)} chars)")
            return (False, BLOCKED_OUTPUT_MESSAGE)


# ============================================================
# FACTORY: Costruisce il GuardrailsChecker
# ============================================================

def build_guardrails_checker(
    enable_pii: bool = True,
    enable_topical: bool = True,
    enable_injection: bool = True,
    enable_toxicity: bool = True,
    enable_hallucination: bool = True,
    enable_code_guard: bool = True,
) -> Optional[GuardrailsChecker]:
    """
    Factory che costruisce un GuardrailsChecker.

    v4.1: NON restituisce più un middleware, ma un checker standalone
    che viene invocato esplicitamente da RAGAgent.chat().

    Returns:
        GuardrailsChecker se configurato correttamente, None altrimenti.
    """
    # --- Verifica API key Groq dedicata ai guardrails ---
    guardrails_api_key = os.environ.get("GROQ_GUARDRAILS_API_KEY", "").strip()
    if not guardrails_api_key:
        logger.warning(
            "⚠️ GROQ_GUARDRAILS_API_KEY non configurata. "
            "I guardrails NON saranno attivati."
        )
        return None

    # --- Verifica directory di configurazione ---
    if not _GUARDRAILS_CONFIG_DIR.exists():
        logger.error(
            f"❌ Directory configurazione non trovata: "
            f"{_GUARDRAILS_CONFIG_DIR}. Guardrails disabilitati."
        )
        return None

    # --- Verifica prompts.yml ---
    prompts_path = _GUARDRAILS_CONFIG_DIR / "prompts.yml"
    if not prompts_path.exists():
        logger.error(f"❌ prompts.yml non trovato: {prompts_path}")
        return None

    try:
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            max_tokens=10,  # Serve solo Yes/No, ma diamo margine
            api_key=guardrails_api_key,
        )

        enable_input = enable_injection or enable_toxicity or enable_topical
        enable_output = enable_code_guard or enable_hallucination or enable_pii

        checker = GuardrailsChecker(
            llm=llm,
            enable_input_check=enable_input,
            enable_output_check=enable_output,
        )

        logger.info("   ✅ GuardrailsChecker configurato:")
        logger.info(f"      - LLM: Groq Llama 3.3 70B (temperature=0.0)")
        logger.info(f"      - Input check: {'ON' if enable_input else 'OFF'}")
        logger.info(f"      - Output check: {'ON' if enable_output else 'OFF'}")

        return checker

    except ImportError as e:
        logger.error(
            f"❌ Impossibile importare langchain_groq: {e}. "
            f"Installa con: pip install langchain-groq. "
            f"Guardrails disabilitati."
        )
        return None

    except Exception as e:
        logger.error(
            f"❌ Errore configurazione GuardrailsChecker: {e}. "
            f"Guardrails disabilitati."
        )
        return None


