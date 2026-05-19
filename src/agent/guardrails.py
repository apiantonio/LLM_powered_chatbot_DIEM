"""Guardrails per il sistema RAG DIEM.

Implementa controlli di sicurezza input/output tramite LLM dedicato (Groq).
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_GUARDRAILS_CONFIG_DIR = Path(__file__).parent / "guardrails_config"

BLOCKED_INPUT_MESSAGE = (
    "Mi dispiace, non posso elaborare questa richiesta. "
    "Sono l'assistente virtuale del Dipartimento DIEM "
    "dell'Universita degli Studi di Salerno. "
    "Posso aiutarti con informazioni su corsi, docenti, esami, "
    "regolamenti, laboratori e servizi universitari."
)

BLOCKED_OUTPUT_MESSAGE = (
    "Mi scuso, non posso fornire questa risposta per motivi di policy. "
    "Posso aiutarti con informazioni sul Dipartimento DIEM."
)


class GuardrailsChecker:
    """Esegue controlli di sicurezza su input utente e output agente via LLM."""

    def __init__(
        self,
        llm,
        enable_input_check: bool = True,
        enable_output_check: bool = True,
    ):
        """Inizializza il checker con un LLM e i flag di abilitazione.

        Args:
            llm: Modello linguistico per i controlli di guardrail.
            enable_input_check: Se True, abilita il controllo sull'input utente.
            enable_output_check: Se True, abilita il controllo sull'output agente.
        """
        self._llm = llm
        self._enable_input = enable_input_check
        self._enable_output = enable_output_check
        self._input_prompt_template = self._load_prompt("self_check_input")
        self._output_prompt_template = self._load_prompt("self_check_output")

        logger.info(
            "GuardrailsChecker inizializzato: input=%s, output=%s",
            "ON" if enable_input_check else "OFF",
            "ON" if enable_output_check else "OFF",
        )

    def _load_prompt(self, task_name: str) -> Optional[str]:
        """Carica un template di prompt dal file prompts.yml.

        Args:
            task_name: Identificativo del task nel file YAML.

        Returns:
            Contenuto del prompt, oppure None se non trovato.
        """
        prompts_path = _GUARDRAILS_CONFIG_DIR / "prompts.yml"
        if not prompts_path.exists():
            logger.error("File prompts.yml non trovato: %s", prompts_path)
            return None

        try:
            import yaml

            with open(prompts_path, "r", encoding="utf-8") as f:
                prompts_data = yaml.safe_load(f)

            for prompt_def in prompts_data.get("prompts", []):
                if prompt_def.get("task") == task_name:
                    return prompt_def.get("content", "")

            logger.error("Prompt task '%s' non trovato in prompts.yml", task_name)
            return None

        except Exception as e:
            logger.error("Errore caricamento prompts.yml: %s", e)
            return None

    def _call_llm_check(self, prompt: str) -> Optional[bool]:
        """Invoca il LLM per un controllo di guardrail.

        Args:
            prompt: Prompt completo da inviare al LLM.

        Returns:
            True se consentito, False se bloccato, None in caso di errore.
        """
        try:
            from langchain_core.messages import HumanMessage

            response = self._llm.invoke([HumanMessage(content=prompt)])
            result_text = response.content.strip().lower()
            first_word = result_text.split()[0].strip(".,;:!?") if result_text else ""

            logger.info(
                "Guardrail LLM check result: '%s' -> first_word='%s'",
                result_text,
                first_word,
            )

            if first_word == "no":
                return True
            if first_word == "yes":
                return False

            logger.warning(
                "Guardrail check: risposta ambigua '%s'. Default: consentito.",
                result_text,
            )
            return True

        except Exception as e:
            logger.error("Errore durante il guardrail check LLM: %s", e)
            return True

    def check_input(self, user_message: str) -> Tuple[bool, str]:
        """Verifica se l'input utente e' consentito dalle policy.

        Args:
            user_message: Messaggio dell'utente da controllare.

        Returns:
            Tupla (consentito, messaggio_blocco). Se consentito, messaggio_blocco e' vuoto.
        """
        if not self._enable_input:
            return (True, "")

        if not self._input_prompt_template:
            logger.warning("Input check template non disponibile, skip.")
            return (True, "")

        prompt = self._input_prompt_template.replace("{{ user_input }}", user_message)

        logger.info("Guardrail INPUT check per: '%s...'", user_message[:80])
        allowed = self._call_llm_check(prompt)

        if allowed is None or allowed:
            return (True, "")

        logger.warning("INPUT BLOCCATO: '%s...'", user_message[:80])
        return (False, BLOCKED_INPUT_MESSAGE)

    def check_output(self, bot_response: str) -> Tuple[bool, str]:
        """Verifica se l'output dell'agente e' consentito dalle policy.

        Args:
            bot_response: Risposta dell'agente da controllare.

        Returns:
            Tupla (consentito, messaggio_blocco). Se consentito, messaggio_blocco e' vuoto.
        """
        if not self._enable_output:
            return (True, "")

        if not self._output_prompt_template:
            logger.warning("Output check template non disponibile, skip.")
            return (True, "")

        if not bot_response or not bot_response.strip():
            logger.debug("Output check: risposta vuota, skip.")
            return (True, "")

        prompt = self._output_prompt_template.replace("{{ bot_response }}", bot_response)

        logger.info("Guardrail OUTPUT check per risposta (%d chars)", len(bot_response))
        allowed = self._call_llm_check(prompt)

        if allowed is None or allowed:
            return (True, "")

        logger.warning("OUTPUT BLOCCATO (%d chars)", len(bot_response))
        return (False, BLOCKED_OUTPUT_MESSAGE)


def build_guardrails_checker(
    enable_pii: bool = True,
    enable_topical: bool = True,
    enable_injection: bool = True,
    enable_toxicity: bool = True,
    enable_hallucination: bool = True,
    enable_code_guard: bool = True,
) -> Optional[GuardrailsChecker]:
    """Factory per la creazione di un GuardrailsChecker configurato.

    Richiede GROQ_GUARDRAILS_API_KEY e i file di configurazione nella
    directory guardrails_config.

    Args:
        enable_pii: Abilita il filtro PII sull'output.
        enable_topical: Abilita il controllo di pertinenza tematica sull'input.
        enable_injection: Abilita il controllo anti-injection sull'input.
        enable_toxicity: Abilita il controllo di tossicita sull'input.
        enable_hallucination: Abilita il controllo anti-allucinazione sull'output.
        enable_code_guard: Abilita il controllo codice sull'output.

    Returns:
        Istanza di GuardrailsChecker, oppure None se non configurabile.
    """
    guardrails_api_key = os.environ.get("GROQ_GUARDRAILS_API_KEY", "").strip()
    if not guardrails_api_key:
        logger.warning(
            "GROQ_GUARDRAILS_API_KEY non configurata. "
            "I guardrails NON saranno attivati."
        )
        return None

    if not _GUARDRAILS_CONFIG_DIR.exists():
        logger.error(
            "Directory configurazione non trovata: %s. Guardrails disabilitati.",
            _GUARDRAILS_CONFIG_DIR,
        )
        return None

    prompts_path = _GUARDRAILS_CONFIG_DIR / "prompts.yml"
    if not prompts_path.exists():
        logger.error("prompts.yml non trovato: %s", prompts_path)
        return None

    try:
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            max_tokens=10,
            api_key=guardrails_api_key,
        )

        enable_input = enable_injection or enable_toxicity or enable_topical
        enable_output = enable_code_guard or enable_hallucination or enable_pii

        checker = GuardrailsChecker(
            llm=llm,
            enable_input_check=enable_input,
            enable_output_check=enable_output,
        )

        logger.info("GuardrailsChecker configurato:")
        logger.info("  - LLM: Groq Llama 3.3 70B (temperature=0.0)")
        logger.info("  - Input check: %s", "ON" if enable_input else "OFF")
        logger.info("  - Output check: %s", "ON" if enable_output else "OFF")

        return checker

    except ImportError as e:
        logger.error(
            "Impossibile importare langchain_groq: %s. "
            "Installa con: pip install langchain-groq. "
            "Guardrails disabilitati.",
            e,
        )
        return None

    except Exception as e:
        logger.error(
            "Errore configurazione GuardrailsChecker: %s. Guardrails disabilitati.",
            e,
        )
        return None