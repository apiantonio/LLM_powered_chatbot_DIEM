"""Guardrails per il sistema RAG DIEM.

Implementa controlli di sicurezza input/output tramite LLM dedicato (Groq).
Include il rilevamento delle domande "meta" (saluti, ringraziamenti, ecc.)
per gestirle senza salvataggio in memoria.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

from src.config.settings import LLMConfig, GuardrailsConfig

logger = logging.getLogger(__name__)

_GUARDRAILS_CONFIG_DIR = Path(__file__).parent / "guardrails_config"


class GuardrailsChecker:
    """Esegue controlli di sicurezza su input utente e output agente via LLM.

    Include anche il rilevamento delle domande meta (saluti, ringraziamenti,
    domande sull'identita dell'assistente) tramite un prompt LLM dedicato.
    """

    def __init__(
        self,
        llm,
        guardrails_config: Optional[GuardrailsConfig] = None,
        enable_input_check: bool = True,
        enable_output_check: bool = True,
        enable_meta_check: bool = True,
    ):
        self._llm = llm
        self._config = guardrails_config or GuardrailsConfig()
        self._enable_input = enable_input_check
        self._enable_output = enable_output_check
        self._enable_meta = enable_meta_check
        self._input_prompt_template = self._load_prompt("self_check_input")
        self._output_prompt_template = self._load_prompt("self_check_output")
        self._meta_prompt_template = self._load_prompt("self_check_meta")

        logger.info(
            "GuardrailsChecker inizializzato: input=%s, output=%s, meta=%s",
            "ON" if enable_input_check else "OFF",
            "ON" if enable_output_check else "OFF",
            "ON" if enable_meta_check else "OFF",
        )

    def _load_prompt(self, task_name: str) -> Optional[str]:
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
        try:
            from langchain_core.messages import HumanMessage

            response = self._llm.invoke([HumanMessage(content=prompt)])
            result_text = response.content.strip().lower()
            first_word = result_text.split()[0].strip(".,;:!?") if result_text else ""

            logger.debug(
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

    def _call_llm_meta_check(self, prompt: str) -> Optional[bool]:
        """Chiama l'LLM per il meta check. Logica invertita rispetto ai check standard.

        Per il meta check, "Yes" significa che E' una domanda meta (positivo),
        "No" significa che NON e' una domanda meta.

        Returns:
            True se la query e' una domanda meta, False altrimenti, None in caso di errore.
        """
        try:
            from langchain_core.messages import HumanMessage

            response = self._llm.invoke([HumanMessage(content=prompt)])
            result_text = response.content.strip().lower()
            first_word = result_text.split()[0].strip(".,;:!?") if result_text else ""

            logger.debug(
                "Guardrail META check result: '%s' -> first_word='%s'",
                result_text,
                first_word,
            )

            if first_word == "yes":
                return True
            if first_word == "no":
                return False

            logger.warning(
                "Meta check: risposta ambigua '%s'. Default: non meta.",
                result_text,
            )
            return False

        except Exception as e:
            logger.error("Errore durante il meta check LLM: %s", e)
            return False

    def check_input(self, user_message: str) -> Tuple[bool, str]:
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
        return (False, self._config.blocked_input_message)

    def check_output(self, bot_response: str) -> Tuple[bool, str]:
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
        return (False, self._config.blocked_output_message)

    def check_meta(self, user_message: str) -> bool:
        """Verifica se il messaggio utente e' una domanda meta."""
        if not self._enable_meta:
            return False

        if not self._meta_prompt_template:
            logger.warning("Meta check template non disponibile, skip.")
            return False

        prompt = self._meta_prompt_template.replace("{{ user_input }}", user_message)

        logger.info("Guardrail META check per: '%s...'", user_message[:80])
        is_meta = self._call_llm_meta_check(prompt)

        if is_meta:
            logger.info("QUERY META rilevata: '%s...'", user_message[:80])

        return bool(is_meta)


def _validate_groq_api_key(api_key: str, llm_config: LLMConfig) -> bool:
    """Verifica che la chiave API Groq sia valida con una chiamata di test."""
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage

        test_llm = ChatGroq(
            model=llm_config.groq_validation_model,
            temperature=0.0,
            max_tokens=llm_config.groq_validation_max_tokens,
            api_key=api_key,
        )
        test_llm.invoke([HumanMessage(content="test")])
        return True
    except Exception as e:
        logger.warning("Validazione API key Groq fallita: %s", e)
        return False


def build_guardrails_checker(
    llm_config: Optional[LLMConfig] = None,
    guardrails_config: Optional[GuardrailsConfig] = None,
    enable_pii: bool = True,
    enable_topical: bool = True,
    enable_injection: bool = True,
    enable_toxicity: bool = True,
    enable_hallucination: bool = True,
    enable_code_guard: bool = True,
    enable_meta: bool = True,
) -> Optional[GuardrailsChecker]:
    """Factory per la creazione di un GuardrailsChecker configurato.

    Args:
        llm_config: Configurazione LLM per modello e chiave API guardrails.
                    Se None, viene caricata da settings.
        guardrails_config: Configurazione messaggi guardrails.
                          Se None, viene caricata da settings.
    """
    if llm_config is None:
        from src.config.settings import load_settings
        settings = load_settings()
        llm_config = settings.llm

    if guardrails_config is None:
        from src.config.settings import load_settings
        settings = load_settings()
        guardrails_config = settings.guardrails

    guardrails_api_key = (llm_config.groq_guardrails_api_key or "").strip()
    if not guardrails_api_key:
        logger.warning(
            "GUARDRAILS DISABILITATI: groq_guardrails_api_key non configurata."
        )
        return None

    if not _GUARDRAILS_CONFIG_DIR.exists():
        logger.error(
            "GUARDRAILS DISABILITATI: directory configurazione non trovata: %s",
            _GUARDRAILS_CONFIG_DIR,
        )
        return None

    prompts_path = _GUARDRAILS_CONFIG_DIR / "prompts.yml"
    if not prompts_path.exists():
        logger.error("GUARDRAILS DISABILITATI: prompts.yml non trovato: %s", prompts_path)
        return None

    logger.info("Validazione GROQ_GUARDRAILS_API_KEY in corso...")
    if not _validate_groq_api_key(guardrails_api_key, llm_config):
        logger.warning(
            "GUARDRAILS DISABILITATI: GROQ_GUARDRAILS_API_KEY non valida (401 Unauthorized). "
            "Controlla la chiave nel file .env."
        )
        return None

    try:
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            model=llm_config.guardrails_model,
            temperature=0.0,
            max_tokens=llm_config.guardrails_max_tokens,
            api_key=guardrails_api_key,
        )

        enable_input = enable_injection or enable_toxicity or enable_topical
        enable_output = enable_code_guard or enable_hallucination or enable_pii

        checker = GuardrailsChecker(
            llm=llm,
            guardrails_config=guardrails_config,
            enable_input_check=enable_input,
            enable_output_check=enable_output,
            enable_meta_check=enable_meta,
        )

        logger.info("GUARDRAILS ATTIVI:")
        logger.info("  - LLM: Groq %s (temperature=0.0)", llm_config.guardrails_model)
        logger.info("  - Input check: %s", "ON" if enable_input else "OFF")
        logger.info("  - Output check: %s", "ON" if enable_output else "OFF")
        logger.info("  - Meta check: %s", "ON" if enable_meta else "OFF")

        return checker

    except ImportError as e:
        logger.error(
            "GUARDRAILS DISABILITATI: impossibile importare langchain_groq: %s. "
            "Installa con: pip install langchain-groq.",
            e,
        )
        return None

    except Exception as e:
        logger.error(
            "GUARDRAILS DISABILITATI: errore configurazione: %s",
            e,
        )
        return None