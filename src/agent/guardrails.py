import os
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_GUARDRAILS_CONFIG_DIR = Path(__file__).parent / "guardrails_config"

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

class GuardrailsChecker:

    def __init__(
        self,
        llm,
        enable_input_check: bool = True,
        enable_output_check: bool = True,
    ):
        self._llm = llm
        self._enable_input = enable_input_check
        self._enable_output = enable_output_check
        
        self._input_prompt_template = self._load_prompt("self_check_input")
        self._output_prompt_template = self._load_prompt("self_check_output")

        logger.info(
            f"GuardrailsChecker inizializzato: "
            f"input={'ON' if enable_input_check else 'OFF'}, "
            f"output={'ON' if enable_output_check else 'OFF'}"
        )

    def _load_prompt(self, task_name: str) -> Optional[str]:
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
        try:
            from langchain_core.messages import HumanMessage

            response = self._llm.invoke(
                [HumanMessage(content=prompt)],
            )

            result_text = response.content.strip().lower()

            first_word = result_text.split()[0].strip(".,;:!?") if result_text else ""

            logger.info(f"Guardrail LLM check result: '{result_text}' → first_word='{first_word}'")

            if first_word == "no":
                return True 
            elif first_word == "yes":
                return False  
            else:
                logger.warning(
                    f"Guardrail check: risposta ambigua '{result_text}'. "
                    f"Default: consentito."
                )
                return True

        except Exception as e:
            logger.error(f"Errore durante il guardrail check LLM: {e}")
    
            return True

    def check_input(self, user_message: str) -> Tuple[bool, str]:
        if not self._enable_input:
            return (True, "")

        if not self._input_prompt_template:
            logger.warning("Input check template non disponibile, skip.")
            return (True, "")

        prompt = self._input_prompt_template.replace("{{ user_input }}", user_message)

        logger.info(f"Guardrail INPUT check per: '{user_message[:80]}...'")
        allowed = self._call_llm_check(prompt)

        if allowed is None or allowed:
            return (True, "")
        else:
            logger.warning(f" INPUT BLOCCATO: '{user_message[:80]}...'")
            return (False, BLOCKED_INPUT_MESSAGE)

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

        logger.info(f"Guardrail OUTPUT check per risposta ({len(bot_response)} chars)")
        allowed = self._call_llm_check(prompt)

        if allowed is None or allowed:
            return (True, "")
        else:
            logger.warning(f" OUTPUT BLOCCATO ({len(bot_response)} chars)")
            return (False, BLOCKED_OUTPUT_MESSAGE)


def build_guardrails_checker(
    enable_pii: bool = True,
    enable_topical: bool = True,
    enable_injection: bool = True,
    enable_toxicity: bool = True,
    enable_hallucination: bool = True,
    enable_code_guard: bool = True,
) -> Optional[GuardrailsChecker]:
    guardrails_api_key = os.environ.get("GROQ_GUARDRAILS_API_KEY", "").strip()
    if not guardrails_api_key:
        logger.warning(
            " GROQ_GUARDRAILS_API_KEY non configurata. "
            "I guardrails NON saranno attivati."
        )
        return None

    if not _GUARDRAILS_CONFIG_DIR.exists():
        logger.error(
            f" Directory configurazione non trovata: "
            f"{_GUARDRAILS_CONFIG_DIR}. Guardrails disabilitati."
        )
        return None

    prompts_path = _GUARDRAILS_CONFIG_DIR / "prompts.yml"
    if not prompts_path.exists():
        logger.error(f" prompts.yml non trovato: {prompts_path}")
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

        logger.info("    GuardrailsChecker configurato:")
        logger.info(f"      - LLM: Groq Llama 3.3 70B (temperature=0.0)")
        logger.info(f"      - Input check: {'ON' if enable_input else 'OFF'}")
        logger.info(f"      - Output check: {'ON' if enable_output else 'OFF'}")

        return checker

    except ImportError as e:
        logger.error(
            f" Impossibile importare langchain_groq: {e}. "
            f"Installa con: pip install langchain-groq. "
            f"Guardrails disabilitati."
        )
        return None

    except Exception as e:
        logger.error(
            f" Errore configurazione GuardrailsChecker: {e}. "
            f"Guardrails disabilitati."
        )
        return None


