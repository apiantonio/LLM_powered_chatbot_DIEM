"""
agent/guardrails.py — Sistema Guardrails basato su NVIDIA NeMo Guardrails.

REFACTORING v4.0 — Da middleware custom a NeMo GuardrailsMiddleware:

  CAMBIAMENTO CHIAVE:
    Tutti i middleware custom (InjectionGuard, ToxicityFilter,
    TopicalGuardrail, HallucinationGuard, CodeGenerationGuard,
    OutputPIIGuard) sono stati SOSTITUITI da un singolo
    GuardrailsMiddleware di NeMo Guardrails.

  ARCHITETTURA:
    NeMo GuardrailsMiddleware si integra nell'agent loop di LangChain
    tramite gli hook before_model / after_model:

      User Input
        → before_model: NeMo input rails
            ├── self_check_input (injection + tossicità + OOD)
            └── se bloccato → jump_to:"end", messaggio di rifiuto
        → MODEL CALL (Qwen 7B/14B via Ollama)
        → after_model: NeMo output rails
            ├── self_check_output (codice, PII, contenuto inappropriato)
            └── se bloccato → sostituisce risposta con messaggio policy
        → Tool call? → esegui tool → torna a before_model
        → No tool call? → END

  LLM PER I CHECK:
    Llama 3.3 70B via Groq (API key dedicata: GROQ_GUARDRAILS_API_KEY)
    Separato dal rewriter (GROQ_API_KEY) e dall'agente (Ollama).

  CONFIGURAZIONE:
    La configurazione NeMo è nella directory guardrails_config/:
      - config.yml: modello + rail attive
      - prompts.yml: prompt self-check personalizzati per dominio DIEM
      - rails/input.co: Colang flow per input rails
      - rails/output.co: Colang flow per output rails

  VANTAGGI RISPETTO AI MIDDLEWARE CUSTOM:
    - Soluzione enterprise testata e mantenuta da NVIDIA
    - Self-check basato su LLM (più accurato del pattern matching regex)
    - Un unico prompt copre injection + tossicità + OOD (meno latenza)
    - Configurazione dichiarativa (YAML + Colang) separata dal codice
    - Logging e tracing integrati
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Path alla directory di configurazione NeMo Guardrails
_GUARDRAILS_CONFIG_DIR = Path(__file__).parent / "guardrails_config"

# ============================================================
# Messaggi di rifiuto (usati dal GuardrailsMiddleware)
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
# FACTORY: Costruisce il GuardrailsMiddleware NeMo
# ============================================================

def build_guardrail_middleware(
    # Parametri legacy mantenuti per retrocompatibilità di firma,
    # ma NON più usati. L'LLM dei check è Groq Llama 70B configurato
    # nel config.yml di NeMo.
    enable_pii: bool = True,
    enable_topical: bool = True,
    enable_injection: bool = True,
    enable_toxicity: bool = True,
    enable_hallucination: bool = True,
    enable_code_guard: bool = True,
) -> list:
    """
    Factory che costruisce la lista middleware per create_agent().

    v4.0: Restituisce una lista con un singolo GuardrailsMiddleware
    di NeMo Guardrails, che sostituisce tutti i middleware custom.

    Il middleware si configura via la directory guardrails_config/
    e usa Llama 3.3 70B via Groq per i self-check.

    NOTA: I limiti model_calls e tool_calls vengono gestiti dalla
    configurazione di LangGraph (recursion_limit) e non più da
    middleware dedicati. Se servono, possono essere aggiunti come
    middleware aggiuntivi nella lista restituita.

    Returns:
        Lista con il GuardrailsMiddleware (+ eventuali limiter).
    """
    middleware_list = []

    # --- Verifica che la API key Groq per i guardrails sia configurata ---
    guardrails_api_key = os.environ.get("GROQ_GUARDRAILS_API_KEY", "").strip()
    if not guardrails_api_key:
        logger.warning(
            "⚠️ GROQ_GUARDRAILS_API_KEY non configurata. "
            "I guardrails NeMo NON saranno attivati. "
            "Per attivarli, imposta la variabile d'ambiente con "
            "la tua API key Groq dedicata ai guardrails."
        )
        logger.info("   🛡️ Guardrails: DISABILITATI (nessuna API key)")
        return middleware_list

    # --- Verifica che la directory di configurazione esista ---
    if not _GUARDRAILS_CONFIG_DIR.exists():
        logger.error(
            f"❌ Directory configurazione NeMo non trovata: "
            f"{_GUARDRAILS_CONFIG_DIR}. Guardrails disabilitati."
        )
        return middleware_list

    try:
        from nemoguardrails.integrations.langchain.middleware import (
            GuardrailsMiddleware,
        )

        # Determina quali rail abilitare in base ai parametri
        # (per retrocompatibilità con la firma della funzione)
        enable_input = enable_injection or enable_toxicity or enable_topical
        enable_output = enable_code_guard or enable_hallucination or enable_pii

        guardrails_mw = GuardrailsMiddleware(
            config_path=str(_GUARDRAILS_CONFIG_DIR),
            blocked_input_message=BLOCKED_INPUT_MESSAGE,
            blocked_output_message=BLOCKED_OUTPUT_MESSAGE,
            enable_input_rails=enable_input,
            enable_output_rails=enable_output,
        )

        middleware_list.append(guardrails_mw)

        logger.info("   ✅ NeMo GuardrailsMiddleware configurato:")
        logger.info(f"      - Config: {_GUARDRAILS_CONFIG_DIR}")
        logger.info(f"      - LLM check: Groq Llama 3.3 70B")
        logger.info(f"      - Input rails: {'ON' if enable_input else 'OFF'}")
        logger.info(f"      - Output rails: {'ON' if enable_output else 'OFF'}")
        logger.info(f"      - Input check: injection + tossicità + OOD (self_check_input)")
        logger.info(f"      - Output check: codice + PII + contenuto (self_check_output)")

    except ImportError as e:
        logger.error(
            f"❌ Impossibile importare NeMo Guardrails: {e}. "
            f"Installa con: pip install nemoguardrails langchain-groq. "
            f"Guardrails disabilitati."
        )
        return middleware_list

    except Exception as e:
        logger.error(
            f"❌ Errore configurazione NeMo Guardrails: {e}. "
            f"Guardrails disabilitati."
        )
        return middleware_list

    return middleware_list