"""Judge LLM robusto per la valutazione RAGAS del sistema RAG DIEM.

Implementa un wrapper attorno al modello Judge che garantisce:
1. Output JSON valido attraverso riparazione automatica
2. Retry con backoff esponenziale su errori transitori
3. Gestione esplicita dei rate limit (HTTP 429)
4. Logging completo di ogni tentativo per tracciabilita'

Il wrapper si interpone tra RAGAS e il provider LLM (Groq),
intercettando le risposte e validandole prima che RAGAS tenti
il parsing. Se il JSON e' malformato, viene riparato o la
chiamata viene ripetuta con un prompt correttivo.

Architettura:
    RAGAS -> LangchainLLMWrapper -> RobustJudgeLLM -> ChatGroq
                                        |
                                   JSON Repair
                                   Retry Logic
                                   Rate Limit Handler
"""

import json
import re
import time
import logging
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration

from src.evaluation.config import EvaluationConfig, JudgeLLMConfig, RetryConfig

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Utilita' di riparazione JSON
# --------------------------------------------------------------------------- #

# Pattern per estrarre blocchi JSON da risposte con testo extra
_JSON_BLOCK_RE = re.compile(
    r"```(?:json)?\s*\n?(.*?)\n?\s*```",
    re.DOTALL,
)

_JSON_OBJECT_RE = re.compile(
    r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
    re.DOTALL,
)

_JSON_ARRAY_RE = re.compile(
    r"\[.*\]",
    re.DOTALL,
)


def _try_parse_json(text: str) -> Optional[dict | list]:
    """Tenta il parsing diretto di una stringa JSON.

    Args:
        text: Stringa candidata al parsing.

    Returns:
        Oggetto Python parsato, oppure None se il parsing fallisce.
    """
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return None


def repair_json_output(raw_text: str) -> str:
    """Ripara e estrae JSON valido da una risposta LLM potenzialmente sporca.

    Strategie applicate in ordine di priorita':
    1. Parsing diretto della risposta completa
    2. Estrazione da blocchi markdown ```json ... ```
    3. Ricerca del primo oggetto JSON { ... } nella risposta
    4. Ricerca del primo array JSON [ ... ] nella risposta
    5. Rimozione di prefissi/suffissi comuni e retry
    6. Restituzione del testo originale come fallback

    Args:
        raw_text: Testo grezzo della risposta del Judge LLM.

    Returns:
        Stringa contenente JSON valido, o il testo originale se
        nessuna strategia di riparazione ha successo.
    """
    if not raw_text or not raw_text.strip():
        return raw_text

    text = raw_text.strip()

    # Strategia 1: parsing diretto
    parsed = _try_parse_json(text)
    if parsed is not None:
        return text

    # Strategia 2: estrazione da blocchi markdown
    md_match = _JSON_BLOCK_RE.search(text)
    if md_match:
        candidate = md_match.group(1).strip()
        parsed = _try_parse_json(candidate)
        if parsed is not None:
            logger.debug("JSON estratto da blocco markdown.")
            return candidate

    # Strategia 3: ricerca oggetto JSON
    obj_match = _JSON_OBJECT_RE.search(text)
    if obj_match:
        candidate = obj_match.group(0)
        parsed = _try_parse_json(candidate)
        if parsed is not None:
            logger.debug("JSON oggetto estratto dalla risposta.")
            return candidate

    # Strategia 4: ricerca array JSON
    arr_match = _JSON_ARRAY_RE.search(text)
    if arr_match:
        candidate = arr_match.group(0)
        parsed = _try_parse_json(candidate)
        if parsed is not None:
            logger.debug("JSON array estratto dalla risposta.")
            return candidate

    # Strategia 5: pulizia prefissi/suffissi comuni
    cleaning_patterns = [
        (r"^[^{\[]*", ""),       # rimuovi tutto prima di { o [
        (r"[^}\]]*$", ""),       # rimuovi tutto dopo } o ]
        (r",\s*([}\]])", r"\1"), # rimuovi virgole finali
        (r"'", '"'),             # apici singoli -> doppi
    ]

    cleaned = text
    for pattern, replacement in cleaning_patterns:
        cleaned = re.sub(pattern, replacement, cleaned)

    parsed = _try_parse_json(cleaned)
    if parsed is not None:
        logger.debug("JSON riparato dopo pulizia pattern.")
        return cleaned

    # Strategia 6: tentativo con json_repair se disponibile
    try:
        import json_repair
        repaired = json_repair.repair_json(text)
        parsed = _try_parse_json(repaired)
        if parsed is not None:
            logger.debug("JSON riparato tramite json_repair.")
            return repaired
    except ImportError:
        pass

    logger.warning(
        "Impossibile riparare il JSON. Restituzione testo originale "
        "(primi 200 chars): '%s'",
        text[:200],
    )
    return raw_text


# --------------------------------------------------------------------------- #
#  Classificazione errori per retry intelligente
# --------------------------------------------------------------------------- #

def _is_rate_limit_error(error: Exception) -> bool:
    """Determina se un'eccezione e' causata da rate limiting (HTTP 429).

    Args:
        error: Eccezione catturata.

    Returns:
        True se l'errore e' un rate limit, False altrimenti.
    """
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()

    rate_limit_indicators = [
        "429",
        "rate_limit",
        "rate limit",
        "ratelimit",
        "too many requests",
        "quota",
        "tokens per minute",
        "requests per minute",
        "exceeded",
    ]

    return any(indicator in error_str or indicator in error_type
               for indicator in rate_limit_indicators)


def _is_transient_error(error: Exception) -> bool:
    """Determina se un'eccezione e' un errore transitorio recuperabile.

    Args:
        error: Eccezione catturata.

    Returns:
        True se l'errore e' transitorio e vale la pena riprovare.
    """
    error_str = str(error).lower()

    transient_indicators = [
        "timeout",
        "connection",
        "503",
        "502",
        "500",
        "server error",
        "temporarily unavailable",
        "service unavailable",
    ]

    return any(indicator in error_str for indicator in transient_indicators)


# --------------------------------------------------------------------------- #
#  Wrapper LLM robusto
# --------------------------------------------------------------------------- #

class RobustJudgeLLM(BaseChatModel):
    """Wrapper trasparente attorno al Judge LLM con retry e JSON-repair.

    Si comporta come un normale BaseChatModel di LangChain, ma intercetta
    ogni risposta per:
    1. Riparare automaticamente output JSON malformati
    2. Riprovare le chiamate in caso di errori transitori
    3. Attendere e riprovare su rate limit (429)

    Questo wrapper e' progettato specificamente per essere usato con
    LangchainLLMWrapper di RAGAS, che si aspetta un BaseChatModel
    compatibile con l'interfaccia LangChain.

    Attributes:
        inner_llm: Il modello LLM effettivo (es. ChatGroq).
        retry_config: Configurazione della policy di retry.
        total_calls: Contatore totale delle chiamate effettuate.
        total_retries: Contatore totale dei retry eseguiti.
        total_repairs: Contatore totale delle riparazioni JSON.
        total_rate_limits: Contatore dei rate limit incontrati.
    """

    inner_llm: BaseChatModel
    retry_config: RetryConfig = RetryConfig()

    # Contatori di osservabilita' (non frozen, aggiornati a runtime)
    total_calls: int = 0
    total_retries: int = 0
    total_repairs: int = 0
    total_rate_limits: int = 0

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return f"robust_judge_{getattr(self.inner_llm, '_llm_type', 'unknown')}"

    @property
    def _identifying_params(self) -> dict:
        return {
            "inner_llm": str(type(self.inner_llm).__name__),
            "max_retries": self.retry_config.max_retries,
        }

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Genera una risposta con retry e JSON-repair.

        Override del metodo core di BaseChatModel. Ogni chiamata passa
        attraverso il ciclo di retry e la riparazione JSON prima di
        restituire il risultato a RAGAS.

        Args:
            messages: Lista di messaggi LangChain da inviare al modello.
            stop: Sequenze di stop opzionali.
            run_manager: Callback manager opzionale.
            **kwargs: Argomenti aggiuntivi passati al modello interno.

        Returns:
            ChatResult con la risposta (eventualmente riparata).

        Raises:
            Exception: Se tutti i tentativi di retry falliscono.
        """
        self.total_calls += 1
        last_exception = None

        for attempt in range(1, self.retry_config.max_retries + 1):
            try:
                result = self.inner_llm._generate(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )

                if result.generations:
                    original_text = result.generations[0].text
                    repaired_text = repair_json_output(original_text)

                    if repaired_text != original_text:
                        self.total_repairs += 1
                        logger.info(
                            "JSON riparato alla chiamata #%d (tentativo %d/%d). "
                            "Originale (primi 100 chars): '%s'",
                            self.total_calls,
                            attempt,
                            self.retry_config.max_retries,
                            original_text[:100],
                        )

                        repaired_message = AIMessage(content=repaired_text)
                        repaired_generation = ChatGeneration(
                            message=repaired_message,
                            text=repaired_text,
                        )
                        result = ChatResult(generations=[repaired_generation])

                return result

            except Exception as e:
                last_exception = e

                if _is_rate_limit_error(e):
                    self.total_rate_limits += 1
                    wait_time = self.retry_config.rate_limit_wait_seconds

                    logger.warning(
                        "Rate limit (429) alla chiamata #%d, tentativo %d/%d. "
                        "Attesa di %.0f secondi prima del retry...",
                        self.total_calls,
                        attempt,
                        self.retry_config.max_retries,
                        wait_time,
                    )
                    time.sleep(wait_time)
                    continue

                if _is_transient_error(e) and attempt < self.retry_config.max_retries:
                    delay = min(
                        self.retry_config.base_delay_seconds * (2 ** (attempt - 1)),
                        self.retry_config.max_delay_seconds,
                    )
                    self.total_retries += 1

                    logger.warning(
                        "Errore transitorio alla chiamata #%d, tentativo %d/%d: %s. "
                        "Retry tra %.1f secondi...",
                        self.total_calls,
                        attempt,
                        self.retry_config.max_retries,
                        str(e)[:150],
                        delay,
                    )
                    time.sleep(delay)
                    continue

                logger.error(
                    "Errore non recuperabile alla chiamata #%d, tentativo %d/%d: %s",
                    self.total_calls,
                    attempt,
                    self.retry_config.max_retries,
                    e,
                )
                raise

        logger.error(
            "Tutti i %d tentativi esauriti per la chiamata #%d. Ultimo errore: %s",
            self.retry_config.max_retries,
            self.total_calls,
            last_exception,
        )
        raise last_exception

    def get_stats(self) -> dict:
        """Restituisce le statistiche di utilizzo del wrapper.

        Returns:
            Dizionario con contatori di chiamate, retry, riparazioni e rate limit.
        """
        return {
            "total_calls": self.total_calls,
            "total_retries": self.total_retries,
            "total_repairs": self.total_repairs,
            "total_rate_limits": self.total_rate_limits,
            "repair_rate": (
                f"{(self.total_repairs / self.total_calls * 100):.1f}%"
                if self.total_calls > 0 else "N/A"
            ),
        }

    def print_stats(self) -> None:
        """Stampa un riepilogo delle statistiche del Judge."""
        stats = self.get_stats()
        logger.info("=" * 50)
        logger.info("  STATISTICHE JUDGE LLM")
        logger.info("=" * 50)
        logger.info("  Chiamate totali:    %d", stats["total_calls"])
        logger.info("  Retry eseguiti:     %d", stats["total_retries"])
        logger.info("  Riparazioni JSON:   %d", stats["total_repairs"])
        logger.info("  Rate limit (429):   %d", stats["total_rate_limits"])
        logger.info("  Tasso riparazione:  %s", stats["repair_rate"])
        logger.info("=" * 50)


# --------------------------------------------------------------------------- #
#  Factory
# --------------------------------------------------------------------------- #

def _create_groq_judge(config: JudgeLLMConfig) -> BaseChatModel:
    """Crea un'istanza di ChatGroq per il Judge.

    Args:
        config: Configurazione del Judge LLM.

    Returns:
        Istanza di ChatGroq configurata.

    Raises:
        ImportError: Se langchain-groq non e' installato.
        ValueError: Se la API key non e' configurata.
    """
    try:
        from langchain_groq import ChatGroq
    except ImportError as e:
        raise ImportError(
            "langchain-groq non installato. Installa con: pip install langchain-groq"
        ) from e

    if not config.api_key:
        raise ValueError(
            "API key mancante per il Judge Groq. Configura EVAL_JUDGE_API_KEY "
            "o una delle chiavi GROQ_* nel file .env."
        )

    llm = ChatGroq(
        model=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        api_key=config.api_key,
        timeout=config.request_timeout,
    )

    logger.info(
        "Judge LLM Groq creato: %s (temperature=%.1f, max_tokens=%d)",
        config.model_name,
        config.temperature,
        config.max_tokens,
    )
    return llm


def _create_ollama_judge(config: JudgeLLMConfig) -> BaseChatModel:
    """Crea un'istanza di ChatOllama per il Judge (fallback locale).

    Args:
        config: Configurazione del Judge LLM.

    Returns:
        Istanza di ChatOllama configurata.

    Raises:
        ImportError: Se langchain-ollama non e' installato.
    """
    try:
        from langchain_ollama import ChatOllama
    except ImportError as e:
        raise ImportError(
            "langchain-ollama non installato. Installa con: pip install langchain-ollama"
        ) from e

    import os
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    llm = ChatOllama(
        model=config.model_name,
        temperature=config.temperature,
        num_predict=config.max_tokens,
        base_url=base_url,
    )

    logger.info(
        "Judge LLM Ollama creato: %s su %s (temperature=%.1f)",
        config.model_name,
        base_url,
        config.temperature,
    )
    return llm


def create_judge_llm(config: EvaluationConfig) -> RobustJudgeLLM:
    """Factory principale per la creazione del Judge LLM robusto.

    Crea il modello LLM interno in base al provider configurato e lo
    avvolge nel RobustJudgeLLM per garantire retry e JSON-repair.

    Args:
        config: Configurazione completa dell'evaluation.

    Returns:
        Istanza di RobustJudgeLLM pronta per essere usata con RAGAS.

    Raises:
        ValueError: Se il provider non e' supportato.
    """
    provider = config.judge.provider.lower()

    if provider == "groq":
        inner_llm = _create_groq_judge(config.judge)
    elif provider == "ollama":
        inner_llm = _create_ollama_judge(config.judge)
    else:
        raise ValueError(
            f"Provider Judge non supportato: '{provider}'. "
            "Valori ammessi: 'groq', 'ollama'."
        )

    robust_llm = RobustJudgeLLM(
        inner_llm=inner_llm,
        retry_config=config.retry,
    )

    logger.info(
        "RobustJudgeLLM assemblato: provider=%s, model=%s, "
        "max_retries=%d, rate_limit_wait=%.0fs",
        provider,
        config.judge.model_name,
        config.retry.max_retries,
        config.retry.rate_limit_wait_seconds,
    )

    return robust_llm