"""Provider LLM per il sistema RAG DIEM.

Supporta HuggingFace, Ollama e Groq come backend per i modelli linguistici.
Implementa una logica di fallback: se il provider primario (Groq) non e'
disponibile o la chiave API non e' valida, il sistema ricade automaticamente
su Ollama con Qwen.

Per i modelli thinking (Nemotron, Qwen3, DeepSeek-R1) via Ollama,
reasoning e' disabilitato per evitare che i thinking tokens producano
risposte vuote quando combinati con tool calling.
"""

import os
import logging

from langchain_core.language_models.chat_models import BaseChatModel

from config.settings import LLMConfig

logger = logging.getLogger(__name__)


_FALLBACK_MODEL = "qwen2.5"
_FALLBACK_BASE_URL = "http://localhost:11434"

_THINKING_MODEL_PREFIXES = (
    "nemotron",
    "deepseek-r1",
    "qwen3",
    "qwen3.5",
    "gpt-oss",
)


def _is_thinking_model(model_name: str) -> bool:
    """Verifica se il modello e' un modello thinking che richiede reasoning=False."""
    name_lower = model_name.lower().split(":")[0]
    return any(name_lower.startswith(prefix) for prefix in _THINKING_MODEL_PREFIXES)


def _validate_groq_key(api_key: str, label: str) -> bool:
    """Verifica che una chiave API Groq sia valida con una chiamata di test.

    Args:
        api_key: La chiave API da validare.
        label: Etichetta per il logging (es. 'GROQ_CHAT_API_KEY').

    Returns:
        True se la chiave e' valida, False altrimenti.
    """
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage

        test_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            max_tokens=5,
            api_key=api_key,
        )
        test_llm.invoke([HumanMessage(content="test")])
        logger.info("Validazione %s: OK", label)
        return True
    except Exception as e:
        logger.warning("Validazione %s fallita: %s", label, e)
        return False


def _create_fallback_chat_model(config: LLMConfig) -> BaseChatModel:
    """Crea un ChatModel di fallback su Ollama con Qwen.

    Args:
        config: Configurazione LLM originale per ereditare temperatura e token limit.

    Returns:
        Istanza di BaseChatModel Ollama/Qwen.

    Raises:
        RuntimeError: Se anche il fallback Ollama non e' raggiungibile.
    """
    from langchain_ollama import ChatOllama

    fallback_base_url = config.ollama_base_url or _FALLBACK_BASE_URL

    try:
        chat_model = ChatOllama(
            model=_FALLBACK_MODEL,
            temperature=config.temperature,
            num_predict=config.max_tokens,
            base_url=fallback_base_url,
        )
        logger.info(
            "CHAT MODEL ATTIVO: Ollama/%s su %s (FALLBACK)",
            _FALLBACK_MODEL,
            fallback_base_url,
        )
        return chat_model
    except Exception as e:
        raise RuntimeError(
            f"Impossibile istanziare il fallback Ollama ({_FALLBACK_MODEL} "
            f"su {fallback_base_url}): {e}. "
            f"Nessun LLM disponibile."
        ) from e


def create_chat_model(config: LLMConfig) -> BaseChatModel:
    """Crea e restituisce un ChatModel in base al provider configurato.

    Per modelli thinking (Nemotron, Qwen3, DeepSeek-R1) via Ollama,
    imposta reasoning=False per evitare conflitti con il tool calling.

    Args:
        config: Configurazione LLM con provider, modello e parametri.

    Returns:
        Istanza di BaseChatModel pronta all'uso.

    Raises:
        ValueError: Se il provider non e' supportato.
        RuntimeError: Se nessun provider e' disponibile (incluso il fallback).
    """
    provider = config.provider.lower()
    logger.info("Inizializzazione ChatModel. Provider richiesto: %s", provider.upper())

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

        if not config.huggingface_api_token:
            logger.warning(
                "HUGGINGFACEHUB_API_TOKEN non configurato. Attivazione fallback."
            )
            return _create_fallback_chat_model(config)

        chat_model = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                repo_id=config.model_name,
                temperature=config.temperature,
                max_new_tokens=config.max_tokens,
                huggingfacehub_api_token=config.huggingface_api_token,
                task="text-generation",
            )
        )
        logger.info("CHAT MODEL ATTIVO: HuggingFace/%s", config.model_name)
        return chat_model

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        ollama_kwargs = dict(
            model=config.model_name,
            temperature=config.temperature,
            num_predict=config.max_tokens,
            base_url=config.ollama_base_url,
        )

        if _is_thinking_model(config.model_name):
            ollama_kwargs["reasoning"] = False
            logger.info(
                "Modello thinking rilevato (%s): reasoning=False per compatibilita tool calling",
                config.model_name,
            )

        chat_model = ChatOllama(**ollama_kwargs)
        logger.info(
            "CHAT MODEL ATTIVO: Ollama/%s su %s",
            config.model_name,
            config.ollama_base_url,
        )
        return chat_model

    if provider == "groq":
        from langchain_groq import ChatGroq

        api_key = config.groq_chat_api_key or config.groq_rewriter_api_key
        if not api_key:
            logger.warning(
                "Nessuna API key Groq configurata (GROQ_CHAT_API_KEY / GROQ_REWRITER_API_KEY). "
                "Attivazione fallback."
            )
            return _create_fallback_chat_model(config)

        key_label = "GROQ_CHAT_API_KEY" if config.groq_chat_api_key else "GROQ_REWRITER_API_KEY"
        logger.info("Validazione %s per ChatModel in corso...", key_label)

        if not _validate_groq_key(api_key, key_label):
            logger.warning(
                "CHAT MODEL: %s non valida (401 Unauthorized). Attivazione fallback.",
                key_label,
            )
            return _create_fallback_chat_model(config)

        try:
            chat_model = ChatGroq(
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                api_key=api_key,
            )
            logger.info(
                "CHAT MODEL ATTIVO: Groq/%s (api_key: %s)",
                config.model_name,
                key_label,
            )
            return chat_model
        except Exception as e:
            logger.error(
                "Errore istanziazione ChatGroq (%s): %s. Attivazione fallback.",
                config.model_name,
                e,
            )
            return _create_fallback_chat_model(config)

    raise ValueError(
        f"Provider LLM non supportato: '{provider}'. "
        "Scegliere tra 'huggingface', 'ollama' o 'groq'."
    )


def create_rewriter_llm(fallback_config: LLMConfig) -> BaseChatModel:
    """Crea un LLM dedicato al query rewriting.

    Se REWRITER_PROVIDER e' configurato, usa un LLM separato (Groq).
    Altrimenti riutilizza il provider principale con temperature=0.0.
    In caso di chiavi mancanti o invalide, ricade su Ollama con Qwen.

    Args:
        fallback_config: Configurazione LLM principale usata come fallback.

    Returns:
        Istanza di BaseChatModel configurata per il rewriting.

    Raises:
        ValueError: Se il provider del rewriter non e' supportato.
        RuntimeError: Se nessun provider e' disponibile (incluso il fallback).
    """
    provider = os.getenv("REWRITER_PROVIDER", "").strip().lower()

    if not provider:
        main_provider = fallback_config.provider.lower()
        logger.info(
            "REWRITER_PROVIDER non configurato. "
            "Uso LLM principale (%s/%s) con temperature=0.0",
            main_provider.upper(),
            fallback_config.model_name,
        )

        if main_provider == "ollama":
            from langchain_ollama import ChatOllama

            ollama_kwargs = dict(
                model=fallback_config.model_name,
                temperature=0.0,
                num_predict=256,
                base_url=fallback_config.ollama_base_url,
            )

            if _is_thinking_model(fallback_config.model_name):
                ollama_kwargs["reasoning"] = False

            llm = ChatOllama(**ollama_kwargs)
            logger.info(
                "REWRITER ATTIVO: Ollama/%s su %s (temperature=0.0)",
                fallback_config.model_name,
                fallback_config.ollama_base_url,
            )
            return llm

        if main_provider == "huggingface":
            from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

            if not fallback_config.huggingface_api_token:
                logger.warning(
                    "HUGGINGFACEHUB_API_TOKEN mancante per rewriter. Attivazione fallback."
                )
                return _create_rewriter_fallback(fallback_config)

            llm = ChatHuggingFace(
                llm=HuggingFaceEndpoint(
                    repo_id=fallback_config.model_name,
                    temperature=0.01,
                    max_new_tokens=256,
                    huggingfacehub_api_token=fallback_config.huggingface_api_token,
                    task="text-generation",
                )
            )
            logger.info(
                "REWRITER ATTIVO: HuggingFace/%s (temperature=0.01)",
                fallback_config.model_name,
            )
            return llm

        if main_provider == "groq":
            from langchain_groq import ChatGroq

            api_key = fallback_config.groq_rewriter_api_key
            if not api_key:
                logger.warning(
                    "GROQ_REWRITER_API_KEY mancante. Attivazione fallback."
                )
                return _create_rewriter_fallback(fallback_config)

            logger.info("Validazione GROQ_REWRITER_API_KEY per Rewriter in corso...")
            if not _validate_groq_key(api_key, "GROQ_REWRITER_API_KEY"):
                logger.warning(
                    "REWRITER: GROQ_REWRITER_API_KEY non valida. Attivazione fallback."
                )
                return _create_rewriter_fallback(fallback_config)

            try:
                llm = ChatGroq(
                    model=fallback_config.model_name,
                    temperature=0.0,
                    max_tokens=256,
                    api_key=api_key,
                )
                logger.info(
                    "REWRITER ATTIVO: Groq/%s (temperature=0.0)",
                    fallback_config.model_name,
                )
                return llm
            except Exception as e:
                logger.error(
                    "Errore istanziazione rewriter ChatGroq: %s. Attivazione fallback.",
                    e,
                )
                return _create_rewriter_fallback(fallback_config)

        return create_chat_model(fallback_config)

    if provider == "groq":
        from langchain_groq import ChatGroq

        model = os.getenv("REWRITER_MODEL", "llama-3.3-70b-versatile").strip()
        api_key = os.getenv("GROQ_REWRITER_API_KEY", "").strip()

        if not api_key:
            logger.warning(
                "GROQ_REWRITER_API_KEY mancante per rewriter esplicito. Attivazione fallback."
            )
            return _create_rewriter_fallback(fallback_config)

        logger.info("Validazione GROQ_REWRITER_API_KEY per Rewriter dedicato in corso...")
        if not _validate_groq_key(api_key, "GROQ_REWRITER_API_KEY"):
            logger.warning(
                "REWRITER: GROQ_REWRITER_API_KEY non valida (401 Unauthorized). "
                "Attivazione fallback."
            )
            return _create_rewriter_fallback(fallback_config)

        try:
            llm = ChatGroq(
                model=model,
                temperature=0.0,
                max_tokens=256,
                api_key=api_key,
            )
            logger.info(
                "REWRITER ATTIVO: Groq/%s dedicato (temperature=0.0)",
                model,
            )
            return llm
        except Exception as e:
            logger.error(
                "Errore istanziazione rewriter ChatGroq (%s): %s. Attivazione fallback.",
                model,
                e,
            )
            return _create_rewriter_fallback(fallback_config)

    raise ValueError(
        f"REWRITER_PROVIDER non supportato: '{provider}'. "
        "Valori ammessi: 'groq' (oppure lasciare vuoto per usare il LLM principale)."
    )


def _create_rewriter_fallback(config: LLMConfig) -> BaseChatModel:
    """Crea un LLM di fallback per il rewriter su Ollama/Qwen.

    Args:
        config: Configurazione LLM per ereditare base_url.

    Returns:
        Istanza di BaseChatModel Ollama/Qwen per il rewriting.
    """
    from langchain_ollama import ChatOllama

    base_url = config.ollama_base_url or _FALLBACK_BASE_URL

    llm = ChatOllama(
        model=_FALLBACK_MODEL,
        temperature=0.0,
        num_predict=256,
        base_url=base_url,
    )
    logger.info(
        "REWRITER ATTIVO: Ollama/%s su %s (FALLBACK, temperature=0.0)",
        _FALLBACK_MODEL,
        base_url,
    )
    return llm