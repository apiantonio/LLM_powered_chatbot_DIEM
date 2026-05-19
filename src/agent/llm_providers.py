"""Provider LLM per il sistema RAG DIEM.

Supporta HuggingFace, Ollama e Groq come backend per i modelli linguistici.
Implementa una logica di fallback: se il provider primario (Groq) non e'
disponibile, il sistema ricade automaticamente su Ollama con Qwen.
"""

import os
import logging

from langchain_core.language_models.chat_models import BaseChatModel

from config.settings import LLMConfig

logger = logging.getLogger(__name__)


_FALLBACK_MODEL = "qwen2.5"
_FALLBACK_BASE_URL = "http://localhost:11434"


def _create_fallback_chat_model(config: LLMConfig) -> BaseChatModel:
    """Crea un ChatModel di fallback su Ollama con Qwen.

    Utilizza i parametri di temperatura e max_tokens dalla configurazione
    originale, ma forza provider Ollama e modello Qwen.

    Args:
        config: Configurazione LLM originale per ereditare temperatura e token limit.

    Returns:
        Istanza di BaseChatModel Ollama/Qwen.

    Raises:
        RuntimeError: Se anche il fallback Ollama non e' raggiungibile.
    """
    from langchain_ollama import ChatOllama

    fallback_base_url = config.ollama_base_url or _FALLBACK_BASE_URL

    logger.warning(
        "FALLBACK ATTIVO: tentativo di connessione a Ollama (%s) con modello %s",
        fallback_base_url,
        _FALLBACK_MODEL,
    )

    try:
        chat_model = ChatOllama(
            model=_FALLBACK_MODEL,
            temperature=config.temperature,
            num_predict=config.max_tokens,
            base_url=fallback_base_url,
        )
        logger.info(
            "Fallback ChatOllama istanziato: %s su %s",
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

    Per il provider Groq, viene utilizzata prioritariamente la chiave API
    dedicata al chat (groq_chat_api_key). Se non disponibile, si ricade
    sulla chiave generica (groq_api_key). Se nessuna chiave Groq e'
    configurata, il sistema ricade automaticamente su Ollama con Qwen.

    Args:
        config: Configurazione LLM con provider, modello e parametri.

    Returns:
        Istanza di BaseChatModel pronta all'uso.

    Raises:
        ValueError: Se il provider non e' supportato.
        RuntimeError: Se nessun provider e' disponibile (incluso il fallback).
    """
    provider = config.provider.lower()
    logger.info("Inizializzazione ChatModel. Provider: %s", provider.upper())

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
        logger.info("ChatHuggingFace istanziato: %s", config.model_name)
        return chat_model

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        chat_model = ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            num_predict=config.max_tokens,
            base_url=config.ollama_base_url,
        )
        logger.info("ChatOllama istanziato: %s su %s", config.model_name, config.ollama_base_url)
        return chat_model

    if provider == "groq":
        from langchain_groq import ChatGroq

        api_key = config.groq_chat_api_key or config.groq_api_key
        if not api_key:
            logger.warning(
                "Nessuna API key Groq configurata (GROQ_CHAT_API_KEY / GROQ_API_KEY). "
                "Attivazione fallback su Ollama/%s.",
                _FALLBACK_MODEL,
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
                "ChatGroq istanziato: %s (api_key: %s)",
                config.model_name,
                "GROQ_CHAT_API_KEY" if config.groq_chat_api_key else "GROQ_API_KEY",
            )
            return chat_model
        except Exception as e:
            logger.error(
                "Errore istanziazione ChatGroq (%s): %s. Attivazione fallback su Ollama/%s.",
                config.model_name,
                e,
                _FALLBACK_MODEL,
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
    In caso di chiavi mancanti, ricade su Ollama con Qwen.

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
            "Uso LLM principale (%s) con temperature=0.0",
            fallback_config.model_name,
        )

        if main_provider == "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=fallback_config.model_name,
                temperature=0.0,
                num_predict=256,
                base_url=fallback_config.ollama_base_url,
            )

        if main_provider == "huggingface":
            from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

            if not fallback_config.huggingface_api_token:
                logger.warning(
                    "HUGGINGFACEHUB_API_TOKEN mancante per rewriter. "
                    "Attivazione fallback su Ollama/%s.",
                    _FALLBACK_MODEL,
                )
                from langchain_ollama import ChatOllama

                return ChatOllama(
                    model=_FALLBACK_MODEL,
                    temperature=0.0,
                    num_predict=256,
                    base_url=fallback_config.ollama_base_url or _FALLBACK_BASE_URL,
                )

            return ChatHuggingFace(
                llm=HuggingFaceEndpoint(
                    repo_id=fallback_config.model_name,
                    temperature=0.01,
                    max_new_tokens=256,
                    huggingfacehub_api_token=fallback_config.huggingface_api_token,
                    task="text-generation",
                )
            )

        if main_provider == "groq":
            from langchain_groq import ChatGroq

            api_key = fallback_config.groq_api_key
            if not api_key:
                logger.warning(
                    "GROQ_API_KEY mancante per rewriter. "
                    "Attivazione fallback su Ollama/%s.",
                    _FALLBACK_MODEL,
                )
                from langchain_ollama import ChatOllama

                return ChatOllama(
                    model=_FALLBACK_MODEL,
                    temperature=0.0,
                    num_predict=256,
                    base_url=fallback_config.ollama_base_url or _FALLBACK_BASE_URL,
                )

            try:
                llm = ChatGroq(
                    model=fallback_config.model_name,
                    temperature=0.0,
                    max_tokens=256,
                    api_key=api_key,
                )
                logger.info(
                    "Rewriter LLM (fallback): ChatGroq (%s) - temperature=0.0",
                    fallback_config.model_name,
                )
                return llm
            except Exception as e:
                logger.error(
                    "Errore istanziazione rewriter ChatGroq: %s. "
                    "Attivazione fallback su Ollama/%s.",
                    e,
                    _FALLBACK_MODEL,
                )
                from langchain_ollama import ChatOllama

                return ChatOllama(
                    model=_FALLBACK_MODEL,
                    temperature=0.0,
                    num_predict=256,
                    base_url=fallback_config.ollama_base_url or _FALLBACK_BASE_URL,
                )

        return create_chat_model(fallback_config)

    if provider == "groq":
        from langchain_groq import ChatGroq

        model = os.getenv("REWRITER_MODEL", "llama-3.3-70b-versatile").strip()
        api_key = os.getenv("GROQ_API_KEY", "").strip()

        if not api_key:
            logger.warning(
                "GROQ_API_KEY mancante per rewriter esplicito. "
                "Attivazione fallback su Ollama/%s.",
                _FALLBACK_MODEL,
            )
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=_FALLBACK_MODEL,
                temperature=0.0,
                num_predict=256,
                base_url=fallback_config.ollama_base_url or _FALLBACK_BASE_URL,
            )

        try:
            llm = ChatGroq(
                model=model,
                temperature=0.0,
                max_tokens=256,
                api_key=api_key,
            )
            logger.info("Rewriter LLM: ChatGroq (%s) - temperature=0.0", model)
            return llm
        except Exception as e:
            logger.error(
                "Errore istanziazione rewriter ChatGroq (%s): %s. "
                "Attivazione fallback su Ollama/%s.",
                model,
                e,
                _FALLBACK_MODEL,
            )
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=_FALLBACK_MODEL,
                temperature=0.0,
                num_predict=256,
                base_url=fallback_config.ollama_base_url or _FALLBACK_BASE_URL,
            )

    raise ValueError(
        f"REWRITER_PROVIDER non supportato: '{provider}'. "
        "Valori ammessi: 'groq' (oppure lasciare vuoto per usare il LLM principale)."
    )