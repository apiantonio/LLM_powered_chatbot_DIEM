"""Provider LLM per il sistema RAG DIEM.

Supporta HuggingFace, Ollama e Groq come backend per i modelli linguistici.
Implementa una logica di fallback: se il provider primario (Groq) non e'
disponibile o la chiave API non e' valida, il sistema ricade automaticamente
su Ollama con il modello di fallback configurato.

Per i modelli thinking (Nemotron, Qwen3, DeepSeek-R1) via Ollama,
reasoning e' disabilitato per evitare che i thinking tokens producano
risposte vuote quando combinati con tool calling.
"""

import logging

from langchain_core.language_models.chat_models import BaseChatModel

from src.config.settings import LLMConfig

logger = logging.getLogger(__name__)


def _validate_groq_key(api_key: str, label: str, config: LLMConfig) -> bool:
    """Verifica che una chiave API Groq sia valida con una chiamata di test.

    Args:
        api_key: La chiave API da validare.
        label: Etichetta per il logging (es. 'GROQ_CHAT_API_KEY').
        config: Configurazione LLM per modello e token limit di validazione.

    Returns:
        True se la chiave e' valida, False altrimenti.
    """
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage

        test_llm = ChatGroq(
            model=config.groq_validation_model,
            temperature=0.0,
            max_tokens=config.groq_validation_max_tokens,
            api_key=api_key,
        )
        test_llm.invoke([HumanMessage(content="test")])
        logger.info("Validazione %s: OK", label)
        return True
    except Exception as e:
        logger.warning("Validazione %s fallita: %s", label, e)
        return False


def _create_fallback_chat_model(config: LLMConfig) -> BaseChatModel:
    """Crea un ChatModel di fallback su Ollama con il modello configurato.

    Args:
        config: Configurazione LLM con modello e URL di fallback.

    Returns:
        Istanza di BaseChatModel Ollama di fallback.

    Raises:
        RuntimeError: Se anche il fallback Ollama non e' raggiungibile.
    """
    from langchain_ollama import ChatOllama

    try:
        chat_model = ChatOllama(
            model=config.fallback_model,
            temperature=config.temperature,
            num_predict=config.max_tokens,
            base_url=config.fallback_base_url,
        )
        logger.info(
            "CHAT MODEL ATTIVO: Ollama/%s su %s (FALLBACK)",
            config.fallback_model,
            config.fallback_base_url,
        )
        return chat_model
    except Exception as e:
        raise RuntimeError(
            f"Impossibile istanziare il fallback Ollama ({config.fallback_model} "
            f"su {config.fallback_base_url}): {e}. "
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

        if not _validate_groq_key(api_key, key_label, config):
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


def create_rewriter_llm(config: LLMConfig) -> BaseChatModel:
    """Crea un LLM dedicato al query rewriting.

    Se rewriter_provider e' configurato, usa un LLM separato (Groq).
    Altrimenti riutilizza il provider principale con temperature configurata.
    In caso di chiavi mancanti o invalide, ricade sul modello di fallback.

    Args:
        config: Configurazione LLM completa con parametri del rewriter.

    Returns:
        Istanza di BaseChatModel configurata per il rewriting.

    Raises:
        ValueError: Se il provider del rewriter non e' supportato.
        RuntimeError: Se nessun provider e' disponibile (incluso il fallback).
    """
    provider = config.rewriter_provider.strip().lower() if config.rewriter_provider else ""

    if not provider:
        main_provider = config.provider.lower()
        logger.info(
            "rewriter_provider non configurato. "
            "Uso LLM principale (%s/%s) con temperature=%.2f",
            main_provider.upper(),
            config.model_name,
            config.rewriter_temperature,
        )

        if main_provider == "ollama":
            from langchain_ollama import ChatOllama

            ollama_kwargs = dict(
                model=config.model_name,
                temperature=config.rewriter_temperature,
                num_predict=config.rewriter_max_tokens,
                base_url=config.ollama_base_url,
            )

            llm = ChatOllama(**ollama_kwargs)
            logger.info(
                "REWRITER ATTIVO: Ollama/%s su %s (temperature=%.2f)",
                config.model_name,
                config.ollama_base_url,
                config.rewriter_temperature,
            )
            return llm

        if main_provider == "huggingface":
            from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

            if not config.huggingface_api_token:
                logger.warning(
                    "HUGGINGFACEHUB_API_TOKEN mancante per rewriter. Attivazione fallback."
                )
                return _create_rewriter_fallback(config)

            llm = ChatHuggingFace(
                llm=HuggingFaceEndpoint(
                    repo_id=config.model_name,
                    temperature=config.rewriter_hf_temperature,
                    max_new_tokens=config.rewriter_max_tokens,
                    huggingfacehub_api_token=config.huggingface_api_token,
                    task="text-generation",
                )
            )
            logger.info(
                "REWRITER ATTIVO: HuggingFace/%s (temperature=%.2f)",
                config.model_name,
                config.rewriter_hf_temperature,
            )
            return llm

        if main_provider == "groq":
            from langchain_groq import ChatGroq

            api_key = config.groq_rewriter_api_key
            if not api_key:
                logger.warning(
                    "GROQ_REWRITER_API_KEY mancante. Attivazione fallback."
                )
                return _create_rewriter_fallback(config)

            logger.info("Validazione GROQ_REWRITER_API_KEY per Rewriter in corso...")
            if not _validate_groq_key(api_key, "GROQ_REWRITER_API_KEY", config):
                logger.warning(
                    "REWRITER: GROQ_REWRITER_API_KEY non valida. Attivazione fallback."
                )
                return _create_rewriter_fallback(config)

            try:
                llm = ChatGroq(
                    model=config.model_name,
                    temperature=config.rewriter_temperature,
                    max_tokens=config.rewriter_max_tokens,
                    api_key=api_key,
                )
                logger.info(
                    "REWRITER ATTIVO: Groq/%s (temperature=%.2f)",
                    config.model_name,
                    config.rewriter_temperature,
                )
                return llm
            except Exception as e:
                logger.error(
                    "Errore istanziazione rewriter ChatGroq: %s. Attivazione fallback.",
                    e,
                )
                return _create_rewriter_fallback(config)

        return create_chat_model(config)

    if provider == "groq":
        from langchain_groq import ChatGroq

        model = config.rewriter_model
        api_key = config.groq_rewriter_api_key or ""

        if not api_key:
            logger.warning(
                "GROQ_REWRITER_API_KEY mancante per rewriter esplicito. Attivazione fallback."
            )
            return _create_rewriter_fallback(config)

        logger.info("Validazione GROQ_REWRITER_API_KEY per Rewriter dedicato in corso...")
        if not _validate_groq_key(api_key, "GROQ_REWRITER_API_KEY", config):
            logger.warning(
                "REWRITER: GROQ_REWRITER_API_KEY non valida (401 Unauthorized). "
                "Attivazione fallback."
            )
            return _create_rewriter_fallback(config)

        try:
            llm = ChatGroq(
                model=model,
                temperature=config.rewriter_temperature,
                max_tokens=config.rewriter_max_tokens,
                api_key=api_key,
            )
            logger.info(
                "REWRITER ATTIVO: Groq/%s dedicato (temperature=%.2f)",
                model,
                config.rewriter_temperature,
            )
            return llm
        except Exception as e:
            logger.error(
                "Errore istanziazione rewriter ChatGroq (%s): %s. Attivazione fallback.",
                model,
                e,
            )
            return _create_rewriter_fallback(config)

    raise ValueError(
        f"rewriter_provider non supportato: '{provider}'. "
        "Valori ammessi: 'groq' (oppure lasciare vuoto per usare il LLM principale)."
    )


def _create_rewriter_fallback(config: LLMConfig) -> BaseChatModel:
    """Crea un LLM di fallback per il rewriter su Ollama.

    Args:
        config: Configurazione LLM per modello e URL di fallback.

    Returns:
        Istanza di BaseChatModel Ollama di fallback per il rewriting.
    """
    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        model=config.fallback_model,
        temperature=config.rewriter_temperature,
        num_predict=config.rewriter_max_tokens,
        base_url=config.fallback_base_url,
    )
    logger.info(
        "REWRITER ATTIVO: Ollama/%s su %s (FALLBACK, temperature=%.2f)",
        config.fallback_model,
        config.fallback_base_url,
        config.rewriter_temperature,
    )
    return llm