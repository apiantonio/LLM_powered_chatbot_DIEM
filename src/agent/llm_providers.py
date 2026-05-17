"""
Factory per l'istanziazione del Chat Model LangChain.

v4: Aggiunto provider "groq" e factory create_rewriter_llm().
Provider supportati: 'huggingface', 'ollama', 'groq'
"""

import os
import logging
from langchain_core.language_models.chat_models import BaseChatModel

from config.settings import LLMConfig

logger = logging.getLogger(__name__)


def create_chat_model(config: LLMConfig) -> BaseChatModel:
    """Factory che costruisce il ChatModel LangChain in base al provider."""
    provider = config.provider.lower()
    logger.info(f"Inizializzazione ChatModel. Provider: {provider.upper()}")

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

        if not config.huggingface_api_token:
            raise ValueError(
                "HUGGINGFACEHUB_API_TOKEN non configurato. "
                "Impostalo come variabile d'ambiente."
            )

        chat_model = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                repo_id=config.model_name,
                temperature=config.temperature,
                max_new_tokens=config.max_tokens,
                huggingfacehub_api_token=config.huggingface_api_token,
                task="text-generation"
            )
        )
        logger.info(f"ChatHuggingFace istanziato: {config.model_name}")
        return chat_model

    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        chat_model = ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            num_predict=config.max_tokens,
            base_url=config.ollama_base_url
        )
        logger.info(f"ChatOllama istanziato: {config.model_name} su {config.ollama_base_url}")
        return chat_model

    elif provider == "groq":
        from langchain_groq import ChatGroq

        api_key = config.groq_api_key
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY non configurata. "
                "Ottieni una API key gratuita su https://console.groq.com/keys"
            )

        chat_model = ChatGroq(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=api_key,
        )
        logger.info(f"ChatGroq istanziato: {config.model_name}")
        return chat_model

    else:
        raise ValueError(
            f"Provider LLM non supportato: '{provider}'. "
            "Scegliere tra 'huggingface', 'ollama' o 'groq'."
        )


def create_rewriter_llm(fallback_config: LLMConfig) -> BaseChatModel:
    """
    Crea il LLM dedicato per il QueryOptimizer (rewrite + multiquery).

    Legge REWRITER_PROVIDER e REWRITER_MODEL dall'ambiente.
    Se non configurati, usa il LLM principale con temperature=0.0.

    Il rewriter usa SEMPRE temperature=0.0 e max_tokens=256.
    """
    provider = os.getenv("REWRITER_PROVIDER", "").strip().lower()

    if not provider:
        main_provider = fallback_config.provider.lower()
        logger.info(
            f"REWRITER_PROVIDER non configurato. "
            f"Uso LLM principale ({fallback_config.model_name}) con temperature=0.0"
        )

        if main_provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=fallback_config.model_name,
                temperature=0.0,
                num_predict=256,
                base_url=fallback_config.ollama_base_url,
            )
        elif main_provider == "huggingface":
            from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
            return ChatHuggingFace(
                llm=HuggingFaceEndpoint(
                    repo_id=fallback_config.model_name,
                    temperature=0.01,
                    max_new_tokens=256,
                    huggingfacehub_api_token=fallback_config.huggingface_api_token,
                    task="text-generation",
                )
            )
        else:
            return create_chat_model(fallback_config)

    if provider == "groq":
        from langchain_groq import ChatGroq

        model = os.getenv("REWRITER_MODEL", "llama-3.3-70b-versatile").strip()
        api_key = os.getenv("GROQ_API_KEY", "").strip()

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY richiesta per il rewriter Groq. "
                "Registrati gratis su https://console.groq.com/keys"
            )

        llm = ChatGroq(
            model=model,
            temperature=0.0,
            max_tokens=256,
            api_key=api_key,
        )
        logger.info(f"Rewriter LLM: ChatGroq ({model}) — temperature=0.0")
        return llm

    else:
        raise ValueError(
            f"REWRITER_PROVIDER non supportato: '{provider}'. "
            "Valori ammessi: 'groq' (oppure lasciare vuoto per usare il LLM principale)."
        )