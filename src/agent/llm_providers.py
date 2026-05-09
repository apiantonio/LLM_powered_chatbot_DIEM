"""
Factory per l'istanziazione del Chat Model LangChain.

Supporta nativamente l'alternanza tra provider multipli (HuggingFace, Ollama)
tramite il parametro 'provider' in LLMConfig.

Restituisce sempre un BaseChatModel compatibile con le LCEL chains.
"""

import logging
from langchain_core.language_models.chat_models import BaseChatModel

from config.settings import LLMConfig

logger = logging.getLogger(__name__)

def create_chat_model(config: LLMConfig) -> BaseChatModel:
    """
    Factory che costruisce il ChatModel LangChain in base al provider scelto.
    
    Usa il lazy loading per gli import specifici dei provider per evitare
    crash se una specifica SDK non è installata nell'ambiente corrente.
    
    Args:
        config: Configurazione LLM da settings.py.
    
    Returns:
        Istanza concreta di BaseChatModel (es. ChatHuggingFace o ChatOllama)
        pronta per l'agente e le chain LCEL.
    
    Raises:
        ValueError: Se il provider non è supportato o mancano credenziali.
    """
    provider = config.provider.lower()
    logger.info(f"Inizializzazione ChatModel. Provider selezionato: {provider.upper()}")

    # ==========================================
    # PROVIDER: HUGGING FACE
    # ==========================================
    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
        
        if not config.huggingface_api_token:
            raise ValueError(
                "HUGGINGFACEHUB_API_TOKEN non configurato per il provider HuggingFace. "
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

    # ==========================================
    # PROVIDER: OLLAMA (LOCALE)
    # ==========================================
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

    # ==========================================
    # FALLBACK: PROVIDER SCONOSCIUTO
    # ==========================================
    else:
        raise ValueError(
            f"Provider LLM non supportato: '{provider}'. "
            "Scegliere tra 'huggingface' o 'ollama'."
        )