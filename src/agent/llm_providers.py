"""
Factory per l'istanziazione del Chat Model LangChain.

Post-refactoring LCEL: eliminato il wrapper HuggingFaceLLMProvider.
ChatHuggingFace è già un BaseChatModel LangChain — supporta nativamente
l'operatore | per le LCEL chains e non richiede un adapter aggiuntivo.

Cambiare modello richiede solo la modifica di LLMConfig in settings.py.
"""

import logging

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from config.settings import LLMConfig

logger = logging.getLogger(__name__)


def create_chat_model(config: LLMConfig) -> ChatHuggingFace:
    """
    Factory che costruisce il ChatModel LangChain dal config.
    
    Il modello restituito è direttamente componibile in chain LCEL:
        chain = prompt | create_chat_model(config) | StrOutputParser()
    
    Args:
        config: Configurazione LLM da settings.py.
    
    Returns:
        ChatHuggingFace pronto per LCEL.
    
    Raises:
        ValueError: Se il token HuggingFace non è configurato.
    """
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
        )
    )
    logger.info(f"ChatModel LangChain inizializzato: {config.model_name}")
    return chat_model