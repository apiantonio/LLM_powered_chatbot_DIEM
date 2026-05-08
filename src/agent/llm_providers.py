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


# ============================================================
# BACKWARD COMPATIBILITY ALIAS
# ============================================================
# Per i moduli che ancora importano HuggingFaceLLMProvider,
# forniamo un alias che mappa al nuovo approccio.
# Rimuovere dopo aver aggiornato tutti i consumer.

class HuggingFaceLLMProvider:
    """
    DEPRECATED — Usare create_chat_model(config) al suo posto.
    
    Mantiene la compatibilità con il codice che usa:
        provider = HuggingFaceLLMProvider(config)
        provider.invoke(prompt)
        provider.langchain_chat_model
    """
    
    def __init__(self, config: LLMConfig):
        import warnings
        warnings.warn(
            "HuggingFaceLLMProvider è deprecato. "
            "Usare create_chat_model(config) per ottenere un ChatHuggingFace diretto.",
            DeprecationWarning, stacklevel=2,
        )
        self._chat = create_chat_model(config)
    
    def invoke(self, prompt: str) -> str:
        return self._chat.invoke(prompt).content
    
    def invoke_with_messages(self, messages: list) -> str:
        return self._chat.invoke(messages).content
    
    def supports_tool_calling(self) -> bool:
        return False
    
    @property
    def langchain_chat_model(self) -> ChatHuggingFace:
        return self._chat