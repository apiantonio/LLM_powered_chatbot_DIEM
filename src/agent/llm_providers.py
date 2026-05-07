"""
Adapter concreto per HuggingFace Endpoint.

Pattern: Adapter (GoF) — wrappa l'API di langchain_huggingface
dietro l'interfaccia LLMProvider definita nel core.

KPI Impact: Ingegneria del Software. Cambiare modello richiede
solo la modifica di LLMConfig.model_name in settings.py.
"""

import logging
from typing import List

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from interfaces.agent_interfaces import LLMProvider
from config.settings import LLMConfig

logger = logging.getLogger(__name__)


class HuggingFaceLLMProvider(LLMProvider):
    """Implementazione concreta per HuggingFace Inference API."""
    
    def __init__(self, config: LLMConfig):
        if not config.huggingface_api_token:
            raise ValueError(
                "HUGGINGFACEHUB_API_TOKEN non configurato. "
                "Impostalo come variabile d'ambiente."
            )
        
        self._chat = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                repo_id=config.model_name,
                temperature=config.temperature,
                max_new_tokens=config.max_tokens,
                huggingfacehub_api_token=config.huggingface_api_token,
            )
        )
        logger.info(f"HuggingFace LLM inizializzato: {config.model_name}")
    
    def invoke(self, prompt: str) -> str:
        response = self._chat.invoke(prompt)
        return response.content
    
    def invoke_with_messages(self, messages: List[dict]) -> str:
        response = self._chat.invoke(messages)
        return response.content
    
    def supports_tool_calling(self) -> bool:
        return False  # HF Inference API non supporta nativamente tool calling
    
    @property
    def langchain_chat_model(self) -> ChatHuggingFace:
        """Espone il modello LangChain sottostante per le LCEL chains."""
        return self._chat
