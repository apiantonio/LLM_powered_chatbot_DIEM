from typing import Dict
from langchain_core.chat_history import InMemoryChatMessageHistory

# Registry in-memory per associare una chat history a uno specifico session_id
_store: Dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    Factory method per ottenere o inizializzare la memoria conversazionale nativa associata
    a un identificativo di sessione utente.
    Utilizza InMemoryChatMessageHistory come prescritto dal reference stack.
    
    Args:
        session_id (str): Identificativo univoco della sessione di chat dell'utente.
        
    Returns:
        InMemoryChatMessageHistory: L'oggetto per la gestione in-memory dello storico messaggi.
    """
    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory()
    return _store[session_id]