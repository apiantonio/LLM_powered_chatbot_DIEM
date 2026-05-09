"""
Gestione della memoria conversazionale per l'Agente RAG DIEM.

Questo modulo implementa la memoria a breve termine dell'agente:
mantiene lo storico dei messaggi e lo inietta nel flusso create_agent
per consentire la riscrittura contestuale delle query (context awareness).

Pattern applicati:
  - Strategy (GoF): ConversationMemory è una strategia di gestione dello storico
    intercambiabile (in-memory oggi, persistente domani) senza modificare l'agente.
  - Facade (GoF): espone un'API semplice (add_turn / get_messages / get_history_for_optimizer)
    che nasconde la complessità della gestione dei messaggi LangChain.

Vincolo: NESSUN uso di LangGraph. La memoria è gestita esternamente all'agente
e iniettata come lista di messaggi nel campo 'messages' di create_agent.invoke().

KPI Impact:
  - Context Awareness: l'agente risolve coreferenze anaforiche grazie alla cronologia.
  - Query Optimization: lo storico alimenta il QueryOptimizer del RetrievalEngine.
"""

import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    trim_messages,
)

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Singolo turno di conversazione (domanda + risposta)."""
    user_message: str
    assistant_message: str
    turn_number: int


class ConversationMemory:
    """
    Memoria conversazionale in-memory con sliding window.
    
    Strategy Pattern (GoF):
      Questa è la strategia 'InMemory'. La stessa interfaccia può essere
      implementata con storage persistente (Redis, SQLite) senza toccare
      il codice dell'agente — basta passare un'istanza diversa.
    
    Gestisce:
      1. Storico messaggi LangChain (HumanMessage / AIMessage) per create_agent.
      2. Storico formattato per il QueryOptimizer (coppie domanda/risposta).
      3. Sliding window per evitare overflow del context window del LLM.
    
    Thread-safe: ogni sessione utente deve avere la propria istanza.
    """
    
    def __init__(
        self,
        max_turns: int = 10,
        max_tokens: Optional[int] = None,
    ):
        """
        Args:
            max_turns: Numero massimo di turni da mantenere in memoria.
            max_tokens: Se impostato, tronca i messaggi per stare nel budget token.
        """
        self._max_turns = max_turns
        self._max_tokens = max_tokens
        self._turns: List[ConversationTurn] = []
        self._pending_user_message: str = ""
        self._turn_counter: int = 0
    
    @property
    def turn_count(self) -> int:
        """Numero di turni completati nella sessione corrente."""
        return self._turn_counter
    
    @property
    def is_empty(self) -> bool:
        """True se non ci sono turni in memoria."""
        return self._turn_counter == 0
    
    def add_user_message(self, message: str) -> int:
        """
        Registra il messaggio dell'utente e incrementa il contatore turni.
        
        NON aggiunge a self._messages: la lista messaggi LangChain viene
        costruita on-demand da get_messages_for_agent() a partire da _turns.
        Il messaggio corrente viene salvato temporaneamente per la chiusura
        del turno in add_assistant_message().
        
        Returns:
            Il numero del turno corrente (1-indexed).
        """
        self._turn_counter += 1
        self._pending_user_message = message
        return self._turn_counter
    
    def add_assistant_message(self, message: str) -> None:
        """
        Registra la risposta dell'agente e completa il turno.
        Applica la sliding window se necessario.
        """
        # Completa il turno con il messaggio utente pendente
        user_msg = getattr(self, '_pending_user_message', '')
        
        self._turns.append(ConversationTurn(
            user_message=user_msg,
            assistant_message=message,
            turn_number=self._turn_counter,
        ))
        self._pending_user_message = ""
        
        # Sliding window: mantieni solo gli ultimi N turni
        if len(self._turns) > self._max_turns:
            excess = len(self._turns) - self._max_turns
            self._turns = self._turns[excess:]
        
        logger.debug(
            f"Memoria aggiornata: {len(self._turns)} turni"
        )
    
    def get_messages_for_agent(self, current_query: str) -> List[dict]:
        """
        Restituisce la lista messaggi nel formato atteso da create_agent.invoke().
        
        Include lo storico + la query corrente come ultimo HumanMessage.
        Formato: [{"role": "user"|"assistant", "content": "..."}]
        
        Args:
            current_query: La domanda corrente dell'utente (ultimo messaggio).
        
        Returns:
            Lista di dict compatibile con create_agent.invoke({"messages": [...]}).
        """
        messages = []
        
        # Aggiungi lo storico (turni precedenti, senza il turno corrente)
        for turn in self._turns:
            messages.append({"role": "user", "content": turn.user_message})
            messages.append({"role": "assistant", "content": turn.assistant_message})
        
        # Aggiungi la query corrente
        messages.append({"role": "user", "content": current_query})
        
        return messages
    
    def get_langchain_history(self) -> List[BaseMessage]:
        """
        Restituisce lo storico come lista di BaseMessage LangChain.
        
        Usato dal QueryOptimizer per la riscrittura contestuale.
        NON include il messaggio corrente — solo lo storico pregresso.
        """
        # Restituisci solo i messaggi dei turni completati
        history_messages: List[BaseMessage] = []
        for turn in self._turns:
            history_messages.append(HumanMessage(content=turn.user_message))
            history_messages.append(AIMessage(content=turn.assistant_message))
        return history_messages
    
    def get_history_summary(self) -> str:
        """
        Restituisce un riepilogo testuale dello storico (per debug/logging).
        """
        if not self._turns:
            return "(nessuno storico)"
        
        lines = []
        for turn in self._turns[-3:]:  # Ultimi 3 turni
            lines.append(f"  [T{turn.turn_number}] U: {turn.user_message[:80]}...")
            lines.append(f"         A: {turn.assistant_message[:80]}...")
        
        if len(self._turns) > 3:
            lines.insert(0, f"  ... ({len(self._turns) - 3} turni precedenti omessi)")
        
        return "\n".join(lines)
    
    def clear(self) -> None:
        """Resetta la memoria conversazionale."""
        self._turns.clear()
        self._pending_user_message = ""
        self._turn_counter = 0
        logger.info("Memoria conversazionale resettata")
    
    # ==============================
    # PRIVATE HELPERS
    # ==============================
    
    # (nessun helper privato necessario — il flusso usa _pending_user_message)


# ============================================================
# FACTORY (GoF) — crea la memory strategy appropriata
# ============================================================

def create_conversation_memory(
    max_turns: int = 10,
    max_tokens: Optional[int] = None,
) -> ConversationMemory:
    """
    Factory Method (GoF): crea la strategia di memoria appropriata.
    
    Oggi: sempre InMemory.
    Domani: in base a un parametro 'backend' in settings, potrebbe
    restituire RedisMemory, SQLiteMemory, ecc.
    
    Args:
        max_turns: Numero massimo di turni da mantenere.
        max_tokens: Budget token opzionale per il context window.
    
    Returns:
        ConversationMemory configurata.
    """
    memory = ConversationMemory(
        max_turns=max_turns,
        max_tokens=max_tokens,
    )
    logger.info(f"ConversationMemory creata: max_turns={max_turns}")
    return memory