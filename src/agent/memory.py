"""
Gestione della memoria conversazionale per l'Agente RAG DIEM.

Questo modulo implementa la memoria a breve termine dell'agente:
mantiene lo storico dei messaggi e lo inietta nel flusso create_agent
per consentire la riscrittura contestuale delle query.

REFACTORING APPLICATO:
  - get_history_summary() ora restituisce TUTTI i turni (non solo 3)
    per il log dell'interazione.
  - Mantenuta la stessa interfaccia pubblica per compatibilità.

Pattern: Strategy (GoF), Facade (GoF).
"""

import logging
import re
from typing import List, Optional
from dataclasses import dataclass

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
)

logger = logging.getLogger(__name__)


# Reminder iniettato alla fine di ogni messaggio utente
_RETRIEVAL_REMINDER = (
    "\n\n[SISTEMA: Se questa domanda riguarda il DIEM, usa un tool di ricerca "
    "per trovare le informazioni necessarie. Se il tool restituisce un errore, "
    "NON riprovare lo stesso tool — usa un tool alternativo oppure comunica "
    "all'utente che l'informazione non è al momento disponibile.]"
)

# Query che NON necessitano di retrieval
_META_PATTERNS = [
    r"^(ciao|salve|buongiorno|buonasera|hey|hi|hello)\b",
    r"^grazie",
    r"^(come stai|come va|chi sei|cosa sai fare)",
]


def _is_meta_query(query: str) -> bool:
    """Rileva saluti e meta-domande che non richiedono retrieval."""
    q = query.strip().lower()
    return any(re.match(p, q) for p in _META_PATTERNS)


@dataclass
class ConversationTurn:
    """Singolo turno di conversazione (domanda + risposta)."""
    user_message: str
    assistant_message: str
    turn_number: int


class ConversationMemory:
    """
    Memoria conversazionale in-memory con sliding window.
    """

    def __init__(
        self,
        max_turns: int = 10,
        max_tokens: Optional[int] = None,
    ):
        self._max_turns = max_turns
        self._max_tokens = max_tokens
        self._turns: List[ConversationTurn] = []
        self._pending_user_message: str = ""
        self._turn_counter: int = 0

    @property
    def turn_count(self) -> int:
        return self._turn_counter

    @property
    def is_empty(self) -> bool:
        return self._turn_counter == 0

    def add_user_message(self, message: str) -> int:
        self._turn_counter += 1
        self._pending_user_message = message
        return self._turn_counter

    def add_assistant_message(self, message: str) -> None:
        user_msg = getattr(self, '_pending_user_message', '')

        self._turns.append(ConversationTurn(
            user_message=user_msg,
            assistant_message=message,
            turn_number=self._turn_counter,
        ))
        self._pending_user_message = ""

        if len(self._turns) > self._max_turns:
            excess = len(self._turns) - self._max_turns
            self._turns = self._turns[excess:]

        logger.debug(f"Memoria aggiornata: {len(self._turns)} turni")

    def get_messages_for_agent(self, current_query: str) -> list:
        """
        Costruisce la lista messaggi per create_agent.invoke().
        Inietta il RETRIEVAL_REMINDER per query non-meta.
        """
        messages = []

        for turn in self._turns:
            messages.append({"role": "user", "content": turn.user_message})
            messages.append({"role": "assistant", "content": turn.assistant_message})

        if _is_meta_query(current_query):
            messages.append({"role": "user", "content": current_query})
        else:
            messages.append({
                "role": "user",
                "content": current_query + _RETRIEVAL_REMINDER,
            })

        return messages

    def get_langchain_history(self) -> List[BaseMessage]:
        """
        Restituisce lo storico come lista di BaseMessage LangChain.
        Usato dal QueryOptimizer per la riscrittura contestuale.
        NON include il messaggio corrente.
        """
        history_messages: List[BaseMessage] = []
        for turn in self._turns:
            history_messages.append(HumanMessage(content=turn.user_message))
            history_messages.append(AIMessage(content=turn.assistant_message))
        return history_messages

    def get_history_summary(self) -> str:
        """
        Restituisce un riepilogo testuale dello storico completo.

        AGGIORNATO: ora restituisce TUTTI i turni (non solo 3)
        per il log dell'interazione. Ogni turno mostra la domanda
        utente e la risposta (troncata a 150 char per leggibilità).
        """
        if not self._turns:
            return "(nessuno storico)"

        lines = []
        for turn in self._turns:
            user_preview = turn.user_message[:150]
            if len(turn.user_message) > 150:
                user_preview += "..."
            assistant_preview = turn.assistant_message[:150]
            if len(turn.assistant_message) > 150:
                assistant_preview += "..."
            lines.append(f"  [Turno {turn.turn_number}] Utente: {user_preview}")
            lines.append(f"                Agente: {assistant_preview}")

        return "\n".join(lines)

    def clear(self) -> None:
        self._turns.clear()
        self._pending_user_message = ""
        self._turn_counter = 0
        logger.info("Memoria conversazionale resettata")


# ============================================================
# FACTORY
# ============================================================

def create_conversation_memory(
    max_turns: int = 10,
    max_tokens: Optional[int] = None,
) -> ConversationMemory:
    """Factory Method: crea la strategia di memoria appropriata."""
    memory = ConversationMemory(
        max_turns=max_turns,
        max_tokens=max_tokens,
    )
    logger.info(f"ConversationMemory creata: max_turns={max_turns}")
    return memory