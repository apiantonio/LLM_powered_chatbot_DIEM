"""
Gestione della memoria conversazionale per l'Agente RAG DIEM.

REFACTORING v4:
  - get_langchain_history() restituisce SOLO l'ultimo turno (1 coppia
    HumanMessage + AIMessage) per il rewriter. Il contesto di
    coreferenza si basa esclusivamente sull'ultima interazione.
  - Invariata l'architettura a due stadi (similarità + summarization)
"""

import logging
import re
import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
)
from langchain_classic.memory import ConversationSummaryBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


# Reminder compatto per 7B (meno token = meno confusione)
_RETRIEVAL_REMINDER = (
    "\n\n[SISTEMA: invoke a search tool. "
    "Pass the user's query INTACT to the tool, without abbreviating it.]"
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
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


class SmartConversationMemory:
    """
    Memoria conversazionale intelligente a due stadi.

    Stadio 1 — Filtraggio per Similarità coseno
    Stadio 2 — Summarization con Token Budget
    """

    def __init__(
        self,
        llm_for_summary,
        embedding_model: HuggingFaceEmbeddings,
        max_turns: int = 10,
        similarity_threshold: float = 0.3,
        max_token_limit: int = 1500,
    ):
        self._max_turns = max_turns
        self._similarity_threshold = similarity_threshold
        self._max_token_limit = max_token_limit
        self._turns: List[ConversationTurn] = []
        self._pending_user_message: str = ""
        self._turn_counter: int = 0
        self._embedding_model = embedding_model
        self._llm_for_summary = llm_for_summary

        logger.info(
            f"SmartConversationMemory inizializzata: "
            f"max_turns={max_turns}, similarity_threshold={similarity_threshold}, "
            f"max_token_limit={max_token_limit}"
        )

    @property
    def turn_count(self) -> int:
        return self._turn_counter

    @property
    def is_empty(self) -> bool:
        return self._turn_counter == 0

    def add_user_message(self, message: str) -> int:
        """Registra il messaggio utente e avanza il contatore turno."""
        self._turn_counter += 1
        self._pending_user_message = message
        return self._turn_counter

    def add_assistant_message(self, message: str) -> None:
        """Registra la risposta, completa il turno, calcola embedding."""
        user_msg = getattr(self, '_pending_user_message', '')

        turn_text = f"{user_msg} {message}"
        try:
            turn_embedding = np.array(
                self._embedding_model.embed_query(turn_text)
            )
        except Exception as e:
            logger.warning(f"Errore embedding turno: {e}. Turno senza embedding.")
            turn_embedding = None

        self._turns.append(ConversationTurn(
            user_message=user_msg,
            assistant_message=message,
            turn_number=self._turn_counter,
            embedding=turn_embedding,
        ))
        self._pending_user_message = ""

        # Sliding window
        if len(self._turns) > self._max_turns:
            excess = len(self._turns) - self._max_turns
            self._turns = self._turns[excess:]

        logger.debug(f"Memoria aggiornata: {len(self._turns)} turni")

    def _filter_turns_by_similarity(self, query: str) -> List[ConversationTurn]:
        """STADIO 1: Filtra i turni per similarità coseno."""
        if not self._turns:
            return []

        try:
            query_embedding = np.array(
                self._embedding_model.embed_query(query)
            )
        except Exception as e:
            logger.warning(f"Errore embedding query: {e}. Restituisco tutti i turni.")
            return list(self._turns)

        filtered = []
        for turn in self._turns:
            if turn.embedding is None:
                filtered.append(turn)
                continue

            similarity = float(
                np.dot(query_embedding, turn.embedding)
                / (np.linalg.norm(query_embedding) * np.linalg.norm(turn.embedding) + 1e-10)
            )

            if similarity >= self._similarity_threshold:
                filtered.append(turn)
                logger.debug(
                    f"Turno #{turn.turn_number} sopra soglia: "
                    f"sim={similarity:.4f} >= {self._similarity_threshold}"
                )
            else:
                logger.debug(
                    f"Turno #{turn.turn_number} SCARTATO: "
                    f"sim={similarity:.4f} < {self._similarity_threshold}"
                )

        logger.info(
            f"Filtraggio similarità: {len(filtered)}/{len(self._turns)} turni "
            f"sopra soglia ({self._similarity_threshold})"
        )
        return filtered

    def _summarize_if_needed(
        self, filtered_turns: List[ConversationTurn]
    ) -> List[BaseMessage]:
        """STADIO 2: Summarization con ConversationSummaryBufferMemory."""
        if not filtered_turns:
            return []

        summary_memory = ConversationSummaryBufferMemory(
            llm=self._llm_for_summary,
            max_token_limit=self._max_token_limit,
            return_messages=True,
            memory_key="history",
            human_prefix="Utente",
            ai_prefix="Assistente",
        )

        for turn in filtered_turns:
            summary_memory.save_context(
                {"input": turn.user_message},
                {"output": turn.assistant_message},
            )

        memory_variables = summary_memory.load_memory_variables({})
        messages = memory_variables.get("history", [])

        logger.info(
            f"Summarization: {len(filtered_turns)} turni → "
            f"{len(messages)} messaggi in output"
        )
        return messages

    def find_exact_match(self, query: str) -> Optional[str]:
        """
        Cerca se la query è già stata posta in questa sessione calcolando
        una "fingerprint" normalizzata.
        """
        if not self._turns:
            return None

        def get_fingerprint(text: str) -> str:
            text = text.lower()
            fingerprint = re.sub(r'[\W_]+', '', text)
            return fingerprint

        query_fingerprint = get_fingerprint(query)

        if not query_fingerprint:
            return None

        for turn in reversed(self._turns):
            if get_fingerprint(turn.user_message) == query_fingerprint:
                logger.info(f"🎯 Cache HIT! Impronta '{query_fingerprint}' trovata al Turno #{turn.turn_number}")
                return turn.assistant_message

        return None

    def get_messages_for_agent(self, current_query: str) -> list:
        """
        Costruisce la lista messaggi per create_agent.invoke().

        Flusso a due stadi + iniezione RETRIEVAL_REMINDER per query non-meta.
        """
        messages = []

        if self._turns:
            filtered_turns = self._filter_turns_by_similarity(current_query)

            if filtered_turns:
                summarized_messages = self._summarize_if_needed(filtered_turns)

                for msg in summarized_messages:
                    if isinstance(msg, HumanMessage):
                        messages.append({"role": "user", "content": msg.content})
                    elif isinstance(msg, AIMessage):
                        messages.append({"role": "assistant", "content": msg.content})
                    elif hasattr(msg, 'type'):
                        if msg.type == 'human':
                            messages.append({"role": "user", "content": msg.content})
                        elif msg.type == 'ai':
                            messages.append({"role": "assistant", "content": msg.content})
                        elif msg.type == 'system':
                            messages.append({"role": "assistant", "content": msg.content})

        # Aggiungi la query corrente
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

        v4: Restituisce SOLO l'ultimo turno completato (1 coppia
        HumanMessage + AIMessage). Il rewriter ha bisogno esclusivamente
        dell'ultima interazione per risolvere le coreferenze.
        """
        if not self._turns:
            return []

        last_turn = self._turns[-1]
        history = [
            HumanMessage(content=last_turn.user_message),
            AIMessage(content=last_turn.assistant_message[:300]
                      + ("..." if len(last_turn.assistant_message) > 300 else "")),
        ]
        return history

    def get_history_summary(self) -> str:
        """Restituisce un riepilogo testuale dello storico completo."""
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
            lines.append(f"                Assistente: {assistant_preview}")

        return "\n".join(lines)

    def clear(self) -> None:
        """Resetta la memoria conversazionale."""
        self._turns.clear()
        self._pending_user_message = ""
        self._turn_counter = 0
        logger.info("SmartConversationMemory resettata")


# ============================================================
# FACTORY
# ============================================================

def create_conversation_memory(
    max_turns: int = 10,
    max_tokens: Optional[int] = None,
    llm_for_summary=None,
    embedding_model: Optional[HuggingFaceEmbeddings] = None,
    similarity_threshold: float = 0.45,
    max_token_limit: int = 1500,
) -> SmartConversationMemory:
    """Factory Method: crea la SmartConversationMemory."""
    if llm_for_summary is None:
        from config.settings import load_settings
        from agent.llm_providers import create_chat_model
        settings = load_settings()
        llm_for_summary = create_chat_model(settings.llm)
        logger.info("SmartMemory: LLM per summarization creato da settings")

    if embedding_model is None:
        from config.settings import load_settings
        settings = load_settings()
        embedding_model = HuggingFaceEmbeddings(
            model_name=settings.embedding.model_name,
            encode_kwargs={"normalize_embeddings": settings.embedding.normalize_embeddings},
        )
        logger.info(f"SmartMemory: Embedding model creato: {settings.embedding.model_name}")

    effective_token_limit = max_token_limit
    if max_tokens is not None:
        effective_token_limit = max_tokens

    memory = SmartConversationMemory(
        llm_for_summary=llm_for_summary,
        embedding_model=embedding_model,
        max_turns=max_turns,
        similarity_threshold=similarity_threshold,
        max_token_limit=effective_token_limit,
    )
    logger.info(
        f"SmartConversationMemory creata: max_turns={max_turns}, "
        f"similarity_threshold={similarity_threshold}, "
        f"max_token_limit={effective_token_limit}"
    )
    return memory