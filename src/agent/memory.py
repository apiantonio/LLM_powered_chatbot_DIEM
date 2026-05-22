"""Gestione della memoria conversazionale per l'agente RAG DIEM.

Implementa una memoria intelligente con filtraggio per similarita semantica
e summarization automatica dei turni precedenti.
"""

import logging
import re
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_classic.memory import ConversationSummaryBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings

from src.config.settings import MemoryConfig

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Rappresenta un singolo turno di conversazione utente-assistente."""

    user_message: str
    assistant_message: str
    turn_number: int
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


class SmartConversationMemory:
    """Memoria conversazionale con filtraggio per similarita e summarization.

    Mantiene una finestra scorrevole di turni, filtra quelli rilevanti
    tramite cosine similarity e produce un riassunto quando necessario.
    """

    def __init__(
        self,
        llm_for_summary,
        embedding_model: HuggingFaceEmbeddings,
        config: Optional[MemoryConfig] = None,
    ):
        """Inizializza la memoria conversazionale.

        Args:
            llm_for_summary: LLM utilizzato per la summarization dei turni.
            embedding_model: Modello di embedding per il calcolo della similarita.
            config: Configurazione della memoria. Se None, usa i default.
        """
        self._config = config or MemoryConfig()
        self._max_turns = self._config.max_turns
        self._similarity_threshold = self._config.similarity_threshold
        self._max_token_limit = self._config.max_token_limit
        self._turns: List[ConversationTurn] = []
        self._pending_user_message: str = ""
        self._turn_counter: int = 0
        self._embedding_model = embedding_model
        self._llm_for_summary = llm_for_summary

        logger.info(
            "SmartConversationMemory inizializzata: "
            "max_turns=%d, similarity_threshold=%.2f, max_token_limit=%d",
            self._max_turns,
            self._similarity_threshold,
            self._max_token_limit,
        )

    @property
    def turn_count(self) -> int:
        """Restituisce il numero totale di turni registrati."""
        return self._turn_counter

    @property
    def is_empty(self) -> bool:
        """Indica se la memoria non contiene alcun turno."""
        return self._turn_counter == 0

    def add_user_message(self, message: str) -> int:
        """Registra un messaggio utente e incrementa il contatore dei turni.

        Args:
            message: Testo del messaggio utente.

        Returns:
            Numero del turno corrente.
        """
        self._turn_counter += 1
        self._pending_user_message = message
        return self._turn_counter

    def add_assistant_message(self, message: str) -> None:
        """Registra la risposta dell'assistente e completa il turno corrente.

        Args:
            message: Testo della risposta dell'assistente.
        """
        user_msg = self._pending_user_message

        turn_text = f"{user_msg} {message}"
        try:
            turn_embedding = np.array(
                self._embedding_model.embed_query(turn_text)
            )
        except Exception as e:
            logger.warning("Errore embedding turno: %s. Turno senza embedding.", e)
            turn_embedding = None

        self._turns.append(ConversationTurn(
            user_message=user_msg,
            assistant_message=message,
            turn_number=self._turn_counter,
            embedding=turn_embedding,
        ))
        self._pending_user_message = ""

        if len(self._turns) > self._max_turns:
            excess = len(self._turns) - self._max_turns
            self._turns = self._turns[excess:]

        logger.debug("Memoria aggiornata: %d turni", len(self._turns))

    def rollback_last_turn(self) -> None:
        """Annulla l'ultimo turno pendente o completato.

        Se esiste un messaggio utente pendente (add_user_message chiamato
        ma add_assistant_message non ancora), decrementa il contatore e
        resetta il messaggio pendente. Questo viene utilizzato quando un
        guardrail blocca la risposta e l'interazione non deve essere
        salvata in memoria.
        """
        if self._pending_user_message:
            self._pending_user_message = ""
            self._turn_counter = max(0, self._turn_counter - 1)
            logger.info(
                "Rollback turno pendente. Contatore turni: %d",
                self._turn_counter,
            )
            return

        if self._turns:
            removed_turn = self._turns.pop()
            self._turn_counter = max(0, self._turn_counter - 1)
            logger.info(
                "Rollback turno completato #%d. Contatore turni: %d",
                removed_turn.turn_number,
                self._turn_counter,
            )

    def _filter_turns_by_similarity(self, query: str) -> List[ConversationTurn]:
        """Filtra i turni in memoria in base alla similarita con la query corrente.

        Args:
            query: Testo della query corrente.

        Returns:
            Lista di turni con similarita sopra soglia.
        """
        if not self._turns:
            return []

        try:
            query_embedding = np.array(
                self._embedding_model.embed_query(query)
            )
        except Exception as e:
            logger.warning("Errore embedding query: %s. Restituisco tutti i turni.", e)
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
                    "Turno #%d sopra soglia: sim=%.4f >= %.4f",
                    turn.turn_number,
                    similarity,
                    self._similarity_threshold,
                )
            else:
                logger.debug(
                    "Turno #%d SCARTATO: sim=%.4f < %.4f",
                    turn.turn_number,
                    similarity,
                    self._similarity_threshold,
                )

        logger.info(
            "Filtraggio similarita: %d/%d turni sopra soglia (%.2f)",
            len(filtered),
            len(self._turns),
            self._similarity_threshold,
        )
        return filtered

    def _summarize_if_needed(
        self, filtered_turns: List[ConversationTurn]
    ) -> List[BaseMessage]:
        """Produce un riassunto dei turni filtrati tramite ConversationSummaryBufferMemory.

        Args:
            filtered_turns: Turni da riassumere.

        Returns:
            Lista di messaggi LangChain risultanti dalla summarization.
        """
        if not filtered_turns:
            return []

        summary_memory = ConversationSummaryBufferMemory(
            llm=self._llm_for_summary,
            max_token_limit=self._max_token_limit,
            return_messages=True,
            memory_key="history",
            human_prefix=self._config.summary_human_prefix,
            ai_prefix=self._config.summary_ai_prefix,
        )

        for turn in filtered_turns:
            summary_memory.save_context(
                {"input": turn.user_message},
                {"output": turn.assistant_message},
            )

        memory_variables = summary_memory.load_memory_variables({})
        messages = memory_variables.get("history", [])

        logger.info(
            "Summarization: %d turni -> %d messaggi in output",
            len(filtered_turns),
            len(messages),
        )
        return messages

    def find_exact_match(self, query: str) -> Optional[str]:
        """Cerca una corrispondenza esatta nella cache dei turni precedenti.

        Args:
            query: Testo della query da cercare.

        Returns:
            Risposta dell'assistente se trovata, None altrimenti.
        """
        if not self._turns:
            return None

        def get_fingerprint(text: str) -> str:
            """Genera un'impronta normalizzata del testo."""
            text = text.lower()
            return re.sub(r'[\W_]+', '', text)

        query_fingerprint = get_fingerprint(query)

        if not query_fingerprint:
            return None

        for turn in reversed(self._turns):
            if get_fingerprint(turn.user_message) == query_fingerprint:
                logger.info(
                    "Cache HIT! Impronta '%s' trovata al Turno #%d",
                    query_fingerprint,
                    turn.turn_number,
                )
                return turn.assistant_message

        return None

    def _build_history_messages(self, current_query: str) -> list:
        """Costruisce la lista di messaggi di storico filtrati e riassunti.

        Metodo interno condiviso da get_messages_for_agent e
        get_messages_for_agent_no_retrieval per evitare duplicazione.

        Args:
            current_query: Query corrente per il filtraggio per similarita.

        Returns:
            Lista di dizionari con chiavi 'role' e 'content' per lo storico.
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

        return messages

    def get_messages_for_agent(self, current_query: str) -> list:
        """Costruisce la lista di messaggi da inviare all'agente.

        Include lo storico filtrato e riassunto, piu la query corrente.
        La query viene passata senza alcuna direttiva aggiuntiva: la
        decisione se invocare un tool o rispondere direttamente e'
        delegata interamente al system prompt dell'agente.

        Args:
            current_query: Query corrente dell'utente.

        Returns:
            Lista di dizionari con chiavi 'role' e 'content'.
        """
        messages = self._build_history_messages(current_query)
        messages.append({"role": "user", "content": current_query})
        return messages

    def get_messages_for_agent_no_retrieval(self, current_query: str) -> list:
        """Costruisce la lista di messaggi per l'agente SENZA retrieval.

        Utilizzato per le domande meta (saluti, ringraziamenti, ecc.)
        dove non e' necessario invocare tool di ricerca. Funzionalmente
        identico a get_messages_for_agent dopo la rimozione del retrieval
        reminder, ma mantenuto per chiarezza semantica nel codice chiamante.

        Args:
            current_query: Query corrente dell'utente.

        Returns:
            Lista di dizionari con chiavi 'role' e 'content'.
        """
        messages = self._build_history_messages(current_query)
        messages.append({"role": "user", "content": current_query})
        return messages

    def get_last_completed_turn(self) -> Tuple[str, str]:
        """Restituisce l'ultimo turno completato (utente, assistente).

        Returns:
            Tupla con messaggio utente e risposta assistente dell'ultimo turno,
            oppure tuple vuote se non ci sono turni.
        """
        if not self._turns:
            return ("", "")

        last_turn = self._turns[-1]
        return (last_turn.user_message, last_turn.assistant_message)

    def get_langchain_history(self) -> List[BaseMessage]:
        """Restituisce l'ultimo turno come lista di messaggi LangChain.

        Returns:
            Lista contenente HumanMessage e AIMessage dell'ultimo turno,
            oppure lista vuota se non ci sono turni.
        """
        if not self._turns:
            return []

        last_turn = self._turns[-1]
        max_chars = self._config.langchain_history_max_chars
        truncated = last_turn.assistant_message[:max_chars]
        if len(last_turn.assistant_message) > max_chars:
            truncated += "..."

        history = [
            HumanMessage(content=last_turn.user_message),
            AIMessage(content=truncated),
        ]
        return history

    def get_history_summary(self) -> str:
        """Produce un riepilogo testuale dello storico conversazione.

        Returns:
            Stringa formattata con tutti i turni in memoria.
        """
        if not self._turns:
            return "(nessuno storico)"

        preview_chars = self._config.history_preview_chars
        lines = []
        for turn in self._turns:
            user_preview = turn.user_message[:preview_chars]
            if len(turn.user_message) > preview_chars:
                user_preview += "..."
            assistant_preview = turn.assistant_message[:preview_chars]
            if len(turn.assistant_message) > preview_chars:
                assistant_preview += "..."
            lines.append(f"  [Turno {turn.turn_number}] Utente: {user_preview}")
            lines.append(f"                Assistente: {assistant_preview}")

        return "\n".join(lines)

    def clear(self) -> None:
        """Resetta completamente la memoria conversazionale."""
        self._turns.clear()
        self._pending_user_message = ""
        self._turn_counter = 0
        logger.info("SmartConversationMemory resettata")


def create_conversation_memory(
    llm_for_summary=None,
    embedding_model: Optional[HuggingFaceEmbeddings] = None,
    config: Optional[MemoryConfig] = None,
) -> SmartConversationMemory:
    """Factory per la creazione di una SmartConversationMemory.

    Se llm_for_summary o embedding_model non sono forniti, vengono creati
    automaticamente a partire dalla configurazione dell'applicazione.

    Args:
        llm_for_summary: LLM per la summarization. Se None, viene creato automaticamente.
        embedding_model: Modello di embedding. Se None, viene creato automaticamente.
        config: Configurazione della memoria. Se None, viene caricata da settings.

    Returns:
        Istanza di SmartConversationMemory configurata.
    """
    if config is None:
        from src.config.settings import load_settings
        settings = load_settings()
        config = settings.memory

    if llm_for_summary is None:
        from src.config.settings import load_settings
        from src.agent.llm_providers import create_chat_model

        settings = load_settings()
        llm_for_summary = create_chat_model(settings.llm)
        logger.info("SmartMemory: LLM per summarization creato da settings")

    if embedding_model is None:
        from src.config.settings import load_settings

        settings = load_settings()
        embedding_model = HuggingFaceEmbeddings(
            model_name=settings.embedding.model_name,
            encode_kwargs={"normalize_embeddings": settings.embedding.normalize_embeddings},
        )
        logger.info("SmartMemory: Embedding model creato: %s", settings.embedding.model_name)

    memory = SmartConversationMemory(
        llm_for_summary=llm_for_summary,
        embedding_model=embedding_model,
        config=config,
    )
    logger.info(
        "SmartConversationMemory creata: max_turns=%d, "
        "similarity_threshold=%.2f, max_token_limit=%d",
        config.max_turns,
        config.similarity_threshold,
        config.max_token_limit,
    )
    return memory