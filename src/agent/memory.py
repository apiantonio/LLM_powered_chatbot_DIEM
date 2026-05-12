"""
Gestione della memoria conversazionale per l'Agente RAG DIEM.

REFACTORING — SMART MEMORY A DUE STADI:

  Stadio 1 — Filtraggio per similarità:
    Ogni turno completato viene embedded. Quando arriva una nuova query,
    si recuperano solo i turni con similarità coseno sopra soglia.
    Usa HuggingFaceEmbeddings (già nel progetto) per gli embedding.

  Stadio 2 — Summarization con token budget:
    I turni filtrati vengono gestiti da ConversationSummaryBufferMemory
    di LangChain: se superano max_token_limit, i più vecchi vengono
    riassunti automaticamente dall'LLM.

MOTIVAZIONE DELLE SCELTE:
  - ConversationSummaryBufferMemory è la classe nativa LangChain che
    gestisce il token budget con summarization automatica.
  - Il filtraggio per similarità usa numpy.dot per il calcolo del coseno
    sugli embedding prodotti da HuggingFaceEmbeddings, evitando la
    necessità di un vector store separato (FAISS/Chroma) in-memory.
  - NON si usano cicli for custom per gestire la memoria: il filtraggio
    è un'operazione di selezione, non di chunking o summarization.
  - La summarization è delegata INTEGRALMENTE a ConversationSummaryBufferMemory.

Pattern: Strategy (GoF), Facade (GoF).
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


# Reminder iniettato alla fine di ogni messaggio utente
_RETRIEVAL_REMINDER = (
    "\n\n[SISTEMA: usa SEMPRE!!! un tool di ricerca "
    "per trovare le informazioni necessarie. Se il tool restituisce un errore, "
    "NON riprovare lo stesso tool — usa un tool alternativo oppure comunica "
    "all'utente che l'informazione non è al momento disponibile. "
    "NON ABBREVIARE MAI LA QUERY ORIGINALE DELL'UTENTE! "
    "Passa la query dell'utente INTEGRA e COMPLETA al tool, "
    "senza riassumerla, comprimerla o ridurla a keyword.]"
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

    Stadio 1 — Filtraggio per Similarità:
      Ogni turno viene embedded con HuggingFaceEmbeddings.
      Data una nuova query, si calcolano le similarità coseno
      con tutti i turni memorizzati e si scartano quelli sotto soglia.

    Stadio 2 — Summarization con Token Budget:
      I turni sopra soglia vengono passati a ConversationSummaryBufferMemory
      che gestisce il token limit: se il totale supera max_token_limit,
      i turni più vecchi vengono riassunti dall'LLM.

    Motivazione:
      - Il filtraggio per similarità evita di iniettare contesto
        irrilevante (es. si parlava di "Angelo Marcelli" e ora si chiede
        "Dove si trova l'aula 126?" — la history su Marcelli è irrilevante).
      - La summarization evita di superare il context window dell'LLM
        quando la conversazione è lunga e molti turni sono rilevanti.
      - Si usano SOLO classi native LangChain per la summarization
        (ConversationSummaryBufferMemory) e per gli embedding
        (HuggingFaceEmbeddings).
    """

    def __init__(
        self,
        llm_for_summary,
        embedding_model: HuggingFaceEmbeddings,
        max_turns: int = 10,
        similarity_threshold: float = 0.3,
        max_token_limit: int = 1500,
    ):
        """
        Args:
            llm_for_summary: BaseChatModel per la summarization dei turni
                             (ConversationSummaryBufferMemory ne ha bisogno).
            embedding_model: HuggingFaceEmbeddings per il calcolo della
                             similarità coseno tra query e turni.
            max_turns: Numero massimo di turni mantenuti in memoria (sliding window).
            similarity_threshold: Soglia di similarità coseno [0, 1].
                                  Turni sotto questa soglia vengono scartati.
            max_token_limit: Limite di token per i turni filtrati.
                             Se superato, i turni più vecchi vengono riassunti.
        """
        self._max_turns = max_turns
        self._similarity_threshold = similarity_threshold
        self._max_token_limit = max_token_limit
        self._turns: List[ConversationTurn] = []
        self._pending_user_message: str = ""
        self._turn_counter: int = 0

        # Embedding model per il filtraggio per similarità (Stadio 1)
        self._embedding_model = embedding_model

        # ConversationSummaryBufferMemory per la summarization (Stadio 2)
        # Viene ricreata ad ogni get_messages_for_agent() con i turni filtrati
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
        """
        Registra la risposta dell'assistente, completa il turno e
        calcola l'embedding del turno per il filtraggio per similarità.
        """
        user_msg = getattr(self, '_pending_user_message', '')

        # Calcola l'embedding del turno (user + assistant concatenati)
        # L'embedding cattura il contesto semantico dell'intero turno
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

        # Sliding window: rimuovi i turni più vecchi
        if len(self._turns) > self._max_turns:
            excess = len(self._turns) - self._max_turns
            self._turns = self._turns[excess:]

        logger.debug(f"Memoria aggiornata: {len(self._turns)} turni")

    def _filter_turns_by_similarity(self, query: str) -> List[ConversationTurn]:
        """
        STADIO 1: Filtra i turni per similarità coseno con la query corrente.

        Calcola la similarità coseno tra l'embedding della query corrente
        e gli embedding di ogni turno memorizzato. Restituisce solo i turni
        con similarità >= similarity_threshold.

        Se non ci sono turni con embedding valido, restituisce tutti i turni
        (fallback sicuro).
        """
        if not self._turns:
            return []

        try:
            query_embedding = np.array(
                self._embedding_model.embed_query(query)
            )
        except Exception as e:
            logger.warning(
                f"Errore embedding query per filtraggio similarità: {e}. "
                f"Restituisco tutti i turni."
            )
            return list(self._turns)

        filtered = []
        for turn in self._turns:
            if turn.embedding is None:
                # Turni senza embedding vengono inclusi per sicurezza
                filtered.append(turn)
                continue

            # Similarità coseno: dot(a, b) / (||a|| * ||b||)
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
        """
        STADIO 2: Applica summarization sui turni filtrati se superano
        il token limit, usando ConversationSummaryBufferMemory di LangChain.

        ConversationSummaryBufferMemory gestisce automaticamente:
          - Mantiene i turni recenti in forma integrale
          - Riassume i turni più vecchi quando si supera max_token_limit
          - Restituisce un buffer di messaggi entro il token budget

        Motivazione: usiamo la classe nativa di LangChain anziché
        reimplementare la logica di summarization con cicli custom.
        """
        if not filtered_turns:
            return []

        # Crea una ConversationSummaryBufferMemory temporanea
        # con i soli turni filtrati per similarità
        summary_memory = ConversationSummaryBufferMemory(
            llm=self._llm_for_summary,
            max_token_limit=self._max_token_limit,
            return_messages=True,
            memory_key="history",
            human_prefix="Utente",
            ai_prefix="Assistente",
        )

        # Inserisci i turni filtrati nella memory nativa
        for turn in filtered_turns:
            summary_memory.save_context(
                {"input": turn.user_message},
                {"output": turn.assistant_message},
            )

        # Recupera i messaggi gestiti dalla memory (con summarization applicata)
        memory_variables = summary_memory.load_memory_variables({})
        messages = memory_variables.get("history", [])

        logger.info(
            f"Summarization: {len(filtered_turns)} turni → "
            f"{len(messages)} messaggi in output"
        )
        return messages

    def get_messages_for_agent(self, current_query: str) -> list:
        """
        Costruisce la lista messaggi per create_agent.invoke().

        FLUSSO A DUE STADI:
          1. Filtraggio per similarità: solo turni rilevanti per la query
          2. Summarization: se i turni filtrati superano il token budget,
             i più vecchi vengono riassunti

        Inietta il RETRIEVAL_REMINDER per query non-meta.
        """
        messages = []

        if self._turns:
            # STADIO 1: Filtraggio per similarità
            filtered_turns = self._filter_turns_by_similarity(current_query)

            # STADIO 2: Summarization con token budget
            if filtered_turns:
                summarized_messages = self._summarize_if_needed(filtered_turns)

                # Converti i BaseMessage LangChain nel formato dict per l'agente
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
                            # SystemMessage dalla summarization
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
        Usato dal QueryOptimizer per la riscrittura contestuale.
        NON include il messaggio corrente.
        NON applica filtraggio per similarità (serve l'intero contesto
        al rewriter per risolvere coreferenze).
        """
        history_messages: List[BaseMessage] = []
        for turn in self._turns:
            history_messages.append(HumanMessage(content=turn.user_message))
            history_messages.append(AIMessage(content=turn.assistant_message))
        return history_messages

    def get_history_summary(self) -> str:
        """
        Restituisce un riepilogo testuale dello storico completo.

        Restituisce TUTTI i turni (non solo quelli filtrati) per il log
        dell'interazione. Ogni turno mostra la domanda utente e la
        risposta (troncata a 150 char per leggibilità).
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
    similarity_threshold: float = 0.3,
    max_token_limit: int = 1500,
) -> SmartConversationMemory:
    """
    Factory Method: crea la SmartConversationMemory.

    Se llm_for_summary non è fornito, ne crea uno di default
    usando le settings correnti.
    Se embedding_model non è fornito, usa il modello di embedding
    configurato nelle settings.

    Args:
        max_turns: Numero massimo di turni in sliding window.
        max_tokens: (legacy, ignorato) Sostituito da max_token_limit.
        llm_for_summary: BaseChatModel per la summarization.
        embedding_model: HuggingFaceEmbeddings per il filtraggio per similarità.
        similarity_threshold: Soglia coseno [0, 1] per il filtraggio.
        max_token_limit: Token limit per la summarization.
    """
    # Lazy import per evitare dipendenze circolari
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

    # Usa max_token_limit se max_tokens è fornito (retrocompatibilità)
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