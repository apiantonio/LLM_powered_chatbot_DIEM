"""
retrieval/engine.py — RetrievalEngine RIVISTO per 3 Vector Store.

REFACTORING v4 — QUERY REWRITING CON LLAMA 3.3 70B (via Groq):

  Il QueryOptimizer usa Llama 3.3 70B (Groq free tier)
  esclusivamente per rewrite e multiquery expansion.

  REWRITE — Regole:
    - Risolve coreferenze su QUALSIASI entità (persone, corsi, aule,
      laboratori, bandi, sedi, ecc.), non solo su nomi di docenti.
    - Il contesto è SOLO l'ultimo turno (ultima domanda utente +
      risposta ottenuta). Non serve l'intera history.
    - Se la query è autosufficiente, viene restituita identica.

  EXPAND:
    - 3 varianti semantiche per multiquery retrieval.

  INVARIATI DA v3:
    - CrossEncoderReranker
    - RetrievalEngine (flusso retrieve/retrieve_from_all)
    - FIX Parent-Child per offerta_formativa

FLUSSO:
  rewrite → expand (multiquery) → retrieval parallelo → merge → rerank
"""

import re
import logging
from typing import List, Optional, Set

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from config.settings import RerankerConfig
from ingestion.router import CollectionTarget

logger = logging.getLogger(__name__)


class QueryOptimizer:
    """
    Pre-Retrieval: Query Rewriting + Multiquery Expansion.

    Usa un LLM dedicato (Llama 3.3 70B via Groq) per:
      1. REWRITE — risoluzione coreferenze basata su ultimo turno
      2. EXPAND — generazione varianti semantiche per retrieval
    """

    # ──────────────────────────────────────────────────────────
    # REWRITE PROMPT — Llama 3.3 70B
    #
    # Il contesto passato è SOLO l'ultimo turno: la domanda
    # precedente dell'utente e la risposta ricevuta.
    # Il modello deve risolvere pronomi e riferimenti impliciti
    # su QUALSIASI tipo di entità (corsi, aule, docenti, bandi, ecc.)
    #
    # Few-shot:
    #   (a) Coreferenza su persona
    #   (b) Coreferenza su corso di laurea
    #   (c) Query autosufficiente (nessuna sostituzione)
    # ──────────────────────────────────────────────────────────
    REWRITE_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "You are a coreference resolver for an Italian university Q&A system. "
         "You receive the last interaction (user question + assistant answer) and a new query. "
         "Replace any pronouns or implicit references (lui, lei, suo, suoi, questo, quella, "
         "lì, ci, ne, quali sono i suoi, ecc.) with the explicit entity from the last interaction. "
         "The entity can be anything: a person, a course, a classroom, a lab, a scholarship, etc. "
         "If the new query is already self-contained, return it unchanged. "
         "Do not answer the question. Output only the rewritten query.\n\n"
         "Last Q: \"Chi è il prof. Rossi?\" Last A: \"Il prof. Rossi insegna...\" → "
         "New query: \"Qual è il suo ricevimento?\" → "
         "Output: Qual è il ricevimento del prof. Rossi?\n\n"
         "Last Q: \"Parlami del corso di Informatica triennale\" Last A: \"Il corso prevede...\" → "
         "New query: \"Quali sono i suoi contenuti?\" → "
         "Output: Quali sono i contenuti del corso di Informatica triennale?\n\n"
         "Last Q: \"Dove si trova l'aula 10?\" Last A: \"L'aula 10 è nel campus...\" → "
         "New query: \"Quanti posti ha?\" → "
         "Output: Quanti posti ha l'aula 10?\n\n"
         "Last Q: \"Parlami di Ingegneria Informatica\" Last A: \"...\" → "
         "New query: \"Dove si trova l'aula 10?\" → "
         "Output: Dove si trova l'aula 10?"),
        ("human",
         "Last Q: \"{last_user_query}\"\nLast A: \"{last_assistant_answer}\"\n\n"
         "New query: \"{question}\""),
    ])

    # ──────────────────────────────────────────────────────────
    # MULTIQUERY PROMPT — Llama 3.3 70B
    # ──────────────────────────────────────────────────────────
    MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "Generate exactly 3 Italian rephrasings of the given question for semantic search. "
         "Preserve all proper nouns and the original intent. "
         "Output one variant per line, no numbering, no explanations."),
        ("human", "{question}"),
    ])

    def __init__(self, llm_chat_model):
        self._llm = llm_chat_model

        # Chain REWRITE: invoca LLM, prende solo la prima riga dell'output
        self._rewrite_chain = self.REWRITE_PROMPT | self._llm | RunnableLambda(
            lambda msg: msg.content.strip().split('\n')[0].strip()
        )

        # Chain EXPAND: invoca LLM, splitta per newline
        self._multi_query_chain = self.MULTI_QUERY_PROMPT | self._llm | RunnableLambda(
            lambda msg: [q.strip() for q in msg.content.strip().split("\n") if q.strip()]
        )

    def rewrite(
        self,
        question: str,
        last_user_query: str = "",
        last_assistant_answer: str = "",
    ) -> str:
        """
        Riscrivi la query risolvendo coreferenze dall'ultimo turno.

        Args:
            question: La query corrente dell'utente.
            last_user_query: La domanda dell'utente nel turno precedente.
            last_assistant_answer: La risposta dell'assistente nel turno precedente.

        Returns:
            La query riscritta con coreferenze risolte, oppure
            la query originale se non c'è contesto o è autosufficiente.
        """
        # Se non c'è un turno precedente, la query è per forza autosufficiente
        if not last_user_query:
            logger.info(f"Nessun turno precedente, query invariata: '{question}'")
            return question

        # Tronca la risposta precedente per non sprecare token
        # (al rewriter serve solo sapere DI COSA si parlava, non l'intera risposta)
        truncated_answer = last_assistant_answer[:300]
        if len(last_assistant_answer) > 300:
            truncated_answer += "..."

        try:
            result = self._rewrite_chain.invoke({
                "question": question,
                "last_user_query": last_user_query,
                "last_assistant_answer": truncated_answer,
            })

            # ── Sanity checks minimali ──

            if not result.strip():
                logger.warning("Rewriting ha prodotto stringa vuota. Uso l'originale.")
                return question

            if len(result) > len(question) * 3:
                logger.warning(
                    f"Rewriting sospetto ({len(result)} chars vs {len(question)}). "
                    f"Uso l'originale."
                )
                return question

            # Verifica conservazione entità della query originale
            original_entities = self._extract_proper_nouns(question)
            if original_entities:
                result_lower = result.lower()
                missing = [e for e in original_entities if e.lower() not in result_lower]
                if missing:
                    logger.warning(f"Rewriting ha perso entità: {missing}. Uso l'originale.")
                    return question

            logger.info(f"Query rewritten: '{question}' → '{result}'")
            return result

        except Exception as e:
            logger.warning(f"Errore rewriting, uso query originale: {e}")
            return question

    def expand(self, question: str) -> List[str]:
        """Genera varianti semantiche della query per multiquery retrieval."""
        try:
            variants = self._multi_query_chain.invoke({"question": question})
            seen: Set[str] = {question.strip().lower()}
            unique_variants = []
            for v in variants[:3]:
                v_clean = v.strip().lstrip('0123456789.-) ')
                if v_clean and v_clean.lower() not in seen:
                    seen.add(v_clean.lower())
                    unique_variants.append(v_clean)
            logger.info(f"Multiquery expansion: {len(unique_variants)} varianti generate")
            return [question] + unique_variants
        except Exception as e:
            logger.warning(f"Errore multi-query expansion: {e}")
            return [question]

    @staticmethod
    def _extract_proper_nouns(text: str) -> list:
        """Estrae probabili nomi propri da un testo italiano."""
        skip_words = {
            "chi", "cosa", "come", "dove", "quando", "quale", "quali",
            "parlami", "dimmi", "spiegami", "cercami", "trovami",
            "il", "la", "lo", "le", "gli", "un", "una", "del", "della",
            "dei", "delle", "nel", "nella", "sul", "sulla",
            "per", "con", "tra", "fra", "che", "non", "sono",
            "qual", "quanto", "quanti", "quante",
        }
        words = text.split()
        entities = []
        for w in words:
            clean = w.strip("?.,!;:'\"()[]")
            if clean and clean[0].isupper() and len(clean) > 2 and clean.lower() not in skip_words:
                entities.append(clean)
        return entities


# ============================================================
# CROSS-ENCODER RERANKER (invariato)
# ============================================================

class CrossEncoderReranker:
    """Post-Retrieval: ri-ordina i documenti candidati con Cross-Encoder."""

    def __init__(self, config: RerankerConfig):
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(config.model_name)
        self._top_n = config.top_n
        self._score_threshold = config.score_treshold
        logger.info(f"Cross-Encoder Reranker: {config.model_name}")

    def rerank(
        self, query: str, documents: List[Document], top_n: Optional[int] = None
    ) -> List[Document]:
        if not documents:
            return []

        top_n = top_n or self._top_n
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

        print(f"\n--- TOP 5 CLASSIFICA (su {len(ranked)} CANDIDATI) ---")
        for i, (doc, score) in enumerate(ranked[:5]):
            print(f"[{i+1}] Score: {score:.4f} | Fonte: {doc.metadata.get('source_url', 'N/D')}")
            print(f"    Testo: {doc.page_content[:150]}...")
        print("-" * 65)

        result = []
        for doc, score in ranked[:top_n]:
            if score < self._score_threshold:
                logger.debug(
                    f"Documento filtrato (score {score:.4f} < threshold "
                    f"{self._score_threshold}): "
                    f"{doc.metadata.get('url_originale', doc.metadata.get('source_url', 'N/D'))[:80]}"
                )
                continue
            doc.metadata["relevance_score"] = float(score)
            result.append(doc)

        if not result and documents:
            logger.warning(
                f"Tutti i {len(documents)} documenti candidati filtrati "
                f"(score < {self._score_threshold}). Query: '{query[:80]}'"
            )

        return result


# ============================================================
# RETRIEVAL ENGINE
# ============================================================

class RetrievalEngine:
    """
    Orchestratore retrieval per 3 Vector Store (audit §8).

    Flusso: rewrite → expand (multiquery) → retrieval parallelo → merge → rerank.

    v4: il rewrite() riceve l'ultimo turno (last_user_query, last_assistant_answer)
    invece dell'intera chat history.
    """

    def __init__(self, indexer, reranker, query_optimizer=None):
        self._indexer = indexer
        self._reranker = reranker
        self._optimizer = query_optimizer

        self._collection_retrievers = {}
        for target in CollectionTarget:
            self._collection_retrievers[target.value] = (
                indexer.get_collection_retriever(target)
            )
        self._pc_retriever = indexer.get_parent_child_retriever()
        self._pc_child_vectorstore = indexer._pc_child_vectorstore

    def retrieve(
        self,
        query: str,
        collection: Optional[str] = None,
        metadata_filter: Optional[dict] = None,
        chat_history: Optional[list] = None,
    ) -> tuple:
        """
        Retrieval completo: rewrite → multiquery → retrieval → rerank.

        v4: chat_history è atteso come lista di BaseMessage.
        Il metodo estrae automaticamente l'ultimo turno (ultima coppia
        HumanMessage + AIMessage) e lo passa al rewriter.
        """
        # ── STEP 1: QUERY REWRITING ──
        effective_query = query
        if self._optimizer and chat_history:
            last_user, last_assistant = self._extract_last_turn(chat_history)
            effective_query = self._optimizer.rewrite(
                question=query,
                last_user_query=last_user,
                last_assistant_answer=last_assistant,
            )
        elif self._optimizer:
            # Nessuna history → rewrite restituirà la query invariata
            effective_query = self._optimizer.rewrite(question=query)

        # ── STEP 2: MULTIQUERY EXPANSION ──
        multi_queries = [effective_query]
        if self._optimizer:
            multi_queries = self._optimizer.expand(effective_query)

        if collection is None:
            docs, _ = self.retrieve_from_all(effective_query)
            return docs, effective_query, multi_queries

        # ── STEP 3: RETRIEVAL PARALLELO PER OGNI VARIANTE ──
        all_candidates = []
        seen_hashes: Set[int] = set()

        for mq in multi_queries:
            if metadata_filter:
                retriever = self._get_filtered_retriever(collection, metadata_filter)
            else:
                retriever = self._collection_retrievers.get(collection)

            if retriever is None:
                logger.error(f"Collection sconosciuta: {collection}")
                continue

            try:
                candidates = retriever.invoke(mq)
                for doc in candidates:
                    h = hash(doc.page_content[:200])
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        all_candidates.append(doc)
            except Exception as e:
                logger.warning(f"Errore retrieval multiquery '{mq[:60]}': {e}")

            if collection == CollectionTarget.OFFERTA_FORMATIVA.value:
                try:
                    pc_docs = self._retrieve_from_parent_child(mq, metadata_filter)
                    for doc in pc_docs:
                        h = hash(doc.page_content[:200])
                        if h not in seen_hashes:
                            seen_hashes.add(h)
                            all_candidates.append(doc)
                except Exception as e:
                    logger.warning(f"Errore retrieval Parent-Child per '{mq[:60]}': {e}")

        logger.info(
            f"Retrieval completato: {len(all_candidates)} candidati unici "
            f"(collection={collection}, filter={metadata_filter})"
        )

        # ── STEP 4: RERANKING ──
        final_docs = self._reranker.rerank(effective_query, all_candidates)
        return final_docs, effective_query, multi_queries

    @staticmethod
    def _extract_last_turn(chat_history: list) -> tuple:
        """
        Estrae l'ultimo turno completo (HumanMessage + AIMessage) dalla history.

        Returns:
            (last_user_query, last_assistant_answer) — stringhe.
            Se la history è vuota o incompleta, restituisce ("", "").
        """
        from langchain_core.messages import HumanMessage, AIMessage

        last_user = ""
        last_assistant = ""

        # Scorri al contrario per trovare l'ultima coppia completa
        for i in range(len(chat_history) - 1, -1, -1):
            msg = chat_history[i]
            if isinstance(msg, AIMessage) and not last_assistant:
                last_assistant = msg.content
            elif isinstance(msg, HumanMessage) and not last_user:
                last_user = msg.content
            if last_user and last_assistant:
                break

        return last_user, last_assistant

    def retrieve_from_all(self, query: str, metadata_filter: Optional[dict] = None) -> tuple:
        """Retrieval cross-collection con multiquery e filtro opzionale."""
        multi_queries = [query]
        if self._optimizer:
            multi_queries = self._optimizer.expand(query)

        all_candidates = []
        seen_hashes: Set[int] = set()

        for mq in multi_queries:
            for collection_name, default_retriever in self._collection_retrievers.items():
                try:
                    retriever = default_retriever
                    if metadata_filter:
                        filtered = self._get_filtered_retriever(collection_name, metadata_filter)
                        if filtered:
                            retriever = filtered

                    docs = retriever.invoke(mq)
                    for doc in docs:
                        h = hash(doc.page_content[:200])
                        if h not in seen_hashes:
                            seen_hashes.add(h)
                            all_candidates.append(doc)
                except Exception as e:
                    logger.warning(f"Errore retrieval da {collection_name}: {e}")

            try:
                pc_docs = self._retrieve_from_parent_child(mq, metadata_filter)
                for doc in pc_docs:
                    h = hash(doc.page_content[:200])
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        all_candidates.append(doc)
            except Exception as e:
                logger.warning(f"Errore retrieval Parent-Child: {e}")

        final_docs = self._reranker.rerank(query, all_candidates)
        return final_docs, query

    # ================================================================
    # PARENT-CHILD RETRIEVAL
    # ================================================================

    def _retrieve_from_parent_child(
        self,
        query: str,
        metadata_filter: Optional[dict] = None,
    ) -> List[Document]:
        """Recupera documenti dal Parent-Child vectorstore."""
        if metadata_filter:
            chroma_where = self._build_chroma_filter(metadata_filter)
            try:
                retriever = self._pc_child_vectorstore.as_retriever(
                    search_type=self._indexer._settings.vectorstore.search_type,
                    search_kwargs={
                        "k": self._indexer._settings.vectorstore.search_k,
                        "filter": chroma_where,
                    },
                )
                docs = retriever.invoke(query)
                logger.info(
                    f"Parent-Child (filtrato): {len(docs)} child chunks "
                    f"(filter={metadata_filter})"
                )
                return docs
            except Exception as e:
                logger.warning(
                    f"Errore Parent-Child con filtro {metadata_filter}: {e}. "
                    f"Fallback a retrieval senza filtro."
                )
                try:
                    return self._pc_retriever.invoke(query)
                except Exception as e2:
                    logger.warning(f"Anche il fallback Parent-Child è fallito: {e2}")
                    return []
        else:
            return self._pc_retriever.invoke(query)

    # ================================================================
    # FILTERED RETRIEVER FACTORY
    # ================================================================

    def _get_filtered_retriever(self, collection_name: str, metadata_filter: dict):
        """Crea un retriever Chroma con filtro metadata."""
        try:
            target = CollectionTarget(collection_name)
        except ValueError:
            return None

        chroma_collection = self._indexer._collections[target]
        chroma_where = self._build_chroma_filter(metadata_filter)

        return chroma_collection.as_retriever(
            search_type=self._indexer._settings.vectorstore.search_type,
            search_kwargs={
                "k": self._indexer._settings.vectorstore.search_k,
                "filter": chroma_where,
            },
        )

    @staticmethod
    def _build_chroma_filter(metadata_filter: dict) -> dict:
        """Converte un dict in formato filtro Chroma."""
        conditions = []

        for key, value in metadata_filter.items():
            if key.startswith("$"):
                return metadata_filter
            elif isinstance(value, dict):
                conditions.append({key: value})
            elif isinstance(value, list):
                conditions.append({key: {"$in": value}})
            else:
                conditions.append({key: {"$eq": value}})

        if len(conditions) == 1:
            return conditions[0]
        elif len(conditions) > 1:
            return {"$and": conditions}

        return {}