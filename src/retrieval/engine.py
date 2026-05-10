"""
retrieval/engine.py — QueryOptimizer con Domain-Aware Rewriting.

FIX APPLICATI:
  1. QueryOptimizer.rewrite(): rimosso early-return che bypassava il rewriting
     per query con >10 parole senza history.
  2. RetrievalEngine.retrieve(): il rewriting è ora SEMPRE attivo quando
     self._optimizer è presente, indipendentemente dalla presenza di chat_history.
  3. CrossEncoderReranker.rerank(): aggiunto score_threshold per filtrare
     documenti con score sotto -5.0 (falsi positivi semantici).
"""

import logging
from datetime import datetime
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from config.settings import RerankerConfig
from ingestion.router import CollectionTarget

logger = logging.getLogger(__name__)


class QueryOptimizer:
    """
    Pre-Retrieval: Domain-Aware Query Expansion.
    
    Sostituisce il vecchio rewriter che "comprimeva" le query.
    Il nuovo prompt:
      1. Espande sempre (mai comprime)
      2. Inietta contesto DIEM
      3. Applica regole specifiche per persone
      4. Risolve coreferenze dalla history
      5. Risolve riferimenti temporali relativi
    """
    
    REWRITE_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         """Sei un ottimizzatore di query per un sistema di ricerca semantica del \
        Dipartimento DIEM dell'Università degli Studi di Salerno.

        Data e ora correnti: {current_datetime}

        Il tuo compito è RISCRIVERE la domanda dell'utente in una query ottimizzata \
        per la ricerca in un database vettoriale. Segui queste regole:

        REGOLA 1 — ESPANSIONE, MAI COMPRESSIONE:
        La query riscritta DEVE essere più specifica e dettagliata dell'originale. \
        MAI ridurre a semplici keyword. Mantieni l'intento interrogativo.
        SBAGLIATO: "Chi è Mario Vento?" → "Mario Vento"
        CORRETTO:  "Chi è Mario Vento?" → "Profilo accademico, qualifica, ruolo e \
        contatti istituzionali del professore Mario Vento del dipartimento DIEM \
        Università di Salerno"

        REGOLA 2 — CONTESTO DI DOMINIO:
        Aggiungi "dipartimento DIEM" o "Università di Salerno" se non presenti.

        REGOLA 3 — QUERY SU PERSONE:
        Quando si chiede CHI È una persona:
        → Espandi verso: curriculum, qualifica accademica, ruolo, contatti, ricevimento
        → NON includere: bandi, progetti di ricerca (a meno che richiesto esplicitamente)
        Quando si chiedono CORSI INSEGNATI:
        → Espandi verso: insegnamenti, corsi di laurea, anno accademico
        Quando si chiede la RICERCA di un docente:
        → Espandi verso: aree di ricerca, pubblicazioni, gruppi

        REGOLA 4 — RISOLUZIONE COREFERENZE:
        Usa la cronologia per risolvere pronomi e riferimenti impliciti.
        "e dove insegna?" → "Quali corsi insegna il professore [NOME dal contesto] \
        del dipartimento DIEM?"

        REGOLA 5 — RISOLUZIONE TEMPORALE:
        Se la domanda contiene riferimenti temporali relativi, risolvili:
        "domani" → la data del giorno successivo a quello corrente
        "lunedì prossimo" → la data del prossimo lunedì

        Rispondi con SOLO la query riscritta. Nient'altro."""),
        ("placeholder", "{history}"),
        ("human", "{question}"),
    ])
    
    MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "Sei un ottimizzatore di ricerca per il dipartimento DIEM UniSA. "
         "Genera esattamente 3 varianti diverse della seguente domanda, "
         "ciascuna che esplori un angolo diverso del tema nel contesto "
         "universitario. Restituisci solo le 3 query, una per riga."),
        ("human", "{question}"),
    ])
    
    def __init__(self, llm_chat_model):
        self._llm = llm_chat_model
        self._rewrite_chain = self.REWRITE_PROMPT | self._llm | RunnableLambda(
            lambda msg: msg.content.strip()
        )
        self._multi_query_chain = self.MULTI_QUERY_PROMPT | self._llm | RunnableLambda(
            lambda msg: [q.strip() for q in msg.content.strip().split("\n") if q.strip()]
        )
    
    @staticmethod
    def _get_current_datetime_str() -> str:
        """Genera una stringa datetime leggibile in italiano."""
        now = datetime.now()
        giorni = ["lunedì", "martedì", "mercoledì", "giovedì",
                  "venerdì", "sabato", "domenica"]
        mesi = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
                "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"]
        return (
            f"{giorni[now.weekday()]} {now.day} {mesi[now.month - 1]} {now.year}, "
            f"ore {now.strftime('%H:%M')}"
        )
    
    def rewrite(self, question: str, history: Optional[list] = None) -> str:
        """
        Riscrivi la query con espansione di dominio e risoluzione temporale.

        FIX APPLICATO:
          Rimosso l'early-return per query lunghe senza history.
          Prima: query con >10 parole senza history venivano restituite
          invariate. Questo impediva l'espansione di dominio su query
          già strutturate ma prive di contesto DIEM.
          Ora: il rewriting viene eseguito SEMPRE, indipendentemente dalla
          lunghezza della query o dalla presenza di history.
          Il sanity check sulla compressione è mantenuto come guardia.

        Args:
            question: La query originale dell'utente.
            history: Lista opzionale di BaseMessage LangChain per contesto.
                     Se presente, il rewriter risolve coreferenze anaforiche.

        Returns:
            La query riscritta (espansa, contestualizzata al DIEM) oppure
            la query originale se il rewriting fallisce o produce output
            sospetto (compressione).
        """
        # ── RIMOSSO EARLY-RETURN ──
        # Prima c'era:
        #   if not history:
        #       if len(question.split()) > 10:
        #           return question
        # Questo causava il bypass del rewriting per query lunghe senza history.
        # Ora il rewriting è SEMPRE eseguito per garantire l'espansione di dominio.

        try:
            result = self._rewrite_chain.invoke({
                "question": question,
                "history": history or [],
                "current_datetime": self._get_current_datetime_str(),
            })

            # Sanity check: se il risultato è più corto del 50% della query
            # originale, il modello ha probabilmente "compresso" la query
            # invece di espanderla → torniamo alla query originale.
            if len(result) < len(question) * 0.5:
                logger.warning(
                    f"Query rewriting sospetto (compressione): "
                    f"'{question}' → '{result}'. Uso l'originale."
                )
                return question

            logger.info(f"Query rewritten: '{question}' → '{result}'")
            return result

        except Exception as e:
            logger.warning(f"Errore rewriting, uso query originale: {e}")
            return question
    
    def expand(self, question: str) -> List[str]:
        """Genera varianti della query per copertura semantica."""
        try:
            variants = self._multi_query_chain.invoke({"question": question})
            return [question] + variants[:3]
        except Exception as e:
            logger.warning(f"Errore multi-query: {e}")
            return [question]


# ============================================================
# CROSS-ENCODER RERANKER (Post-Retrieval)
# ============================================================

class CrossEncoderReranker:
    """
    Post-Retrieval: ri-ordina i documenti candidati con Cross-Encoder.

    FIX APPLICATO:
      Aggiunto score_threshold per filtrare documenti con score sotto -5.0.
      Il CrossEncoder ms-marco-MiniLM-L6-v2 restituisce logits raw dove
      valori negativi indicano bassa rilevanza. Un threshold di -5.0
      elimina i risultati chiaramente irrilevanti pur mantenendo un
      margine di tolleranza.
    """

    # Soglia minima di rilevanza. I documenti con score inferiore vengono
    # esclusi dai risultati finali. Valore scelto empiricamente:
    # - Score > 0: alta rilevanza (match semantico forte)
    # - Score tra -5 e 0: rilevanza incerta (potenzialmente utile)
    # - Score < -5: irrilevante (rumore semantico)
    DEFAULT_SCORE_THRESHOLD = -5.0

    def __init__(self, config: RerankerConfig):
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(config.model_name)
        self._top_n = config.top_n
        # Il threshold può essere reso configurabile in RerankerConfig
        self._score_threshold = self.DEFAULT_SCORE_THRESHOLD
        logger.info(f"Cross-Encoder Reranker: {config.model_name}")

    def rerank(
        self, query: str, documents: List[Document], top_n: Optional[int] = None
    ) -> List[Document]:
        """
        Ri-ordina i documenti per rilevanza rispetto alla query.

        FIX APPLICATO:
          Aggiunto filtraggio per score_threshold DOPO l'ordinamento.
          Prima: tutti i top_n documenti venivano restituiti, anche con
          score -10 (chiaramente irrilevanti).
          Ora: solo i documenti con score >= threshold vengono inclusi.
          Se tutti i documenti sono sotto threshold, restituisce lista vuota.
          Questo permette al tool di mostrare il messaggio "Non ho trovato
          informazioni pertinenti" invece di restituire risultati fuorvianti.
        """
        if not documents:
            return []

        top_n = top_n or self._top_n

        # Calcola gli score di rilevanza per ogni coppia (query, documento)
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self._model.predict(pairs)

        # Ordina per score decrescente
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

        result = []
        for doc, score in ranked[:top_n]:
            # ── NUOVO: Filtraggio per score threshold ──
            # Documenti con score sotto la soglia sono considerati irrilevanti
            # e vengono esclusi. Questo previene i "falsi positivi semantici"
            # dove Chroma restituisce i documenti più vicini nel vectorspace
            # anche se non sono realmente pertinenti alla query.
            if score < self._score_threshold:
                logger.debug(
                    f"Documento filtrato (score {score:.4f} < threshold "
                    f"{self._score_threshold}): "
                    f"{doc.metadata.get('source_url', 'N/D')[:80]}"
                )
                continue

            doc.metadata["relevance_score"] = float(score)
            result.append(doc)

        # Log se tutti i documenti sono stati filtrati
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
    Orchestratore retrieval — AGGIORNATO con metadata filtering e
    query rewriting sempre attiva.
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

    def retrieve(
        self,
        query: str,
        collection: Optional[str] = None,
        metadata_filter: Optional[dict] = None,
        chat_history: Optional[list] = None,
    ) -> tuple:
        """
        Retrieval con query rewriting SEMPRE attiva e supporto metadata filtering.

        FIX APPLICATO:
          Prima: il rewriting era condizionato a `if self._optimizer and chat_history`.
          Poiché i tool non passavano mai chat_history, il rewriting era SEMPRE
          saltato. Ora la condizione è `if self._optimizer` (senza `and chat_history`).
          La chat_history è opzionale e serve solo per risolvere coreferenze.

        Args:
            query: La query di ricerca originale dell'utente.
            collection: CollectionTarget.value (es. "dipartimento_e_ricerca").
                        Se None, esegue retrieve_from_all().
            metadata_filter: Filtro Chroma per restringere i risultati
                             (es. {"doc_category": "aula"}).
            chat_history: Lista di BaseMessage LangChain per contesto
                          conversazionale. Opzionale.

        Returns:
            Tupla (documenti_reranked: List[Document], query_effettiva: str).
            La query_effettiva è quella riscritta dal QueryOptimizer (o l'originale
            se il rewriting fallisce o non è configurato).
        """
        # ── QUERY REWRITING (SEMPRE ATTIVO) ──
        # FIX: rimossa la condizione `and chat_history`.
        # Prima: if self._optimizer and chat_history → il rewriting era condizionato
        #        alla presenza di chat_history, che non veniva mai passata dai tool.
        # Ora:   if self._optimizer → il rewriting viene eseguito SEMPRE.
        #        La chat_history è opzionale e serve solo per risolvere coreferenze.
        effective_query = query
        if self._optimizer:
            effective_query = self._optimizer.rewrite(query, chat_history)

        if collection is None:
            return self.retrieve_from_all(effective_query)

        # Se c'è un metadata_filter, crea un retriever filtrato ad-hoc
        if metadata_filter:
            retriever = self._get_filtered_retriever(collection, metadata_filter)
        else:
            retriever = self._collection_retrievers.get(collection)

        if retriever is None:
            logger.error(f"Collection sconosciuta: {collection}")
            return [], effective_query

        candidates = retriever.invoke(effective_query)

        # Parent-Child merge per offerta_formativa (invariato)
        if collection == CollectionTarget.OFFERTA_FORMATIVA.value and not metadata_filter:
            pc_docs = self._pc_retriever.invoke(effective_query)
            seen = {hash(d.page_content[:200]) for d in candidates}
            for doc in pc_docs:
                h = hash(doc.page_content[:200])
                if h not in seen:
                    seen.add(h)
                    candidates.append(doc)

        final_docs = self._reranker.rerank(effective_query, candidates)
        return final_docs, effective_query

    def retrieve_from_all(self, query: str) -> tuple:
        """
        Retrieval cross-collection: cerca in tutte le collection e nel
        Parent-Child retriever, poi aggrega e rerank.
        """
        all_candidates = []
        seen_hashes = set()

        for collection_name, retriever in self._collection_retrievers.items():
            try:
                docs = retriever.invoke(query)
                for doc in docs:
                    h = hash(doc.page_content[:200])
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        all_candidates.append(doc)
            except Exception as e:
                logger.warning(f"Errore retrieval da {collection_name}: {e}")

        # Aggiungi risultati Parent-Child
        try:
            pc_docs = self._pc_retriever.invoke(query)
            for doc in pc_docs:
                h = hash(doc.page_content[:200])
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    all_candidates.append(doc)
        except Exception as e:
            logger.warning(f"Errore retrieval Parent-Child: {e}")

        final_docs = self._reranker.rerank(query, all_candidates)
        return final_docs, query

    def _get_filtered_retriever(self, collection_name: str, metadata_filter: dict):
        """
        Crea un retriever Chroma con filtro metadata.
        
        Chroma supporta i filtri nativamente:
          {"docente_sezione": "didattica"}
          → Chroma where clause: {"docente_sezione": {"$eq": "didattica"}}
        
        Per filtri multipli (OR):
          {"$or": [{"doc_category": "aula"}, {"doc_category": "laboratorio"}]}
        """
        try:
            target = CollectionTarget(collection_name)
        except ValueError:
            return None

        # Accede alla collection Chroma dall'indexer
        chroma_collection = self._indexer._collections[target]

        # Costruisce il filtro Chroma-native
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
        """
        Converte un dict in formato filtro Chroma.
        
        GESTIONE CASI (fix del bug che causava il loop):
        {"docente_sezione": "didattica"}         → {"docente_sezione": {"$eq": "didattica"}}
        {"doc_category": ["aula", "lab"]}        → {"doc_category": {"$in": ["aula", "lab"]}}
        {"doc_category": {"$in": ["aula"]}}      → passa direttamente (già formato Chroma)
        {"$or": [...]}                           → passa direttamente (operatore top-level)
        """
        conditions = []
        
        for key, value in metadata_filter.items():
            if key.startswith("$"):
                # Operatore top-level ($or, $and) — passa direttamente
                return metadata_filter
            elif isinstance(value, dict):
                # Già in formato Chroma (es. {"$in": [...]}) — passa direttamente
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