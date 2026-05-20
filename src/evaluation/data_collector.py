"""Collettore dati per la valutazione RAGAS dell'Agente RAG DIEM.

Cattura le quattro variabili necessarie a RAGAS (user_input, retrieved_contexts,
response, reference) per ogni interazione con l'agente, strutturandole nel
formato richiesto dal framework di valutazione.
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RAGASDataCollector:
    """Cattura i dati di ogni interazione e li struttura nel formato RAGAS.

    Per ogni domanda del test set, invoca l'agente, estrae i contesti
    recuperati tramite get_last_search_meta() e assembla il dizionario
    nel formato atteso da EvaluationDataset di RAGAS.
    """

    def __init__(self):
        """Inizializza il collettore dati."""
        self._collected_samples: List[Dict[str, Any]] = []
        logger.info("RAGASDataCollector inizializzato")

    @property
    def samples(self) -> List[Dict[str, Any]]:
        """Restituisce la lista dei sample raccolti."""
        return self._collected_samples

    def collect_from_chat(
        self,
        agent,
        question: str,
        ground_truth: str,
    ) -> Dict[str, Any]:
        """Raccoglie i dati di una singola interazione con l'agente.

        Invoca agent.chat(question), estrae i contesti recuperati dai
        metadati dell'ultima ricerca e assembla il dizionario nel formato
        richiesto da RAGAS.

        Args:
            agent: Istanza di RAGAgent da interrogare.
            question: La domanda da porre all'agente.
            ground_truth: La risposta attesa (reference) dal test set.

        Returns:
            Dizionario con le quattro variabili RAGAS:
            user_input, retrieved_contexts, response, reference.
        """
        logger.info("Raccolta dati per domanda: '%s'", question[:80])

        try:
            # Invoca l'agente con la domanda
            chat_result = agent.chat(question)
            response = chat_result.get("response", "")
            is_blocked = chat_result.get("blocked", False)

            if is_blocked:
                logger.warning(
                    "Risposta bloccata dal guardrail per: '%s'. "
                    "Uso la risposta di blocco come response.",
                    question[:80],
                )

            # Estrae i contesti recuperati dai metadati dell'ultima ricerca
            retrieved_contexts = self._extract_contexts(agent, chat_result)

            sample = {
                "user_input": question,
                "retrieved_contexts": retrieved_contexts,
                "response": response,
                "reference": ground_truth,
            }

            self._collected_samples.append(sample)

            logger.info(
                "Sample raccolto: %d contesti, risposta di %d caratteri",
                len(retrieved_contexts),
                len(response),
            )

            return sample

        except Exception as e:
            logger.error(
                "Errore durante la raccolta dati per '%s': %s",
                question[:80],
                e,
                exc_info=True,
            )
            # Restituisce un sample con valori vuoti in caso di errore
            error_sample = {
                "user_input": question,
                "retrieved_contexts": [],
                "response": f"Errore durante l'elaborazione: {str(e)}",
                "reference": ground_truth,
            }
            self._collected_samples.append(error_sample)
            return error_sample

    def _extract_contexts(self, agent, chat_result: Dict[str, Any]) -> List[str]:
        """Estrae i testi raw dei contesti recuperati dall'ultima ricerca.

        Utilizza tre strategie in ordine di priorita':
        1. Campo 'retrieved_texts' in get_last_search_meta() (aggiunto dalla modifica a tools.py)
        2. Campo 'retrieved_contexts' dal trace dict del chat_result
        3. Lista vuota come fallback

        Args:
            agent: Istanza di RAGAgent (usato per accedere ai metadati).
            chat_result: Dizionario restituito da agent.chat().

        Returns:
            Lista di stringhe con i testi dei contesti recuperati.
        """
        # Strategia 1: testi raw da get_last_search_meta()
        try:
            from agent.tools import get_last_search_meta
            search_meta = get_last_search_meta()
            retrieved_texts = search_meta.get("retrieved_texts", [])
            if retrieved_texts:
                logger.debug(
                    "Contesti estratti da search_meta: %d testi",
                    len(retrieved_texts),
                )
                return retrieved_texts
        except ImportError:
            logger.warning("Impossibile importare get_last_search_meta")
        except Exception as e:
            logger.warning("Errore accesso a search_meta: %s", e)

        # Strategia 2: contesti dal trace dict nel risultato della chat
        trace = chat_result.get("trace", {})
        if isinstance(trace, dict):
            contexts = trace.get("retrieved_contexts", [])
            if contexts:
                logger.debug(
                    "Contesti estratti dal trace dict: %d testi",
                    len(contexts),
                )
                return [ctx for ctx in contexts if ctx]

        # Strategia 3: fallback - lista vuota
        logger.warning(
            "Nessun contesto recuperato per la domanda corrente. "
            "Verificare che agent/tools.py esponga i testi raw dei chunk."
        )
        return []

    def collect_batch(
        self,
        agent,
        testset: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """Raccoglie i dati per tutte le domande del test set.

        Itera su ogni coppia question/ground_truth nel test set,
        invocando collect_from_chat per ciascuna. Resetta la memoria
        dell'agente prima di ogni domanda per garantire indipendenza
        tra le valutazioni.

        Args:
            agent: Istanza di RAGAgent da interrogare.
            testset: Lista di dizionari con chiavi 'question' e 'ground_truth'.

        Returns:
            Lista completa dei sample raccolti.
        """
        logger.info(
            "Avvio raccolta batch: %d domande nel test set",
            len(testset),
        )

        batch_results = []

        for idx, item in enumerate(testset, 1):
            question = item.get("question", "")
            ground_truth = item.get("ground_truth", "")

            if not question:
                logger.warning("Domanda vuota all'indice %d, skip.", idx)
                continue

            logger.info(
                "[%d/%d] Elaborazione: '%s'",
                idx,
                len(testset),
                question[:60],
            )

            # Resetta la memoria dell'agente per evitare contaminazione
            # tra le domande del test set
            try:
                agent.reset_memory()
            except Exception as e:
                logger.warning(
                    "Impossibile resettare la memoria dell'agente: %s", e
                )

            sample = self.collect_from_chat(agent, question, ground_truth)
            batch_results.append(sample)

            logger.info(
                "[%d/%d] Completata. Contesti: %d, Risposta: %d chars",
                idx,
                len(testset),
                len(sample.get("retrieved_contexts", [])),
                len(sample.get("response", "")),
            )

        logger.info(
            "Raccolta batch completata: %d/%d sample raccolti",
            len(batch_results),
            len(testset),
        )

        return batch_results

    def reset(self) -> None:
        """Resetta la lista dei sample raccolti."""
        self._collected_samples.clear()
        logger.info("RAGASDataCollector resettato")

    def to_list(self) -> List[Dict[str, Any]]:
        """Restituisce i sample raccolti come lista di dizionari.

        Formato compatibile con EvaluationDataset.from_list() di RAGAS.

        Returns:
            Lista di dizionari con le quattro variabili RAGAS.
        """
        return list(self._collected_samples)