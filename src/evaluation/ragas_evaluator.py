"""Pipeline di valutazione RAGAS per l'Agente RAG DIEM.

Orchestra la valutazione end-to-end: raccolta dati, costruzione del
dataset RAGAS, esecuzione delle metriche e generazione dei report.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RAGASEvaluator:
    """Orchestratore della valutazione RAGAS end-to-end.

    Gestisce la configurazione del LLM judge, le metriche di valutazione,
    l'esecuzione della valutazione e la generazione dei report.
    """

    def __init__(
        self,
        llm_judge=None,
        embedding_model=None,
        metrics: Optional[List[str]] = None,
        max_workers: int = 1,
    ):
        """Inizializza il valutatore RAGAS.

        Args:
            llm_judge: Modello LLM da usare come judge per le metriche RAGAS.
                       Se None, verra' creato automaticamente da settings.
            embedding_model: Modello di embedding per le metriche che lo richiedono.
                             Se None, verra' creato automaticamente da settings.
            metrics: Lista dei nomi delle metriche da calcolare. Se None,
                     usa tutte le metriche di default.
            max_workers: Numero massimo di chiamate LLM concorrenti durante
                         la valutazione. Impostare a 1 per API con rate limit
                         stringenti (es. Ollama cloud, Groq free tier).
                         Default: 1 (sequenziale, nessun rate limit).
        """
        self._llm_judge = llm_judge
        self._embedding_model = embedding_model
        self._metrics_names = metrics or [
            "context_precision",
            "context_recall",
            "response_relevancy",
            "faithfulness",
            "factual_correctness",
        ]
        self._max_workers = max(1, max_workers)
        self._results = None

        logger.info(
            "RAGASEvaluator inizializzato con metriche: %s, max_workers: %d",
            self._metrics_names,
            self._max_workers,
        )

    def _ensure_llm_judge(self) -> None:
        """Assicura che il LLM judge sia inizializzato.

        Se non fornito nel costruttore, tenta di crearlo dalla
        configurazione dell'applicazione.
        """
        if self._llm_judge is not None:
            return

        logger.info("LLM judge non fornito, creazione automatica da settings...")
        try:
            from config.settings import load_settings
            from agent.llm_providers import create_chat_model

            settings = load_settings()
            self._llm_judge = create_chat_model(settings.llm)
            logger.info("LLM judge creato automaticamente")
        except Exception as e:
            logger.error(
                "Impossibile creare LLM judge automaticamente: %s", e
            )
            raise RuntimeError(
                "LLM judge non disponibile. Fornirlo esplicitamente nel "
                "costruttore o configurare correttamente settings.yml."
            ) from e

    def _ensure_embedding_model(self) -> None:
        """Assicura che il modello di embedding sia inizializzato.

        Se non fornito nel costruttore, tenta di crearlo dalla
        configurazione dell'applicazione.
        """
        if self._embedding_model is not None:
            return

        logger.info("Embedding model non fornito, creazione automatica da settings...")
        try:
            from config.settings import load_settings
            from langchain_huggingface import HuggingFaceEmbeddings

            settings = load_settings()
            self._embedding_model = HuggingFaceEmbeddings(
                model_name=settings.embedding.model_name,
                encode_kwargs={
                    "normalize_embeddings": settings.embedding.normalize_embeddings,
                },
            )
            logger.info(
                "Embedding model creato automaticamente: %s",
                settings.embedding.model_name,
            )
        except Exception as e:
            logger.error(
                "Impossibile creare embedding model automaticamente: %s", e
            )
            raise RuntimeError(
                "Embedding model non disponibile. Fornirlo esplicitamente "
                "nel costruttore o configurare correttamente settings.yml."
            ) from e

    def _build_metrics(self) -> list:
        """Costruisce la lista di oggetti metrica RAGAS.

        Returns:
            Lista di istanze di metriche RAGAS configurate.
        """
        from ragas.metrics import (
            LLMContextPrecisionWithoutReference,
            LLMContextRecall,
            ResponseRelevancy,
            Faithfulness,
            FactualCorrectness,
        )

        metrics_map = {
            "context_precision": LLMContextPrecisionWithoutReference,
            "context_recall": LLMContextRecall,
            "response_relevancy": ResponseRelevancy,
            "faithfulness": Faithfulness,
            "factual_correctness": lambda: FactualCorrectness(mode="f1"),
        }

        metrics = []
        for name in self._metrics_names:
            if name in metrics_map:
                factory = metrics_map[name]
                if callable(factory) and not isinstance(factory, type):
                    metric = factory()
                else:
                    metric = factory()
                metrics.append(metric)
                logger.debug("Metrica aggiunta: %s", name)
            else:
                logger.warning(
                    "Metrica sconosciuta: '%s'. Metriche disponibili: %s",
                    name,
                    list(metrics_map.keys()),
                )

        logger.info("Metriche RAGAS configurate: %d", len(metrics))
        return metrics

    def evaluate(self, collected_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Esegue la valutazione RAGAS sui dati raccolti.

        Args:
            collected_data: Lista di dizionari con le quattro variabili RAGAS
                            (user_input, retrieved_contexts, response, reference).

        Returns:
            Dizionario con i risultati della valutazione, inclusi score
            aggregati e per-sample.
        """
        if not collected_data:
            logger.error("Nessun dato fornito per la valutazione")
            return {"error": "Nessun dato fornito"}

        logger.info(
            "Avvio valutazione RAGAS su %d sample", len(collected_data)
        )

        # Assicura che LLM e embedding siano disponibili
        self._ensure_llm_judge()
        self._ensure_embedding_model()

        try:
            from ragas import EvaluationDataset, evaluate
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper

            # Costruisce l'EvaluationDataset di RAGAS
            evaluation_dataset = EvaluationDataset.from_list(collected_data)
            logger.info(
                "EvaluationDataset RAGAS creato: %d sample",
                len(collected_data),
            )

            # Configura le metriche
            metrics = self._build_metrics()
            if not metrics:
                logger.error("Nessuna metrica valida configurata")
                return {"error": "Nessuna metrica configurata"}

            # Wrappa il LLM judge e l'embedding model per RAGAS
            wrapped_llm = LangchainLLMWrapper(self._llm_judge)
            wrapped_embeddings = LangchainEmbeddingsWrapper(self._embedding_model)

            # Configura il parallelismo per rispettare i rate limit dell'API
            from ragas.run_config import RunConfig
            run_config = RunConfig(
                max_workers=self._max_workers,
                max_wait=180,
                max_retries=5,
            )
            logger.info(
                "RunConfig: max_workers=%d, max_wait=180s, max_retries=5",
                self._max_workers,
            )

            # Esegue la valutazione
            logger.info("Esecuzione evaluate() di RAGAS in corso...")
            result = evaluate(
                dataset=evaluation_dataset,
                metrics=metrics,
                llm=wrapped_llm,
                embeddings=wrapped_embeddings,
                run_config=run_config,
            )

            self._results = result
            logger.info("Valutazione RAGAS completata con successo")

            # Estrae i risultati aggregati
            result_dict = {
                "scores": dict(result) if hasattr(result, '__iter__') else {},
                "num_samples": len(collected_data),
                "metrics_used": self._metrics_names,
            }

            # Log dei risultati aggregati
            if result_dict.get("scores"):
                logger.info("=" * 50)
                logger.info("  RISULTATI RAGAS AGGREGATI")
                logger.info("=" * 50)
                for metric_name, score in result_dict["scores"].items():
                    if isinstance(score, (int, float)):
                        logger.info("  %s: %.4f", metric_name, score)
                logger.info("=" * 50)

            return result_dict

        except ImportError as e:
            logger.error(
                "Impossibile importare RAGAS. Installare con: "
                "pip install ragas>=0.2.0. Errore: %s",
                e,
            )
            return {"error": f"RAGAS non installato: {e}"}

        except Exception as e:
            logger.error(
                "Errore durante la valutazione RAGAS: %s",
                e,
                exc_info=True,
            )
            return {"error": str(e)}

    def get_results_dataframe(self):
        """Restituisce i risultati come DataFrame Pandas.

        Returns:
            DataFrame con una riga per ogni sample e le colonne delle metriche,
            oppure None se la valutazione non e' stata eseguita.
        """
        if self._results is None:
            logger.warning(
                "Nessun risultato disponibile. Eseguire evaluate() prima."
            )
            return None

        try:
            df = self._results.to_pandas()
            logger.info(
                "DataFrame risultati: %d righe x %d colonne",
                len(df),
                len(df.columns),
            )
            return df
        except Exception as e:
            logger.error("Errore conversione risultati in DataFrame: %s", e)
            return None

    def run_full_evaluation(
        self,
        agent,
        testset_path: str = "evaluation/testset.json",
        output_dir: str = "evaluation/results",
    ) -> Dict[str, Any]:
        """Esegue l'intero flusso di valutazione: carica testset, raccoglie dati, valuta, genera report.

        Metodo di convenienza che orchestra tutte le fasi della valutazione
        in un'unica chiamata.

        Args:
            agent: Istanza di RAGAgent da valutare.
            testset_path: Percorso del file testset.json.
            output_dir: Directory di output per i report.

        Returns:
            Dizionario con i risultati completi della valutazione.
        """
        logger.info("=" * 60)
        logger.info("  AVVIO VALUTAZIONE COMPLETA RAGAS")
        logger.info("=" * 60)

        # Fase 1: Caricamento test set
        logger.info("Fase 1: Caricamento test set da '%s'", testset_path)
        testset = self._load_testset(testset_path)
        if not testset:
            return {"error": f"Test set vuoto o non trovato: {testset_path}"}
        logger.info("Test set caricato: %d domande", len(testset))

        # Fase 2: Raccolta dati
        logger.info("Fase 2: Raccolta dati tramite DataCollector")
        from evaluation.data_collector import RAGASDataCollector

        collector = RAGASDataCollector()
        collected_data = collector.collect_batch(agent, testset)
        logger.info("Dati raccolti: %d sample", len(collected_data))

        # Fase 3: Valutazione RAGAS
        logger.info("Fase 3: Valutazione RAGAS")
        results = self.evaluate(collected_data)

        # Fase 4: Generazione report
        logger.info("Fase 4: Generazione report")
        from evaluation.report_generator import ReportGenerator

        report_gen = ReportGenerator(output_dir=output_dir)

        df = self.get_results_dataframe()
        if df is not None:
            report_gen.generate_console_summary(results)
            report_gen.generate_csv_report(df)
            report_gen.generate_markdown_report(results, df)
        else:
            logger.warning(
                "DataFrame non disponibile, generazione report limitata"
            )
            report_gen.generate_console_summary(results)

        logger.info("=" * 60)
        logger.info("  VALUTAZIONE COMPLETA RAGAS TERMINATA")
        logger.info("=" * 60)

        return results

    @staticmethod
    def _load_testset(testset_path: str) -> List[Dict[str, str]]:
        """Carica il test set da file JSON.

        Args:
            testset_path: Percorso del file JSON contenente il test set.

        Returns:
            Lista di dizionari con chiavi 'question' e 'ground_truth'.
        """
        path = Path(testset_path)

        if not path.exists():
            logger.error("File test set non trovato: %s", testset_path)
            return []

        try:
            with open(path, "r", encoding="utf-8") as f:
                testset = json.load(f)

            if not isinstance(testset, list):
                logger.error(
                    "Il test set deve essere una lista JSON. Tipo trovato: %s",
                    type(testset).__name__,
                )
                return []

            # Validazione delle chiavi richieste
            valid_items = []
            for idx, item in enumerate(testset):
                if "question" not in item:
                    logger.warning(
                        "Elemento %d del test set senza chiave 'question', skip.",
                        idx,
                    )
                    continue
                if "ground_truth" not in item:
                    logger.warning(
                        "Elemento %d del test set senza chiave 'ground_truth', skip.",
                        idx,
                    )
                    continue
                valid_items.append(item)

            logger.info(
                "Test set caricato: %d/%d elementi validi",
                len(valid_items),
                len(testset),
            )
            return valid_items

        except json.JSONDecodeError as e:
            logger.error("Errore parsing JSON del test set: %s", e)
            return []
        except Exception as e:
            logger.error("Errore caricamento test set: %s", e)
            return []