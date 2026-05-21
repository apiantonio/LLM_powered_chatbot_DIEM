"""Pipeline di valutazione RAGAS per l'Agente RAG DIEM.

Orchestra la valutazione end-to-end: raccolta dati, costruzione del
dataset RAGAS, esecuzione delle metriche e generazione dei report.

NOTA: Include un wrapper (CleanOutputChatModel) che intercetta e pulisce
gli output del LLM judge prima che RAGAS tenti il parsing JSON.

Risolve DUE classi di problemi:
1. Output sporco: tag <think>, markdown fences, prefissi testuali.
2. Struttura JSON non conforme: modelli locali che producono
   {"statements": ["str1", "str2"]} invece del formato Pydantic atteso
   {"statements": [{"text": "str1"}, {"text": "str2"}]}.
   Questo causa "validation error for StringIO / text Field required".
"""

import re
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Regex pre-compilate per la pulizia degli output LLM
# --------------------------------------------------------------------------- #
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
_LEADING_TEXT_BEFORE_JSON_RE = re.compile(
    r"^[^{\[]*?(?=[\{\[])", re.DOTALL
)


def _clean_llm_output(text: str) -> str:
    """Pulisce l'output di un LLM per facilitare il parsing JSON di RAGAS.

    Operazioni eseguite in ordine:
    1. Rimuove i tag <think>...</think> (modelli thinking).
    2. Estrae il contenuto da markdown code fences (```json ... ```).
    3. Rimuove eventuale testo prima del primo '{' o '['.
    4. Strip finale.

    Args:
        text: Testo raw dall'LLM.

    Returns:
        Testo pulito, pronto per il parsing JSON.
    """
    if not text:
        return text

    # 1. Rimuovi <think>...</think>
    cleaned = _THINK_TAG_RE.sub("", text)

    # 2. Estrai da markdown fences
    fence_match = _JSON_FENCE_RE.search(cleaned)
    if fence_match:
        cleaned = fence_match.group(1)

    # 3. Rimuovi testo prima del primo JSON token
    cleaned = cleaned.strip()
    if cleaned and cleaned[0] not in ('{', '['):
        match = _LEADING_TEXT_BEFORE_JSON_RE.match(cleaned)
        if match:
            cleaned = cleaned[match.end():]

    return cleaned.strip()


def _fix_ragas_json_structure(text: str) -> str:
    """Corregge la struttura JSON per conformarla agli schema Pydantic di RAGAS.

    I modelli locali (Ollama, Qwen, Nemotron) spesso producono JSON valido
    ma con struttura diversa da quella attesa da RAGAS. Questa funzione
    intercetta i pattern noti e li trasforma nel formato corretto.

    Trasformazioni applicate:

    1. statements: ["str"] -> statements: [{"text": "str"}]
       (richiesto da StringIO in Faithfulness/FactualCorrectness)

    2. verdicts: [{"statement": "...", "verdict": 1}]
       -> verdicts: [{"statement": "...", "verdict": 1, "reason": ""}]
       (aggiunge campo reason mancante se assente)

    3. {"question": "...", "answer": ...}
       -> invariato (gia' conforme per ResponseRelevancy)

    4. claims: ["str"] -> claims: [{"text": "str"}]
       (variante usata da alcune versioni di RAGAS)

    Args:
        text: JSON string (gia' pulita da _clean_llm_output).

    Returns:
        JSON string con struttura corretta per RAGAS.
    """
    if not text or not text.strip():
        return text

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        # Non e' JSON valido, restituisci invariato
        return text

    if not isinstance(data, dict):
        return text

    modified = False

    # --- Fix 1: statements come lista di stringhe -> lista di oggetti ---
    # RAGAS Faithfulness/FactualCorrectness usa StatementGeneratorOutput
    # che contiene una lista di StringIO, ognuno con campo "text".
    if "statements" in data and isinstance(data["statements"], list):
        new_statements = []
        for item in data["statements"]:
            if isinstance(item, str):
                new_statements.append({"text": item})
                modified = True
            elif isinstance(item, dict):
                # Gia' un oggetto, ma potrebbe mancare "text"
                if "text" not in item:
                    # Prova a trovare il campo giusto
                    if "statement" in item:
                        item["text"] = item["statement"]
                        modified = True
                    elif len(item) == 1:
                        # Singolo campo, usalo come text
                        item["text"] = next(iter(item.values()))
                        modified = True
                new_statements.append(item)
            else:
                new_statements.append({"text": str(item)})
                modified = True
        if modified:
            data["statements"] = new_statements

    # --- Fix 2: verdicts - assicura che ogni verdict abbia "reason" ---
    # RAGAS Faithfulness usa StatementFaithfulnessAnswer con campo reason
    if "verdicts" in data and isinstance(data["verdicts"], list):
        for verdict in data["verdicts"]:
            if isinstance(verdict, dict):
                if "reason" not in verdict:
                    verdict["reason"] = ""
                    modified = True
                # Normalizza verdict a intero se e' stringa
                if "verdict" in verdict and isinstance(verdict["verdict"], str):
                    v_lower = verdict["verdict"].lower().strip()
                    if v_lower in ("1", "yes", "true", "supported"):
                        verdict["verdict"] = 1
                        modified = True
                    elif v_lower in ("0", "no", "false", "not supported",
                                     "unsupported"):
                        verdict["verdict"] = 0
                        modified = True

    # --- Fix 3: claims come lista di stringhe -> lista di oggetti ---
    if "claims" in data and isinstance(data["claims"], list):
        new_claims = []
        for item in data["claims"]:
            if isinstance(item, str):
                new_claims.append({"text": item})
                modified = True
            elif isinstance(item, dict) and "text" not in item:
                if "claim" in item:
                    item["text"] = item["claim"]
                    modified = True
                new_claims.append(item)
            else:
                new_claims.append(item)
        if modified:
            data["claims"] = new_claims

    # --- Fix 4: sentences come lista di stringhe -> lista di oggetti ---
    if "sentences" in data and isinstance(data["sentences"], list):
        new_sentences = []
        for item in data["sentences"]:
            if isinstance(item, str):
                new_sentences.append({"text": item})
                modified = True
            elif isinstance(item, dict) and "text" not in item:
                if "sentence" in item:
                    item["text"] = item["sentence"]
                    modified = True
                new_sentences.append(item)
            else:
                new_sentences.append(item)
        if modified:
            data["sentences"] = new_sentences

    # --- Fix 5: score come stringa -> float ---
    if "score" in data and isinstance(data["score"], str):
        try:
            data["score"] = float(data["score"])
            modified = True
        except ValueError:
            pass

    if modified:
        result = json.dumps(data, ensure_ascii=False)
        logger.debug(
            "_fix_ragas_json_structure: struttura JSON corretta. "
            "Chiavi modificate nel payload."
        )
        return result

    return text


class CleanOutputChatModel(BaseChatModel):
    """Wrapper attorno a un BaseChatModel che pulisce gli output per RAGAS.

    Intercetta tutte le chiamate (invoke, generate, agenerate) e applica:
    1. _clean_llm_output: rimuove <think> tags, markdown fences, prefissi
    2. _fix_ragas_json_structure: corregge strutture JSON non conformi

    Questo garantisce che RAGAS riceva JSON pulito E strutturalmente
    conforme ai suoi schema Pydantic interni.
    """

    # Campo per il modello interno wrappato
    wrapped_model: BaseChatModel

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return f"clean_output_wrapper({self.wrapped_model._llm_type})"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return self.wrapped_model._identifying_params

    def _process_output(self, content: str) -> str:
        """Applica pulizia e fix strutturale al contenuto."""
        cleaned = _clean_llm_output(content)
        fixed = _fix_ragas_json_structure(cleaned)
        if fixed != content:
            logger.debug(
                "CleanOutputChatModel: output processato "
                "(%d chars -> %d chars)",
                len(content),
                len(fixed),
            )
        return fixed

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Override di _generate per pulire e fixare gli output."""
        result = self.wrapped_model._generate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )

        for generation in result.generations:
            if generation.message and isinstance(generation.message, AIMessage):
                original = generation.message.content
                processed = self._process_output(original)
                generation.message.content = processed
                generation.text = processed

        return result

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        """Override asincrono di _agenerate per pulire e fixare gli output."""
        result = await self.wrapped_model._agenerate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )

        for generation in result.generations:
            if generation.message and isinstance(generation.message, AIMessage):
                original = generation.message.content
                processed = self._process_output(original)
                generation.message.content = processed
                generation.text = processed

        return result

    def bind_tools(self, tools, **kwargs):
        """Delega bind_tools al modello wrappato."""
        return self.wrapped_model.bind_tools(tools, **kwargs)


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

    def _wrap_llm_judge(self, llm_judge):
        """Wrappa il LLM judge con CleanOutputChatModel per pulire gli output.

        Se il judge e' gia' un CleanOutputChatModel, lo restituisce invariato.

        Args:
            llm_judge: Istanza di BaseChatModel da wrappare.

        Returns:
            Istanza di CleanOutputChatModel che wrappa il judge originale.
        """
        if isinstance(llm_judge, CleanOutputChatModel):
            logger.debug("LLM judge gia' wrappato, skip.")
            return llm_judge

        wrapped = CleanOutputChatModel(wrapped_model=llm_judge)
        logger.info(
            "LLM judge wrappato con CleanOutputChatModel "
            "(pulizia output + fix struttura JSON per RAGAS)"
        )
        return wrapped

    def evaluate(self, collected_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Esegue la valutazione RAGAS sui dati raccolti.

        Il LLM judge viene automaticamente wrappato con CleanOutputChatModel
        per garantire che gli output siano puliti e strutturalmente conformi
        agli schema Pydantic interni di RAGAS.

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

        self._ensure_llm_judge()
        self._ensure_embedding_model()

        # --- Pulizia dei dati raccolti ---
        cleaned_data = []
        for sample in collected_data:
            cleaned_sample = dict(sample)
            contexts = cleaned_sample.get("retrieved_contexts", [])
            cleaned_sample["retrieved_contexts"] = [
                ctx for ctx in contexts
                if ctx and isinstance(ctx, str) and ctx.strip()
            ]
            if not cleaned_sample["retrieved_contexts"]:
                cleaned_sample["retrieved_contexts"] = [
                    "Nessun documento recuperato dalla knowledge base."
                ]
                logger.warning(
                    "Sample senza contesti per query: '%s'. "
                    "Aggiunto placeholder.",
                    cleaned_sample.get("user_input", "")[:60],
                )
            cleaned_data.append(cleaned_sample)

        try:
            from ragas import EvaluationDataset, evaluate
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper

            evaluation_dataset = EvaluationDataset.from_list(cleaned_data)
            logger.info(
                "EvaluationDataset RAGAS creato: %d sample",
                len(cleaned_data),
            )

            metrics = self._build_metrics()
            if not metrics:
                logger.error("Nessuna metrica valida configurata")
                return {"error": "Nessuna metrica configurata"}

            # Wrappa il judge con CleanOutputChatModel
            clean_judge = self._wrap_llm_judge(self._llm_judge)
            wrapped_llm = LangchainLLMWrapper(clean_judge)
            wrapped_embeddings = LangchainEmbeddingsWrapper(self._embedding_model)

            from ragas.run_config import RunConfig
            run_config = RunConfig(
                max_workers=self._max_workers,
                max_wait=240,
                max_retries=10,
            )
            logger.info(
                "RunConfig: max_workers=%d, max_wait=240s, max_retries=10",
                self._max_workers,
            )

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

            result_dict = {
                "scores": dict(result) if hasattr(result, '__iter__') else {},
                "num_samples": len(cleaned_data),
                "metrics_used": self._metrics_names,
            }

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
        run_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Esegue l'intero flusso di valutazione end-to-end.

        Args:
            agent: Istanza di RAGAgent da valutare.
            testset_path: Percorso del file testset.json.
            output_dir: Directory di output per i report.
            run_metadata: Dizionario opzionale con metadati aggiuntivi.

        Returns:
            Dizionario con i risultati completi della valutazione.
        """
        logger.info("=" * 60)
        logger.info("  AVVIO VALUTAZIONE COMPLETA RAGAS")
        logger.info("=" * 60)

        logger.info("Fase 1: Caricamento test set da '%s'", testset_path)
        testset = self._load_testset(testset_path)
        if not testset:
            return {"error": f"Test set vuoto o non trovato: {testset_path}"}
        logger.info("Test set caricato: %d domande", len(testset))

        logger.info("Fase 2: Raccolta dati tramite DataCollector")
        from evaluation.data_collector import RAGASDataCollector

        collector = RAGASDataCollector()
        collected_data = collector.collect_batch(agent, testset)
        logger.info("Dati raccolti: %d sample", len(collected_data))

        logger.info("Fase 3: Valutazione RAGAS")
        results = self.evaluate(collected_data)

        logger.info("Fase 4: Generazione report")
        from evaluation.report_generator import ReportGenerator

        report_gen = ReportGenerator(output_dir=output_dir)

        df = self.get_results_dataframe()
        if df is not None:
            report_gen.generate_console_summary(results)
            report_gen.generate_csv_report(df)
            report_gen.generate_markdown_report(results, df)
            report_gen.generate_json_summary(results, df, run_metadata)
        else:
            logger.warning(
                "DataFrame non disponibile, generazione report limitata"
            )
            report_gen.generate_console_summary(results)
            report_gen.generate_json_summary(results, run_metadata=run_metadata)

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