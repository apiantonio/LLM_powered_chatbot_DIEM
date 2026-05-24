"""Caricamento dataset e costruzione EvaluationDataset per RAGAS.

Gestisce il flusso completo dalla lettura del file JSON di input
(domande + ground truth) all'interazione con l'agente RAG per
ottenere risposte e contesti recuperati, fino alla costruzione
dell'EvaluationDataset richiesto dal framework RAGAS.

Formato JSON di input atteso:
{
    "dataset_name": "Nome descrittivo del dataset",
    "description": "Descrizione opzionale",
    "samples": [
        {
            "question": "Domanda per il sistema RAG",
            "ground_truth": "Risposta attesa di riferimento"
        },
        ...
    ]
}
"""

import json
import re
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from src.evaluation.config import EvaluationConfig

logger = logging.getLogger(__name__)


_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]+\)")

_STANDALONE_URL_RE = re.compile(
    r"https?://[^\s<>\"\')}\]]+",
    re.IGNORECASE,
)

_MD_HEADER_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)

_MD_BOLD_ITALIC_RE = re.compile(r"(\*{1,3}|_{1,3})(.*?)\1")

_MD_STRIKETHROUGH_RE = re.compile(r"~~(.*?)~~")

_MD_INLINE_CODE_RE = re.compile(r"`([^`]+)`")

_MD_CODE_BLOCK_RE = re.compile(r"```[a-z]*\n?.*?\n?```", re.DOTALL)

_MD_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+", re.MULTILINE)

_MD_BLOCKQUOTE_RE = re.compile(r"^\s*>\s?", re.MULTILINE)

_MD_HR_RE = re.compile(r"^\s*[-*_]{3,}\s*$", re.MULTILINE)

_HTML_TAG_RE = re.compile(r"<[^>]+>")

_MULTI_SPACE_RE = re.compile(r"[ \t]+")

_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def normalize_text_for_comparison(text: str) -> str:
    """Normalizza il testo rimuovendo formattazione markdown e link URL.

    Trasformazioni applicate (in ordine):
    1. Blocchi di codice rimossi
    2. Link markdown [testo](url) -> testo (preserva il testo informativo)
    3. URL standalone rimossi
    4. Header markdown (# ## ###) rimossi
    5. Bold/italic (** * __ _) rimossi, testo preservato
    6. Strikethrough (~~) rimosso, testo preservato
    7. Inline code (``) rimosso, testo preservato
    8. Marcatori di lista (- * + 1.) rimossi
    9. Blockquote (>) rimossi
    10. Linee orizzontali (--- ***) rimosse
    11. Tag HTML residui rimossi
    12. Normalizzazione case -> lowercase
    13. Normalizzazione spazi e righe vuote

    Args:
        text: Testo da normalizzare (response o ground_truth).

    Returns:
        Testo normalizzato pronto per il confronto.
    """
    if not text:
        return ""

    result = text

    result = _MD_CODE_BLOCK_RE.sub("", result)

    result = _MD_LINK_RE.sub(r"\1", result)

    result = _STANDALONE_URL_RE.sub("", result)

    result = _MD_HEADER_RE.sub("", result)

    for _ in range(3):
        result = _MD_BOLD_ITALIC_RE.sub(r"\2", result)

    result = _MD_STRIKETHROUGH_RE.sub(r"\1", result)

    result = _MD_INLINE_CODE_RE.sub(r"\1", result)

    result = _MD_LIST_RE.sub("", result)

    result = _MD_BLOCKQUOTE_RE.sub("", result)

    result = _MD_HR_RE.sub("", result)

    result = _HTML_TAG_RE.sub("", result)

    result = result.lower()

    result = _MULTI_SPACE_RE.sub(" ", result)
    result = _MULTI_NEWLINE_RE.sub("\n\n", result)
    result = result.strip()

    return result


class DatasetValidationError(Exception):
    """Eccezione sollevata quando il file JSON di input non e' valido."""
    pass


def _validate_input_schema(data: dict) -> None:
    """Valida la struttura del file JSON di input.

    Verifica che il JSON contenga il campo 'samples' con la struttura
    attesa e che ogni sample abbia i campi obbligatori.

    Args:
        data: Dizionario caricato dal file JSON.

    Raises:
        DatasetValidationError: Se la struttura non e' conforme.
    """
    if not isinstance(data, dict):
        raise DatasetValidationError(
            "Il file JSON deve contenere un oggetto radice, "
            f"trovato: {type(data).__name__}"
        )

    if "samples" not in data:
        raise DatasetValidationError(
            "Campo obbligatorio 'samples' mancante nel JSON. "
            "Struttura attesa: {\"samples\": [{\"question\": ..., \"ground_truth\": ...}]}"
        )

    samples = data["samples"]
    if not isinstance(samples, list):
        raise DatasetValidationError(
            f"Il campo 'samples' deve essere una lista, trovato: {type(samples).__name__}"
        )

    if len(samples) == 0:
        raise DatasetValidationError("Il campo 'samples' e' vuoto. Inserisci almeno una domanda.")

    for i, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise DatasetValidationError(
                f"Sample #{i + 1}: deve essere un oggetto, trovato: {type(sample).__name__}"
            )

        if "question" not in sample:
            raise DatasetValidationError(
                f"Sample #{i + 1}: campo obbligatorio 'question' mancante."
            )

        if "ground_truth" not in sample:
            raise DatasetValidationError(
                f"Sample #{i + 1}: campo obbligatorio 'ground_truth' mancante."
            )

        if not sample["question"].strip():
            raise DatasetValidationError(
                f"Sample #{i + 1}: il campo 'question' e' vuoto."
            )

        if not sample["ground_truth"].strip():
            raise DatasetValidationError(
                f"Sample #{i + 1}: il campo 'ground_truth' e' vuoto."
            )



class ProcessedSample:
    """Rappresenta un singolo sample elaborato dal sistema RAG.

    Contiene la domanda originale, la ground truth, la risposta del
    sistema, i contesti recuperati e i metadati di tracciamento.

    Attributes:
        index: Indice progressivo del sample nel dataset.
        question: Domanda originale dal file JSON.
        ground_truth: Risposta attesa di riferimento.
        response: Risposta generata dal sistema RAG.
        retrieved_contexts: Lista di testi dei chunk recuperati.
        trace: Dizionario con metadati di tracciamento dall'agente.
        processing_time_ms: Tempo di elaborazione in millisecondi.
        was_blocked: Se True, la risposta e' stata bloccata da un guardrail.
        block_reason: Motivo del blocco (se was_blocked e' True).
        error: Messaggio di errore se l'elaborazione e' fallita.
    """

    def __init__(
        self,
        index: int,
        question: str,
        ground_truth: str,
    ):
        self.index = index
        self.question = question
        self.ground_truth = ground_truth
        self.response: str = ""
        self.retrieved_contexts: List[str] = []
        self.trace: Dict[str, Any] = {}
        self.processing_time_ms: float = 0.0
        self.was_blocked: bool = False
        self.block_reason: Optional[str] = None
        self.error: Optional[str] = None

    def to_ragas_dict(self, enable_normalization: bool = True) -> Dict[str, Any]:
        """Converte il sample nel formato atteso da RAGAS EvaluationDataset.

        Quando la normalizzazione e' abilitata, response e reference
        vengono ripuliti da formattazione markdown, link URL e differenze
        di case prima del confronto RAGAS.

        Args:
            enable_normalization: Se True, normalizza response e reference.

        Returns:
            Dizionario con chiavi user_input, retrieved_contexts, response, reference.
        """
        response = self.response if self.response else ""
        reference = self.ground_truth

        if enable_normalization:
            response = normalize_text_for_comparison(response)
            reference = normalize_text_for_comparison(reference)

        return {
            "user_input": self.question,
            "retrieved_contexts": self.retrieved_contexts if self.retrieved_contexts else [""],
            "response": response,
            "reference": reference,
        }

    def to_full_dict(self) -> Dict[str, Any]:
        """Serializza il sample completo con tutti i metadati.

        Returns:
            Dizionario completo per il file JSON di output.
        """
        return {
            "index": self.index,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "response": self.response,
            "response_normalized": normalize_text_for_comparison(self.response),
            "ground_truth_normalized": normalize_text_for_comparison(self.ground_truth),
            "retrieved_contexts": self.retrieved_contexts,
            "was_blocked": self.was_blocked,
            "block_reason": self.block_reason,
            "error": self.error,
            "processing_time_ms": round(self.processing_time_ms, 1),
            "trace": {
                "tool_name": self.trace.get("tool_name", ""),
                "tools_invoked": [
                    t.get("name", "") for t in self.trace.get("tools", [])
                ],
                "rewritten_query": self.trace.get("query_rewritten", ""),
                "collection": self.trace.get("collection", ""),
                "source_urls": self.trace.get("source_urls", []),
                "llm_calls": self.trace.get("llm_calls", 0),
                "total_duration_ms": self.trace.get("total_duration_ms", 0),
            },
        }


class EvaluationDatasetBuilder:
    """Costruisce un EvaluationDataset RAGAS a partire dal file JSON di input.

    Il flusso e':
    1. Carica e valida il file JSON
    2. Per ogni domanda, invoca agent.chat() per ottenere risposta e contesti
    3. Raccoglie i retrieved_contexts dal modulo tools (_last_search_meta)
    4. Costruisce l'EvaluationDataset di RAGAS
    5. Supporta salvataggio intermedio per recovery

    Attributes:
        config: Configurazione dell'evaluation.
        processed_samples: Lista dei sample elaborati.
        dataset_metadata: Metadati del dataset (nome, descrizione, timestamp).
    """

    def __init__(self, config: EvaluationConfig):
        """Inizializza il builder con la configurazione.

        Args:
            config: Configurazione completa dell'evaluation.
        """
        self.config = config
        self.processed_samples: List[ProcessedSample] = []
        self.dataset_metadata: Dict[str, Any] = {}
        self._raw_samples: List[Dict[str, str]] = []

    def load_from_json(self, filepath: Optional[str] = None) -> int:
        """Carica e valida il file JSON con domande e ground truth.

        Args:
            filepath: Percorso del file JSON. Se None, usa il percorso
                     dalla configurazione.

        Returns:
            Numero di sample caricati.

        Raises:
            FileNotFoundError: Se il file non esiste.
            DatasetValidationError: Se la struttura JSON non e' valida.
            json.JSONDecodeError: Se il file non e' JSON valido.
        """
        path = Path(filepath or self.config.input_file)

        if not path.exists():
            raise FileNotFoundError(
                f"File di input non trovato: {path}. "
                f"Crea il file con la struttura: "
                f"{{\"samples\": [{{\"question\": ..., \"ground_truth\": ...}}]}}"
            )

        logger.info("Caricamento dataset da: %s", path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        _validate_input_schema(data)

        self._raw_samples = data["samples"]
        self.dataset_metadata = {
            "dataset_name": data.get("dataset_name", path.stem),
            "description": data.get("description", ""),
            "source_file": str(path),
            "num_samples": len(self._raw_samples),
            "loaded_at": datetime.now().isoformat(),
        }

        logger.info(
            "Dataset caricato: '%s' — %d domande",
            self.dataset_metadata["dataset_name"],
            len(self._raw_samples),
        )

        return len(self._raw_samples)

    def process_with_agent(self, agent) -> List[ProcessedSample]:
        """Elabora tutte le domande attraverso l'agente RAG.

        Per ogni domanda nel dataset:
        1. Invoca agent.chat() per ottenere la risposta
        2. Estrae i contesti recuperati da agent.tools._last_search_meta
        3. Registra tempi, errori, blocchi guardrail
        4. Salva risultati intermedi se configurato

        L'agente viene resettato prima dell'elaborazione per garantire
        che ogni domanda parta da uno stato pulito senza memoria.

        Args:
            agent: Istanza di RAGAgent configurata e pronta.

        Returns:
            Lista di ProcessedSample elaborati.
        """
        if not self._raw_samples:
            raise ValueError(
                "Nessun sample caricato. Chiama load_from_json() prima di process_with_agent()."
            )

        from src.agent.tools import get_last_search_meta

        total = len(self._raw_samples)
        self.processed_samples = []

        logger.info("=" * 60)
        logger.info("  INIZIO ELABORAZIONE: %d domande", total)
        logger.info("=" * 60)

        for i, raw_sample in enumerate(self._raw_samples):
            sample = ProcessedSample(
                index=i + 1,
                question=raw_sample["question"],
                ground_truth=raw_sample["ground_truth"],
            )

            logger.info(
                "[%d/%d] Elaborazione: '%s'",
                i + 1, total, sample.question[:80],
            )

            agent.reset_memory()

            start_time = time.time()

            try:
                result = agent.chat(sample.question)

                sample.response = result.get("response", "")
                sample.was_blocked = result.get("blocked", False)
                sample.block_reason = result.get("block_reason")

                search_meta = get_last_search_meta()
                sample.retrieved_contexts = search_meta.get("retrieved_texts", [])

                traces = agent.get_all_traces()
                if traces:
                    sample.trace = traces[-1]

                sample.processing_time_ms = (time.time() - start_time) * 1000

                if sample.was_blocked:
                    logger.warning(
                        "[%d/%d] BLOCCATO (%s): '%s'",
                        i + 1, total, sample.block_reason, sample.question[:60],
                    )
                else:
                    logger.info(
                        "[%d/%d] OK (%.0fms, %d contesti): '%s'",
                        i + 1, total,
                        sample.processing_time_ms,
                        len(sample.retrieved_contexts),
                        sample.response[:80],
                    )

            except Exception as e:
                sample.processing_time_ms = (time.time() - start_time) * 1000
                sample.error = str(e)
                logger.error(
                    "[%d/%d] ERRORE: %s — '%s'",
                    i + 1, total, e, sample.question[:60],
                )

            self.processed_samples.append(sample)

            if self.config.pipeline.save_intermediate:
                self._save_intermediate(sample)

            if i < total - 1 and self.config.pipeline.batch_delay_seconds > 0:
                time.sleep(self.config.pipeline.batch_delay_seconds)

        logger.info("=" * 60)
        logger.info("  ELABORAZIONE COMPLETATA: %d/%d domande", len(self.processed_samples), total)
        logger.info("=" * 60)

        return self.processed_samples

    def build_ragas_dataset(self):
        """Costruisce l'EvaluationDataset di RAGAS dai sample elaborati.

        Filtra i sample con errori o bloccati dai guardrail, poiche'
        non hanno risposte valide per la valutazione.

        La normalizzazione del testo (rimozione markdown, URL, case)
        viene applicata in base alla configurazione
        (config.normalization.enable_normalization).

        Returns:
            Istanza di EvaluationDataset pronta per evaluate().

        Raises:
            ValueError: Se non ci sono sample elaborati.
            ImportError: Se ragas non e' installato.
        """
        if not self.processed_samples:
            raise ValueError(
                "Nessun sample elaborato. Chiama process_with_agent() prima."
            )

        try:
            from ragas import EvaluationDataset
        except ImportError as e:
            raise ImportError(
                "ragas non installato. Installa con: pip install ragas"
            ) from e

        valid_samples = [
            s for s in self.processed_samples
            if not s.was_blocked and s.error is None and s.response.strip()
        ]

        skipped = len(self.processed_samples) - len(valid_samples)
        if skipped > 0:
            logger.warning(
                "%d sample esclusi dalla valutazione RAGAS "
                "(bloccati, errori, o risposta vuota).",
                skipped,
            )

        if not valid_samples:
            raise ValueError(
                "Nessun sample valido per la valutazione RAGAS. "
                "Tutti i sample sono stati bloccati o hanno generato errori."
            )

        enable_norm = self.config.normalization.enable_normalization
        if enable_norm:
            logger.info(
                "Normalizzazione testo ATTIVA: rimozione markdown, URL, "
                "normalizzazione case per confronto RAGAS."
            )
        else:
            logger.info("Normalizzazione testo DISATTIVA: confronto raw.")

        ragas_list = [
            s.to_ragas_dict(enable_normalization=enable_norm)
            for s in valid_samples
        ]

        dataset = EvaluationDataset.from_list(ragas_list)

        logger.info(
            "EvaluationDataset RAGAS costruito: %d sample validi su %d totali.",
            len(valid_samples),
            len(self.processed_samples),
        )

        return dataset

    def get_all_processed(self) -> List[Dict[str, Any]]:
        """Restituisce tutti i sample elaborati come lista di dizionari.

        Include anche i sample bloccati o con errori per completezza
        del report.

        Returns:
            Lista di dizionari con tutti i dati di ogni sample.
        """
        return [s.to_full_dict() for s in self.processed_samples]

    def get_valid_indices(self) -> List[int]:
        """Restituisce gli indici (0-based) dei sample validi per RAGAS.

        Necessario per riallineare i risultati di RAGAS (che riceve
        solo i sample validi) con la lista completa dei sample.

        Returns:
            Lista di indici dei sample non bloccati e senza errori.
        """
        return [
            i for i, s in enumerate(self.processed_samples)
            if not s.was_blocked and s.error is None and s.response.strip()
        ]

    def _save_intermediate(self, sample: ProcessedSample) -> None:
        """Salva un sample elaborato come file intermedio per recovery.

        In caso di interruzione, i file intermedi permettono di
        riprendere l'elaborazione senza rielaborare domande gia'
        completate.

        Args:
            sample: Sample elaborato da salvare.
        """
        try:
            intermediate_dir = Path(self.config.pipeline.intermediate_dir)
            intermediate_dir.mkdir(parents=True, exist_ok=True)

            filename = f"sample_{sample.index:04d}.json"
            filepath = intermediate_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(sample.to_full_dict(), f, indent=2, ensure_ascii=False)

            logger.debug("Intermedio salvato: %s", filepath)

        except Exception as e:
            logger.warning("Errore salvataggio intermedio sample #%d: %s", sample.index, e)

    def load_intermediate_results(self) -> int:
        """Carica i risultati intermedi salvati da un'esecuzione precedente.

        Utile per riprendere un'elaborazione interrotta. I sample gia'
        elaborati vengono caricati e aggiunti a processed_samples.

        Returns:
            Numero di sample intermedi caricati.
        """
        intermediate_dir = Path(self.config.pipeline.intermediate_dir)

        if not intermediate_dir.exists():
            return 0

        loaded = 0
        for filepath in sorted(intermediate_dir.glob("sample_*.json")):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                sample = ProcessedSample(
                    index=data["index"],
                    question=data["question"],
                    ground_truth=data["ground_truth"],
                )
                sample.response = data.get("response", "")
                sample.retrieved_contexts = data.get("retrieved_contexts", [])
                sample.was_blocked = data.get("was_blocked", False)
                sample.block_reason = data.get("block_reason")
                sample.error = data.get("error")
                sample.processing_time_ms = data.get("processing_time_ms", 0)
                sample.trace = data.get("trace", {})

                self.processed_samples.append(sample)
                loaded += 1

            except Exception as e:
                logger.warning("Errore caricamento intermedio %s: %s", filepath, e)

        if loaded > 0:
            logger.info(
                "Caricati %d risultati intermedi da: %s",
                loaded, intermediate_dir,
            )

        return loaded

    def clear_intermediate(self) -> None:
        """Rimuove tutti i file intermedi dalla directory.

        Da chiamare dopo un'elaborazione completata con successo.
        """
        intermediate_dir = Path(self.config.pipeline.intermediate_dir)

        if not intermediate_dir.exists():
            return

        count = 0
        for filepath in intermediate_dir.glob("sample_*.json"):
            try:
                filepath.unlink()
                count += 1
            except Exception as e:
                logger.warning("Errore rimozione %s: %s", filepath, e)

        if count > 0:
            logger.info("Rimossi %d file intermedi.", count)