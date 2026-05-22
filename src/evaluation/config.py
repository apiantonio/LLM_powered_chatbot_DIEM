"""Configurazione del modulo di valutazione RAG DIEM.

Definisce i parametri configurabili per l'intero flusso di evaluation:
Judge LLM, metriche RAGAS, rate limiting, percorsi di I/O e retry policy.

La configurazione viene caricata da variabili d'ambiente con fallback
su valori di default sensati per il contesto DIEM.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JudgeLLMConfig:
    """Configurazione del modello LLM utilizzato come Judge per RAGAS.

    Il Judge e' il modello che valuta la qualita' delle risposte.
    Deve essere affidabile nella produzione di output JSON strutturato.

    Attributes:
        provider: Provider del Judge ('groq' raccomandato per affidabilita' JSON).
        model_name: Nome del modello Judge.
        api_key: Chiave API per il provider. Se None, viene letta da env.
        temperature: Temperatura di generazione (0.0 per massima determinismo).
        max_tokens: Limite massimo di token per risposta del Judge.
        request_timeout: Timeout in secondi per singola richiesta.
    """

    provider: str = "groq"
    model_name: str = "llama-3.3-70b-versatile"
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 2048
    request_timeout: int = 120


@dataclass(frozen=True)
class RetryConfig:
    """Configurazione della policy di retry per le chiamate al Judge LLM.

    Gestisce i tentativi di ripetizione in caso di errori di parsing JSON
    o di rate limiting da parte del provider API.

    Attributes:
        max_retries: Numero massimo di tentativi per singola chiamata.
        base_delay_seconds: Ritardo base tra tentativi (backoff esponenziale).
        max_delay_seconds: Ritardo massimo tra tentativi.
        retry_on_rate_limit: Se True, attende e riprova su errori 429.
        rate_limit_wait_seconds: Attesa fissa su rate limit prima del retry.
    """

    max_retries: int = 3
    base_delay_seconds: float = 2.0
    max_delay_seconds: float = 60.0
    retry_on_rate_limit: bool = True
    rate_limit_wait_seconds: float = 30.0


@dataclass(frozen=True)
class MetricsConfig:
    """Configurazione delle metriche RAGAS da calcolare.

    Attributes:
        enable_context_precision: Precisione dei contesti recuperati.
        enable_context_recall: Recall dei contesti rispetto alla ground truth.
        enable_response_relevancy: Rilevanza della risposta alla domanda.
        enable_faithfulness: Fedelta' della risposta ai contesti recuperati.
        enable_factual_correctness: Correttezza fattuale rispetto alla ground truth.
        factual_correctness_mode: Modalita' di calcolo ('precision', 'recall', 'f1').
    """

    enable_context_precision: bool = True
    enable_context_recall: bool = True
    enable_response_relevancy: bool = True
    enable_faithfulness: bool = True
    enable_factual_correctness: bool = True
    factual_correctness_mode: str = "f1"


@dataclass(frozen=True)
class OutputConfig:
    """Configurazione dei percorsi e formati di output dell'evaluation.

    Attributes:
        output_dir: Directory per i file di output generati.
        export_csv: Se True, esporta il report in formato CSV.
        export_excel: Se True, esporta il report in formato Excel (.xlsx).
        export_json: Se True, esporta le metriche dettagliate in JSON.
        json_indent: Indentazione del file JSON di output.
    """

    output_dir: str = "results/evaluation"
    export_csv: bool = True
    export_excel: bool = True
    export_json: bool = True
    json_indent: int = 2


@dataclass(frozen=True)
class PipelineConfig:
    """Configurazione del comportamento della pipeline di evaluation.

    Attributes:
        batch_delay_seconds: Pausa tra l'elaborazione di domande successive
            per rispettare i rate limit delle API.
        show_progress: Se True, mostra una barra di progresso durante l'esecuzione.
        save_intermediate: Se True, salva risultati intermedi per recovery
            in caso di interruzione.
        intermediate_dir: Directory per i file intermedi.
    """

    batch_delay_seconds: float = 2.0
    show_progress: bool = True
    save_intermediate: bool = True
    intermediate_dir: str = "results/evaluation/.intermediate"


@dataclass(frozen=True)
class EvaluationConfig:
    """Aggregatore di tutte le sezioni di configurazione dell'evaluation.

    Attributes:
        judge: Configurazione del modello Judge LLM.
        retry: Policy di retry per le chiamate API.
        metrics: Metriche RAGAS da calcolare.
        output: Percorsi e formati di output.
        pipeline: Comportamento della pipeline.
        input_file: Percorso del file JSON con domande e ground truth.
    """

    judge: JudgeLLMConfig = field(default_factory=JudgeLLMConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    input_file: str = "data/evaluation/questions.json"


def load_evaluation_config() -> EvaluationConfig:
    """Carica la configurazione dell'evaluation da variabili d'ambiente.

    Le variabili d'ambiente sovrascrivono i valori di default.
    Prefisso: EVAL_ per tutte le variabili del modulo.

    Per la API key del Judge, la priorita' di lettura e':
    1. EVAL_JUDGE_API_KEY (dedicata all'evaluation)
    2. GROQ_REWRITER_API_KEY (riutilizzo chiave rewriter)
    3. GROQ_GUARDRAILS_API_KEY (riutilizzo chiave guardrails)
    4. GROQ_CHAT_API_KEY (riutilizzo chiave chat)

    Returns:
        Istanza di EvaluationConfig completamente configurata.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    judge_api_key = (
        os.getenv("EVAL_JUDGE_API_KEY")
        or os.getenv("GROQ_REWRITER_API_KEY")
        or os.getenv("GROQ_GUARDRAILS_API_KEY")
        or os.getenv("GROQ_CHAT_API_KEY")
    )

    if not judge_api_key:
        logger.warning(
            "Nessuna API key trovata per il Judge LLM. "
            "Configura EVAL_JUDGE_API_KEY o una delle chiavi GROQ_* nel .env."
        )

    config = EvaluationConfig(
        judge=JudgeLLMConfig(
            provider=os.getenv("EVAL_JUDGE_PROVIDER", "groq"),
            model_name=os.getenv("EVAL_JUDGE_MODEL", "llama-3.3-70b-versatile"),
            api_key=judge_api_key,
            temperature=float(os.getenv("EVAL_JUDGE_TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("EVAL_JUDGE_MAX_TOKENS", "2048")),
            request_timeout=int(os.getenv("EVAL_JUDGE_TIMEOUT", "120")),
        ),
        retry=RetryConfig(
            max_retries=int(os.getenv("EVAL_MAX_RETRIES", "3")),
            base_delay_seconds=float(os.getenv("EVAL_RETRY_BASE_DELAY", "2.0")),
            max_delay_seconds=float(os.getenv("EVAL_RETRY_MAX_DELAY", "60.0")),
            retry_on_rate_limit=os.getenv("EVAL_RETRY_ON_RATE_LIMIT", "true").lower() == "true",
            rate_limit_wait_seconds=float(os.getenv("EVAL_RATE_LIMIT_WAIT", "30.0")),
        ),
        metrics=MetricsConfig(
            enable_context_precision=os.getenv("EVAL_METRIC_CTX_PRECISION", "true").lower() == "true",
            enable_context_recall=os.getenv("EVAL_METRIC_CTX_RECALL", "true").lower() == "true",
            enable_response_relevancy=os.getenv("EVAL_METRIC_RESPONSE_REL", "true").lower() == "true",
            enable_faithfulness=os.getenv("EVAL_METRIC_FAITHFULNESS", "true").lower() == "true",
            enable_factual_correctness=os.getenv("EVAL_METRIC_FACTUAL", "true").lower() == "true",
            factual_correctness_mode=os.getenv("EVAL_FACTUAL_MODE", "f1"),
        ),
        output=OutputConfig(
            output_dir=os.getenv("EVAL_OUTPUT_DIR", "results/evaluation"),
            export_csv=os.getenv("EVAL_EXPORT_CSV", "true").lower() == "true",
            export_excel=os.getenv("EVAL_EXPORT_EXCEL", "true").lower() == "true",
            export_json=os.getenv("EVAL_EXPORT_JSON", "true").lower() == "true",
        ),
        pipeline=PipelineConfig(
            batch_delay_seconds=float(os.getenv("EVAL_BATCH_DELAY", "2.0")),
            show_progress=os.getenv("EVAL_SHOW_PROGRESS", "true").lower() == "true",
            save_intermediate=os.getenv("EVAL_SAVE_INTERMEDIATE", "true").lower() == "true",
            intermediate_dir=os.getenv("EVAL_INTERMEDIATE_DIR", "results/evaluation/.intermediate"),
        ),
        input_file=os.getenv("EVAL_INPUT_FILE", "data/evaluation/questions.json"),
    )

    logger.info("Configurazione Evaluation caricata:")
    logger.info("  Judge: %s/%s (temperature=%.1f)", config.judge.provider, config.judge.model_name, config.judge.temperature)
    logger.info("  API Key: %s", "CONFIGURATA" if config.judge.api_key else "MANCANTE")
    logger.info("  Retry: max=%d, base_delay=%.1fs", config.retry.max_retries, config.retry.base_delay_seconds)
    logger.info("  Input: %s", config.input_file)
    logger.info("  Output: %s", config.output.output_dir)

    return config