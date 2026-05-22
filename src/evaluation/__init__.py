"""Modulo di valutazione (Evaluation) per il sistema RAG DIEM.

Fornisce strumenti per la valutazione automatizzata delle prestazioni
del sistema RAG utilizzando il framework RAGAS con un Judge LLM dedicato.

Componenti principali:
- EvaluationConfig: configurazione dell'evaluation
- RobustJudgeLLM: wrapper LLM con retry e JSON-repair per RAGAS
- EvaluationDatasetBuilder: caricamento domande e costruzione dataset
- EvaluationRunner: orchestratore principale dell'intero flusso
"""

from src.evaluation.config import EvaluationConfig
from src.evaluation.judge import create_judge_llm
from src.evaluation.dataset import EvaluationDatasetBuilder
from src.evaluation.runner import EvaluationRunner

__all__ = [
    "EvaluationConfig",
    "create_judge_llm",
    "EvaluationDatasetBuilder",
    "EvaluationRunner",
]