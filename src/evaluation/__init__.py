"""Modulo di valutazione RAGAS per l'Agente RAG DIEM.

Fornisce strumenti per la valutazione quantitativa delle performance
di retrieval e generazione dell'agente, basati sul framework RAGAS.
"""

from evaluation.data_collector import RAGASDataCollector
from evaluation.ragas_evaluator import RAGASEvaluator
from evaluation.report_generator import ReportGenerator

__all__ = [
    "RAGASDataCollector",
    "RAGASEvaluator",
    "ReportGenerator",
]

# ESEMPIO DI UTILIZZO:
#from evaluation import RAGASEvaluator

#evaluator = RAGASEvaluator(llm_judge=llm, embedding_model=embeddings)
#results = evaluator.run_full_evaluation(agent, "evaluation/testset.json")