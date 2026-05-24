"""Valutazione dei Guardrails del sistema RAG DIEM.

Carica un dataset di prompt etichettati, esegue i tre check del
GuardrailsChecker (input, output, meta) e calcola accuracy, precision,
recall e F1-score per ciascun check.

Il checker utilizzato e' quello reale di produzione (Groq/Llama 3.3 70B),
istanziato tramite la factory build_guardrails_checker esistente.
"""

import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from src.config.settings import load_settings
from src.agent.guardrails import build_guardrails_checker, GuardrailsChecker

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Calcolo metriche
# --------------------------------------------------------------------------- #

@dataclass
class CheckMetrics:
    """Metriche di classificazione per un singolo check."""

    check_type: str
    total: int = 0
    correct: int = 0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_type": self.check_type,
            "total": self.total,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "confusion": {"tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn},
            "errors": self.errors,
        }


# --------------------------------------------------------------------------- #
#  Runner
# --------------------------------------------------------------------------- #

class GuardrailsEvaluationRunner:
    """Orchestratore della valutazione dei guardrails.

    Carica il dataset, istanzia il GuardrailsChecker di produzione,
    esegue i check e produce un report con le metriche.
    """

    def __init__(self, testset_path: str = "data/evaluation/guardrails_testset.json"):
        self._testset_path = testset_path
        self._checker: Optional[GuardrailsChecker] = None
        self._samples: List[Dict[str, str]] = []
        self._metrics: Dict[str, CheckMetrics] = {}
        self._run_metadata: Dict[str, Any] = {}
        self._latencies: List[float] = []

    def run(self) -> Dict[str, Any]:
        """Esegue il flusso completo: caricamento, check, metriche, report."""
        start_time = time.time()
        self._run_metadata["started_at"] = datetime.now().isoformat()

        self._load_dataset()
        self._init_checker()
        self._execute_checks()

        self._run_metadata["completed_at"] = datetime.now().isoformat()
        self._run_metadata["duration_seconds"] = round(time.time() - start_time, 1)

        report = self._build_report()
        self._export_report(report)
        self._print_summary()

        return report

    def _load_dataset(self) -> None:
        """Carica e valida il dataset di test."""
        path = Path(self._testset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset non trovato: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._samples = data["samples"]
        self._run_metadata["dataset_name"] = data.get("dataset_name", path.stem)
        self._run_metadata["total_samples"] = len(self._samples)

        logger.info(
            "Dataset caricato: '%s' — %d sample",
            self._run_metadata["dataset_name"],
            len(self._samples),
        )

    def _init_checker(self) -> None:
        """Istanzia il GuardrailsChecker di produzione."""
        settings = load_settings()
        self._checker = build_guardrails_checker(
            llm_config=settings.llm,
            guardrails_config=settings.guardrails,
            enable_pii=True,
            enable_topical=True,
            enable_injection=True,
            enable_toxicity=True,
            enable_hallucination=True,
            enable_code_guard=True,
            enable_meta=True,
        )

        if self._checker is None:
            raise RuntimeError(
                "Impossibile istanziare il GuardrailsChecker. "
                "Verifica che GROQ_GUARDRAILS_API_KEY sia configurata nel .env."
            )

        logger.info("GuardrailsChecker istanziato (produzione).")

    def _execute_checks(self) -> None:
        """Itera sui sample ed esegue il check appropriato per ciascuno."""
        self._metrics = {
            "input": CheckMetrics(check_type="input"),
            "meta": CheckMetrics(check_type="meta"),
            "output": CheckMetrics(check_type="output"),
        }

        total = len(self._samples)

        for i, sample in enumerate(self._samples):
            check_type = sample["check_type"]
            text = sample["text"]
            expected = sample["expected"]
            category = sample.get("category", "")

            logger.info("[%d/%d] %s | %s | '%s'", i + 1, total, check_type, category, text[:60])

            t0 = time.time()
            predicted = self._run_single_check(check_type, text)
            latency_ms = (time.time() - t0) * 1000
            self._latencies.append(latency_ms)

            metrics = self._metrics[check_type]
            metrics.total += 1

            is_correct = (predicted == expected)
            if is_correct:
                metrics.correct += 1

            # Confusion matrix: "positivo" = la classe da rilevare
            #   input:  positivo = blocked
            #   output: positivo = blocked
            #   meta:   positivo = meta
            positive_label = "blocked" if check_type in ("input", "output") else "meta"
            negative_label = "allowed" if check_type in ("input", "output") else "not_meta"

            if predicted == positive_label and expected == positive_label:
                metrics.tp += 1
            elif predicted == positive_label and expected == negative_label:
                metrics.fp += 1
            elif predicted == negative_label and expected == negative_label:
                metrics.tn += 1
            elif predicted == negative_label and expected == positive_label:
                metrics.fn += 1

            if not is_correct:
                metrics.errors.append({
                    "text": text,
                    "category": category,
                    "expected": expected,
                    "predicted": predicted,
                })
                logger.warning(
                    "  ERRORE: atteso=%s, predetto=%s | '%s'",
                    expected, predicted, text[:60],
                )

            # Pausa per rate limit Groq (piano Free)
            if i < total - 1:
                time.sleep(4.0)

    def _run_single_check(self, check_type: str, text: str) -> str:
        """Esegue un singolo check e restituisce la decisione normalizzata."""
        if check_type == "input":
            allowed, _ = self._checker.check_input(text)
            return "allowed" if allowed else "blocked"

        elif check_type == "output":
            allowed, _ = self._checker.check_output(text)
            return "allowed" if allowed else "blocked"

        elif check_type == "meta":
            is_meta = self._checker.check_meta(text)
            return "meta" if is_meta else "not_meta"

        else:
            logger.error("check_type sconosciuto: %s", check_type)
            return "error"

    def _build_report(self) -> Dict[str, Any]:
        """Costruisce il report completo."""
        avg_latency = (
            sum(self._latencies) / len(self._latencies)
            if self._latencies else 0.0
        )

        return {
            "metadata": {
                "run": self._run_metadata,
                "guardrails_model": load_settings().llm.guardrails_model,
                "avg_latency_ms": round(avg_latency, 1),
            },
            "results": {
                name: m.to_dict() for name, m in self._metrics.items()
            },
        }

    def _export_report(self, report: Dict[str, Any]) -> None:
        """Esporta il report in JSON."""
        output_dir = Path("results/evaluation")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"guardrails_eval_{timestamp}.json"

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info("Report esportato: %s", filepath)

    def _print_summary(self) -> None:
        """Stampa il riepilogo dell'evaluation su console."""
        lines = []
        lines.append("=" * 60)
        lines.append("  RIEPILOGO EVALUATION GUARDRAILS")
        lines.append("=" * 60)
        lines.append(f"  Dataset:  {self._run_metadata.get('dataset_name', 'N/D')}")
        lines.append(f"  Samples:  {self._run_metadata.get('total_samples', 0)}")
        lines.append(f"  Durata:   {self._run_metadata.get('duration_seconds', 0)}s")

        avg_latency = (
            sum(self._latencies) / len(self._latencies)
            if self._latencies else 0.0
        )
        lines.append(f"  Latenza media per check: {avg_latency:.0f}ms")
        lines.append("")

        for name, m in self._metrics.items():
            lines.append(f"  [{name.upper()} CHECK]")
            lines.append(f"    Accuracy:   {m.accuracy:.4f}  ({m.correct}/{m.total})")
            lines.append(f"    Precision:  {m.precision:.4f}")
            lines.append(f"    Recall:     {m.recall:.4f}")
            lines.append(f"    F1-Score:   {m.f1:.4f}")
            lines.append(f"    TP={m.tp}  FP={m.fp}  TN={m.tn}  FN={m.fn}")
            if m.errors:
                lines.append(f"    Errori ({len(m.errors)}):")
                for err in m.errors:
                    lines.append(
                        f"      - [{err['category']}] atteso={err['expected']}, "
                        f"predetto={err['predicted']}: \"{err['text'][:50]}...\""
                    )
            lines.append("")

        lines.append("=" * 60)

        summary = "\n".join(lines)
        print(summary)
        logger.info("\n%s", summary)