"""Generatore di report per i risultati della valutazione RAGAS.

Produce report in formato console, CSV, Markdown e JSON con i risultati
dettagliati e aggregati della valutazione dell'Agente RAG DIEM.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Formattatore e generatore di report per i risultati RAGAS.

    Produce output in quattro formati:
    - Console summary: stampa riassuntiva con le metriche aggregate
    - CSV dettagliato: una riga per ogni domanda con tutti gli score
    - Markdown report: report leggibile con tabella dei risultati
    - JSON summary: riepilogo strutturato per confronto tra run diverse
    """

    def __init__(self, output_dir: str = "evaluation/results"):
        """Inizializza il generatore di report.

        Args:
            output_dir: Directory di output per i file dei report.
        """
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info("ReportGenerator inizializzato: output_dir='%s'", output_dir)

    def generate_console_summary(self, results: Dict[str, Any]) -> None:
        """Stampa un riepilogo dei risultati sulla console.

        Args:
            results: Dizionario con i risultati della valutazione RAGAS.
        """
        print("\n" + "=" * 60)
        print("  RIEPILOGO VALUTAZIONE RAGAS - Agente RAG DIEM")
        print("=" * 60)

        if "error" in results:
            print(f"\n  ERRORE: {results['error']}")
            print("=" * 60)
            return

        num_samples = results.get("num_samples", 0)
        print(f"\n  Campioni valutati: {num_samples}")
        print(f"  Metriche calcolate: {', '.join(results.get('metrics_used', []))}")

        scores = results.get("scores", {})
        if scores:
            print("\n  --- Score Aggregati ---")
            for metric_name, score in scores.items():
                if isinstance(score, (int, float)):
                    bar = self._generate_bar(score)
                    print(f"  {metric_name:<30s} {score:.4f}  {bar}")
                else:
                    print(f"  {metric_name:<30s} {score}")

        print("\n" + "=" * 60)
        print()

        logger.info("Riepilogo console stampato")

    def generate_csv_report(self, df, filename: Optional[str] = None) -> str:
        """Genera un report CSV dettagliato con i risultati per-sample.

        Args:
            df: DataFrame Pandas con i risultati dettagliati.
            filename: Nome del file di output. Se None, viene generato
                      automaticamente con timestamp.

        Returns:
            Percorso del file CSV generato.
        """
        if df is None:
            logger.warning("DataFrame nullo, impossibile generare report CSV")
            return ""

        if filename is None:
            filename = f"ragas_eval_{self._timestamp}.csv"

        filepath = self._output_dir / filename

        try:
            column_mapping = {
                "user_input": "query",
                "retrieved_contexts": "contexts",
                "response": "answer",
                "reference": "ground_truth",
                "context_precision": "ctx_prec",
                "context_recall": "ctx_rec",
                "answer_relevancy": "ans_rel",
                "response_relevancy": "ans_rel",
                "faithfulness": "faith",
                "factual_correctness": "fact_f1",
            }

            df_report = df.copy()
            df_report.rename(
                columns={
                    k: v for k, v in column_mapping.items()
                    if k in df_report.columns
                },
                inplace=True,
            )

            df_report.to_csv(filepath, index=False, encoding="utf-8-sig")

            logger.info("Report CSV generato: %s", filepath)
            print(f"  Report CSV salvato: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error("Errore generazione report CSV: %s", e)
            return ""

    def generate_json_summary(
        self,
        results: Dict[str, Any],
        df=None,
        run_metadata: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
    ) -> str:
        """Genera un file JSON strutturato con il riepilogo della run.

        Pensato per il confronto tra run diverse: ogni file contiene
        timestamp, configurazione, metriche aggregate e score per-sample.
        Puo' essere caricato facilmente con json.load() o pandas per
        costruire tabelle comparative nel report finale.

        Args:
            results: Dizionario con i risultati aggregati della valutazione.
            df: DataFrame Pandas con i risultati dettagliati (opzionale).
                Se fornito, include gli score per-sample nel JSON.
            run_metadata: Dizionario opzionale con metadati aggiuntivi
                sulla run (es. modello usato, parametri retriever, note).
            filename: Nome del file di output. Se None, viene generato
                      automaticamente con timestamp.

        Returns:
            Percorso del file JSON generato.
        """
        if filename is None:
            filename = f"results_summary_{self._timestamp}.json"

        filepath = self._output_dir / filename

        try:
            summary = {
                "run_id": self._timestamp,
                "timestamp": datetime.now().isoformat(),
                "num_samples": results.get("num_samples", 0),
                "metrics_used": results.get("metrics_used", []),
                "aggregate_scores": {},
                "per_sample_scores": [],
            }

            if run_metadata:
                summary["run_metadata"] = run_metadata

            scores = results.get("scores", {})
            for metric_name, score in scores.items():
                if isinstance(score, (int, float)):
                    summary["aggregate_scores"][metric_name] = round(score, 4)
                else:
                    summary["aggregate_scores"][metric_name] = score

            numeric_scores = {
                k: v for k, v in scores.items()
                if isinstance(v, (int, float))
            }
            if numeric_scores:
                summary["average_score"] = round(
                    sum(numeric_scores.values()) / len(numeric_scores), 4
                )

            if "error" in results:
                summary["error"] = results["error"]

            if df is not None and not df.empty:
                metric_cols = [
                    c for c in df.columns
                    if c not in [
                        "user_input", "response", "reference",
                        "retrieved_contexts",
                    ]
                ]

                for idx, row in df.iterrows():
                    sample_entry = {
                        "sample_index": idx,
                        "question": str(row.get("user_input", "")),
                        "scores": {},
                    }
                    for col in metric_cols:
                        val = row.get(col)
                        if isinstance(val, float):
                            sample_entry["scores"][col] = round(val, 4)
                        elif val is not None:
                            sample_entry["scores"][col] = val
                    summary["per_sample_scores"].append(sample_entry)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            logger.info("JSON summary generato: %s", filepath)
            print(f"  JSON summary salvato: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error("Errore generazione JSON summary: %s", e)
            return ""

    def generate_markdown_report(
        self,
        results: Dict[str, Any],
        df=None,
        filename: Optional[str] = None,
    ) -> str:
        """Genera un report Markdown leggibile con tabella dei risultati.

        Args:
            results: Dizionario con i risultati aggregati della valutazione.
            df: DataFrame Pandas con i risultati dettagliati (opzionale).
            filename: Nome del file di output. Se None, viene generato
                      automaticamente con timestamp.

        Returns:
            Percorso del file Markdown generato.
        """
        if filename is None:
            filename = f"ragas_report_{self._timestamp}.md"

        filepath = self._output_dir / filename

        try:
            lines = []

            lines.append("# Report Valutazione RAGAS - Agente RAG DIEM")
            lines.append("")
            lines.append(
                f"**Data**: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
            )
            lines.append(
                f"**Campioni valutati**: {results.get('num_samples', 0)}"
            )
            lines.append("")

            if "error" in results:
                lines.append("## Errore")
                lines.append("")
                lines.append(f"La valutazione ha riscontrato un errore: {results['error']}")
                lines.append("")
                self._write_file(filepath, lines)
                return str(filepath)

            lines.append("## Metriche Aggregate")
            lines.append("")
            lines.append("| Metrica | Score |")
            lines.append("|---------|-------|")

            scores = results.get("scores", {})
            for metric_name, score in scores.items():
                if isinstance(score, (int, float)):
                    lines.append(f"| {metric_name} | {score:.4f} |")
                else:
                    lines.append(f"| {metric_name} | {score} |")

            lines.append("")

            lines.append("## Interpretazione")
            lines.append("")
            self._add_interpretation(lines, scores)
            lines.append("")

            if df is not None and not df.empty:
                lines.append("## Risultati Dettagliati per Domanda")
                lines.append("")

                display_cols = []
                for col in ["user_input", "response", "reference"]:
                    if col in df.columns:
                        display_cols.append(col)

                metric_cols = [
                    c for c in df.columns
                    if c not in [
                        "user_input", "response", "reference",
                        "retrieved_contexts",
                    ]
                ]
                display_cols.extend(metric_cols)

                if display_cols:
                    header = "| # | " + " | ".join(display_cols) + " |"
                    separator = "|---|" + "|".join(
                        ["---" for _ in display_cols]
                    ) + "|"
                    lines.append(header)
                    lines.append(separator)

                    for idx, row in df.iterrows():
                        cells = []
                        for col in display_cols:
                            val = row.get(col, "")
                            if isinstance(val, float):
                                cells.append(f"{val:.4f}")
                            elif isinstance(val, str) and len(val) > 80:
                                cells.append(val[:77] + "...")
                            else:
                                cells.append(str(val).replace("|", "\\|"))
                        line = f"| {idx + 1} | " + " | ".join(cells) + " |"
                        lines.append(line)

                lines.append("")

            lines.append("## Note")
            lines.append("")
            lines.append(
                "- **ContextPrecision**: Misura la qualita' dei documenti recuperati "
                "(penalizza il rumore)."
            )
            lines.append(
                "- **ContextRecall**: Misura la completezza del retrieval "
                "(informazioni necessarie recuperate)."
            )
            lines.append(
                "- **Faithfulness**: Controllo anti-allucinazione "
                "(risposta basata sui documenti)."
            )
            lines.append(
                "- **ResponseRelevancy**: Pertinenza della risposta alla domanda."
            )
            lines.append(
                "- **FactualCorrectness**: Confronto con la ground truth (F1)."
            )
            lines.append("")

            self._write_file(filepath, lines)

            logger.info("Report Markdown generato: %s", filepath)
            print(f"  Report Markdown salvato: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error("Errore generazione report Markdown: %s", e)
            return ""

    @staticmethod
    def _generate_bar(score: float, width: int = 20) -> str:
        """Genera una barra di progresso testuale per uno score.

        Args:
            score: Valore dello score tra 0.0 e 1.0.
            width: Larghezza totale della barra in caratteri.

        Returns:
            Stringa con la barra di progresso.
        """
        filled = int(score * width)
        empty = width - filled
        return f"[{'█' * filled}{'░' * empty}]"

    @staticmethod
    def _add_interpretation(lines: list, scores: Dict[str, Any]) -> None:
        """Aggiunge un paragrafo di interpretazione dei risultati.

        Args:
            lines: Lista di righe del report a cui aggiungere l'interpretazione.
            scores: Dizionario con gli score aggregati.
        """
        if not scores:
            lines.append("Nessuno score disponibile per l'interpretazione.")
            return

        numeric_scores = {
            k: v for k, v in scores.items() if isinstance(v, (int, float))
        }

        if not numeric_scores:
            lines.append("Nessuno score numerico disponibile.")
            return

        avg_score = sum(numeric_scores.values()) / len(numeric_scores)
        lines.append(
            f"**Score medio complessivo**: {avg_score:.4f}"
        )
        lines.append("")

        best_metric = max(numeric_scores, key=numeric_scores.get)
        worst_metric = min(numeric_scores, key=numeric_scores.get)

        lines.append(
            f"- **Punto di forza**: {best_metric} ({numeric_scores[best_metric]:.4f})"
        )
        lines.append(
            f"- **Area di miglioramento**: {worst_metric} ({numeric_scores[worst_metric]:.4f})"
        )

        lines.append("")
        lines.append("### Suggerimenti")
        lines.append("")

        ctx_prec = numeric_scores.get("context_precision", None)
        ctx_rec = numeric_scores.get("context_recall", None)
        faith = numeric_scores.get("faithfulness", None)
        rel = numeric_scores.get("response_relevancy", None)
        fact = numeric_scores.get("factual_correctness", None)

        if ctx_prec is not None and ctx_prec < 0.7:
            lines.append(
                "- **Context Precision bassa**: i documenti recuperati contengono "
                "troppo rumore. Considerare: chunking piu' fine, filtri metadata "
                "piu' restrittivi, soglia reranker piu' alta."
            )

        if ctx_rec is not None and ctx_rec < 0.7:
            lines.append(
                "- **Context Recall basso**: il sistema non recupera tutte le "
                "informazioni necessarie. Considerare: multi-query expansion, "
                "agentic retrieval, embedding model piu' potente."
            )

        if faith is not None and faith < 0.7:
            lines.append(
                "- **Faithfulness bassa**: il modello genera informazioni non "
                "supportate dai documenti (allucinazioni). Considerare: "
                "system prompt piu' restrittivo, temperatura LLM piu' bassa."
            )

        if rel is not None and rel < 0.7:
            lines.append(
                "- **Response Relevancy bassa**: le risposte non sono "
                "sufficientemente pertinenti alla domanda. Considerare: "
                "miglioramento del system prompt, query rewriting."
            )

        if fact is not None and fact < 0.7:
            lines.append(
                "- **Factual Correctness bassa**: le risposte non corrispondono "
                "alla ground truth. Verificare: qualita' del retrieval, "
                "copertura del knowledge base, accuratezza delle ground truth."
            )

        if all(
            v >= 0.7
            for v in numeric_scores.values()
        ):
            lines.append(
                "- Tutti gli score sono sopra la soglia di 0.7. "
                "Il sistema mostra buone performance complessive."
            )

    @staticmethod
    def _write_file(filepath: Path, lines: list) -> None:
        """Scrive le righe in un file.

        Args:
            filepath: Percorso del file da scrivere.
            lines: Lista di righe da scrivere.
        """
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))