import concurrent.futures
from pathlib import Path
from typing import Tuple, List
from tqdm import tqdm
from transform.core.base_rule import CleaningRule

class HTMLCleaner:
    def __init__(self, directory: str, rules: List[CleaningRule], report_filename: str = "resoconto_eliminazioni.txt"):
        self.directory = Path(directory)
        self.report_filename = self.directory / report_filename
        self.rules = rules # Le regole vengono iniettate dall'esterno

    def _evaluate_file(self, filepath: Path) -> Tuple[bool, str, Path]:
        # Logica immutata: controlla prima file name, poi content
        for rule in self.rules:
            if not rule.requires_content:
                if rule.should_delete(filepath):
                    return True, rule.name, filepath

        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception:
            return False, "", filepath

        for rule in self.rules:
            if rule.requires_content:
                if rule.should_delete(filepath, content):
                    return True, rule.name, filepath

        return False, "", filepath

    def run(self):
        if not self.directory.exists() or not self.directory.is_dir():
            print(f"Errore: La directory '{self.directory}' non esiste.")
            return

        html_files = list(self.directory.glob("*.html"))
        total_files = len(html_files)
        
        if total_files == 0:
            print("Nessun file .html trovato.")
            return

        print(f"Inizio scansione di {total_files} file...")
        deleted_records = []

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(self._evaluate_file, filepath): filepath for filepath in html_files}
            
            with tqdm(total=total_files, desc="Analisi in corso", unit="file") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    pbar.update(1)
                    should_delete, reason, filepath = future.result()
                    if should_delete:
                        try:
                            filepath.unlink() 
                            deleted_records.append(f"{filepath.name} | Motivo: {reason}")
                        except Exception as e:
                            print(f"\nImpossibile eliminare {filepath.name}: {e}")

        self._write_report(deleted_records, total_files)

    def _write_report(self, deleted_records: List[str], total_files: int):
        with open(self.report_filename, 'w', encoding='utf-8') as report:
            report.write(f"RESOCONTO PULIZIA\nFile analizzati: {total_files}\nFile eliminati: {len(deleted_records)}\n")
            report.write("=" * 60 + "\n")
            for record in deleted_records:
                report.write(record + "\n")
        print(f"\nPulizia completata! Eliminati: {len(deleted_records)}")