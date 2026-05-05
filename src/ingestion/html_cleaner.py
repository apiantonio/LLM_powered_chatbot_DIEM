import os
import concurrent.futures
from abc import ABC, abstractmethod
from pathlib import Path
from bs4 import BeautifulSoup
from typing import Tuple, List, Optional

# =====================================================================
# 1. DEFINIZIONE DELLE REGOLE (PATTERN STRATEGY PER ESTENDIBILITÀ)
# =====================================================================

class CleaningRule(ABC):
    """Interfaccia base per tutte le regole di pulizia."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Nome o descrizione della regola (usato per il resoconto)."""
        pass

    @property
    def requires_content(self) -> bool:
        """
        Se True, il sistema leggerà l'HTML e lo passerà alla regola.
        Se False, valuterà solo il nome del file (ottimizzazione).
        """
        return True

    @abstractmethod
    def should_delete(self, filepath: Path, content: Optional[str] = None) -> bool:
        """Restituisce True se il file deve essere eliminato."""
        pass


class FilenameRule(CleaningRule):
    @property
    def name(self) -> str:
        return "Nome file da scartare (-en-, -zh-, -zh)"
    
    @property
    def requires_content(self) -> bool:
        return False
        
    def should_delete(self, filepath: Path, content: Optional[str] = None) -> bool:
        targets = ["-en-", "-zh-", "-zh"]
        return any(target in filepath.name for target in targets)


class EmptyBodyRule(CleaningRule):
    @property
    def name(self) -> str:
        return "Tag <body> vuoto o privo di contenuti utili"
        
    def should_delete(self, filepath: Path, content: Optional[str] = None) -> bool:
        # Controllo rapido testuale per ottimizzare i tempi
        clean_content = content.replace(" ", "").replace("\n", "").lower()
        if "<body></body>" in clean_content:
            return True
            
        # Parsing robusto se il controllo rapido fallisce
        soup = BeautifulSoup(content, 'lxml')
        body = soup.find('body')
        
        if not body:
            return False # Se non c'è il body, la struttura è anomala, meglio non rischiare di cancellare
            
        # Controlliamo se c'è testo visibile
        if body.get_text(strip=True):
            return False
            
        # Controlliamo se ci sono tag media che potrebbero indicare contenuto (es. immagini, iframe)
        if body.find(['img', 'iframe', 'video', 'audio']):
            return False
            
        return True


class NoContentInsertedRule(CleaningRule):
    @property
    def name(self) -> str:
        return "Pagina senza contenuti (Placeholder: 'Nessun contenuto inserito')"
        
    def should_delete(self, filepath: Path, content: Optional[str] = None) -> bool:
        # Cerca solo il placeholder di sistema, indipendentemente dal docente
        return "Nessun contenuto inserito" in content


class PageNotFoundRule(CleaningRule):
    @property
    def name(self) -> str:
        return "Pagina Errore 404 (Non trovata)"
        
    def should_delete(self, filepath: Path, content: Optional[str] = None) -> bool:
        return "404 Pagina non Trovata" in content and "Oops!" in content


# =====================================================================
# 2. MOTORE DI ELABORAZIONE E PULIZIA
# =====================================================================

class HTMLCleaner:
    def __init__(self, directory: str, report_filename: str = "resoconto_eliminazioni.txt"):
        self.directory = Path(directory)
        self.report_filename = self.directory / report_filename
        
        # Inizializza la lista delle regole in ordine di priorità
        # (le più veloci da controllare vanno per prime)
        self.rules: List[CleaningRule] = [
            FilenameRule(),
            NoContentInsertedRule(),
            PageNotFoundRule(),
            EmptyBodyRule()
        ]

    def _evaluate_file(self, filepath: Path) -> Tuple[bool, str, Path]:
        """
        Valuta un singolo file contro tutte le regole.
        Restituisce (Da_Eliminare, Nome_Regola_Violata, Percorso).
        """
        # 1. Controlla prima le regole che NON richiedono la lettura del file (I/O optimization)
        for rule in self.rules:
            if not rule.requires_content:
                if rule.should_delete(filepath):
                    return True, rule.name, filepath

        # 2. Se passa i controlli base, leggiamo il contenuto per le altre regole
        try:
            # errors='ignore' evita crash in caso di file corrotti o con encoding strano
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            # In caso di errore di lettura, per sicurezza NON eliminiamo il file
            return False, "", filepath

        # 3. Controlla le regole basate sul contenuto
        for rule in self.rules:
            if rule.requires_content:
                if rule.should_delete(filepath, content):
                    return True, rule.name, filepath

        return False, "", filepath

    def run(self):
        """Esegue la scansione e la pulizia della directory in parallelo."""
        if not self.directory.exists() or not self.directory.is_dir():
            print(f"Errore: La directory '{self.directory}' non esiste.")
            return

        print(f"Scansione dei file in {self.directory} (potrebbe richiedere qualche minuto...)")
        html_files = list(self.directory.glob("*.html"))
        total_files = len(html_files)
        
        if total_files == 0:
            print("Nessun file .html trovato nella directory.")
            return

        deleted_records = []

        # Utilizziamo ProcessPool per sfruttare il multi-core (ottimo per il parsing di 50k file)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Sottomettiamo i job all'executor
            futures = {executor.submit(self._evaluate_file, filepath): filepath for filepath in html_files}
            
            processed = 0
            for future in concurrent.futures.as_completed(futures):
                processed += 1
                should_delete, reason, filepath = future.result()
                
                if should_delete:
                    try:
                        filepath.unlink() # Eliminazione diretta del file
                        deleted_records.append(f"{filepath.name} | Motivo: {reason}")
                    except Exception as e:
                        print(f"Impossibile eliminare {filepath.name}: {e}")
                
                # Semplice barra di progresso
                if processed % 5000 == 0 or processed == total_files:
                    print(f"Progresso: {processed}/{total_files} file analizzati...")

        # Generazione del file di tracciabilità
        self._write_report(deleted_records, total_files)

    def _write_report(self, deleted_records: List[str], total_files: int):
        with open(self.report_filename, 'w', encoding='utf-8') as report:
            report.write(f"RESOCONTO PULIZIA DIRECTORY\n")
            report.write(f"File totali analizzati: {total_files}\n")
            report.write(f"File eliminati: {len(deleted_records)}\n")
            report.write("=" * 60 + "\n")
            for record in deleted_records:
                report.write(record + "\n")
                
        print(f"\nPulizia completata con successo!")
        print(f"Totale eliminati: {len(deleted_records)}")
        print(f"Resoconto salvato in: {self.report_filename}")


if __name__ == "__main__":
    # INSERISCI QUI IL PERCORSO DELLA TUA DIRECTORY. 
    # Usa "." per la directory in cui ti trovi attualmente.
    DIRECTORY_PATH = "./data/raw/html_samples_v7" 
    
    cleaner = HTMLCleaner(directory=DIRECTORY_PATH)
    cleaner.run()