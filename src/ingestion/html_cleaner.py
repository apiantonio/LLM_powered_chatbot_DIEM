import os
import concurrent.futures
from abc import ABC, abstractmethod
from pathlib import Path
from bs4 import BeautifulSoup
from typing import Tuple, List, Optional
import re
from tqdm import tqdm  # <--- Nuova importazione per la barra di avanzamento

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

class ObsoleteUrlRule(CleaningRule):
    def __init__(self, cutoff_year: int = 2020):
        self.cutoff_year = cutoff_year
        # Intercetta il parametro anno= seguito da numeri, ora direttamente dal nome del file
        self.url_year_pattern = re.compile(r'anno=(\d+)')

    @property
    def name(self) -> str:
        return f"Nome file con anno obsoleto (< {self.cutoff_year}) o non valido (anno=0)"

    @property
    def requires_content(self) -> bool:
        # VANTAGGIO ENORME: impostando a False, lo script NON aprirà il file.
        # Leggerà solo il nome, risparmiando I/O sul disco.
        return False

    def should_delete(self, filepath: Path, content: Optional[str] = None) -> bool:
        # Cerca la regex direttamente in filepath.name (il nome del file)
        match = self.url_year_pattern.search(filepath.name)
        
        if match:
            try:
                # Estrae il valore numerico dell'anno
                year = int(match.group(1))
                
                # Elimina se l'anno è 0 oppure strettamente minore del 2020
                if year == 0 or (1000 < year < self.cutoff_year): 
                    return True
                    
            except ValueError:
                # In caso di errori di conversione, conserviamo il file
                pass
                
        # Se non c'è il parametro 'anno' nel nome, o l'anno è >= 2020, lo teniamo
        return False


class FilenameRule(CleaningRule):
    @property
    def name(self) -> str:
        return "Nome file da scartare (-en-, -zh-, -zh)"
    
    @property
    def requires_content(self) -> bool:
        return False
        
    def should_delete(self, filepath: Path, content: Optional[str] = None) -> bool:
        targets = ["-en-", "-zh-", "-zh", "-en."]
        return any(target in filepath.name for target in targets)


class EmptyBodyRule(CleaningRule):
    @property
    def name(self) -> str:
        return "Tag <body> vuoto o privo di contenuti utili"
        
    def should_delete(self, filepath: Path, content: Optional[str] = None) -> bool:
        clean_content = content.replace(" ", "").replace("\n", "").lower()
        if "<body></body>" in clean_content:
            return True
            
        soup = BeautifulSoup(content, 'lxml')
        body = soup.find('body')
        
        if not body:
            return False 
            
        if body.get_text(strip=True):
            return False
            
        if body.find(['img', 'iframe', 'video', 'audio']):
            return False
            
        return True


class NoContentInsertedRule(CleaningRule):
    @property
    def name(self) -> str:
        return "Pagina senza contenuti (Placeholder: 'Nessun contenuto inserito')"
        
    def should_delete(self, filepath: Path, content: Optional[str] = None) -> bool:
        return "Nessun contenuto inserito" in content


class PageNotFoundRule(CleaningRule):
    @property
    def name(self) -> str:
        return "Pagina Errore 404 (Non trovata)"
        
    def should_delete(self, filepath: Path, content: Optional[str] = None) -> bool:
        return "404 Pagina non Trovata" in content and "Oops!" in content


class PublicationTipRule(CleaningRule):
    def __init__(self):
        # La regex cerca esattamente: 
        # 1. Un trattino seguito da numeri (la matricola, es: -005501)
        # 2. La dicitura fissa "-ricerca-pubblicazioni" (o "pubblicazione")
        # 3. Qualsiasi cosa in mezzo (es. l'anno)
        # 4. L'attributo "tip="
        self.target_pattern = re.compile(r'-\d+-ricerca-pubblicazioni?.*tip=')

    @property
    def name(self) -> str:
        return "Pubblicazioni filtrate per attributo 'tip'"

    @property
    def requires_content(self) -> bool:
        # Estremamente efficiente: non apriamo il file in memoria
        return False

    def should_delete(self, filepath: Path, content: Optional[str] = None) -> bool:
        # Cerca il pattern specifico all'interno del nome del file
        # Se trova il match esatto, elimina. Altrimenti conserva.
        return bool(self.target_pattern.search(filepath.name))

class DidatticaFilterRule(CleaningRule):
    def __init__(self, directory: Path):
        self.complete_ids = set()
        self.id_pattern = re.compile(r'id=(\d+)')
        
        # PRE-CALCOLO: Legge i nomi dei file una sola volta per scovare i duplicati completi
        # Cerca tutti i file che hanno contemporaneamente id, cId e pId
        for filepath in directory.glob("*.html"):
            filename = filepath.name
            if "-didattica-" in filename and "id=" in filename and "cId=" in filename and "pId=" in filename:
                match = self.id_pattern.search(filename)
                if match:
                    # Salva in memoria l'ID che possiede la versione completa
                    self.complete_ids.add(match.group(1))

    @property
    def name(self) -> str:
        return "Filtro Didattica (scartato id+cId senza pId, e file 'solo id' ridondanti)"

    @property
    def requires_content(self) -> bool:
        # Altamente efficiente: non apriamo il file
        return False

    def should_delete(self, filepath: Path, content: Optional[str] = None) -> bool:
        filename = filepath.name
        
        # 1. Agisce SOLO sui file che appartengono alla sezione didattica
        if "-didattica-" not in filename:
            return False
            
        # 2. Mappatura dei parametri presenti nel nome del file
        has_id = "id=" in filename
        has_cid = "cId=" in filename
        has_pid = "pId=" in filename
        
        # 3. Applicazione della PRIMA logica:
        # Se ha 'id' E ha 'cId' MA NON ha 'pId' -> Elimina
        if has_id and has_cid and not has_pid:
            return True
            
        # 4. Applicazione della SECONDA logica (Relazionale):
        # Se il file ha SOLO l'id (nessun cId e nessun pId)
        if has_id and not has_cid and not has_pid:
            match = self.id_pattern.search(filename)
            if match:
                file_id = match.group(1)
                # Verifica in tempo zero se questo ID ha una sua versione completa pre-calcolata
                if file_id in self.complete_ids:
                    return True # Esiste la versione completa id+cId+pId, quindi scartiamo questa
            
        # In tutti gli altri casi, il file viene conservato
        return False

# =====================================================================
# 2. MOTORE DI ELABORAZIONE E PULIZIA
# =====================================================================

class HTMLCleaner:
    def __init__(self, directory: str, report_filename: str = "resoconto_eliminazioni.txt", cutoff_year: int = 2020):
        self.directory = Path(directory)
        self.cutoff_year = cutoff_year
        self.report_filename = self.directory / report_filename
        
        # Inizializza la lista delle regole in ordine di priorità
        self.rules: List[CleaningRule] = [
            FilenameRule(),
            NoContentInsertedRule(),
            ObsoleteUrlRule(cutoff_year=self.cutoff_year),
            DidatticaFilterRule(directory=self.directory),
            PublicationTipRule(),
            PageNotFoundRule(),
            EmptyBodyRule()
        ]

    def _evaluate_file(self, filepath: Path) -> Tuple[bool, str, Path]:
        for rule in self.rules:
            if not rule.requires_content:
                if rule.should_delete(filepath):
                    return True, rule.name, filepath

        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
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
            print("Nessun file .html trovato nella directory.")
            return

        print(f"Inizio scansione di {total_files} file in {self.directory}")
        deleted_records = []

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(self._evaluate_file, filepath): filepath for filepath in html_files}
            
            # Utilizzo di tqdm per la barra di avanzamento in tempo reale
            with tqdm(total=total_files, desc="Analisi in corso", unit="file") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    # Aggiorna la barra di 1 step
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
    DIRECTORY_PATH = "./data/raw/html_samples_v7" 
    
    cleaner = HTMLCleaner(directory=DIRECTORY_PATH, report_filename="eliminazioni_didattica.txt")
    cleaner.run()