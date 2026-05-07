import re
from pathlib import Path
from typing import Optional
from transform.core.base_rule import CleaningRule

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
        return "Filtro Lingua URL/Filename (-en-, -zh-)"
    
    @property
    def requires_content(self) -> bool:
        return False
        
    def should_delete(self, filepath: Path, content: Optional[str] = None) -> bool:
        # Nel crawler inline 'filepath.name' è una versione sanitizzata dell'URL.
        # I path come '/en/' o '/zh/' diventano magicamente '-en-' e '-zh-'.
        targets = ["-en-", "-zh-", "-zh", "-en.", "_en_", "_zh_"]
        return any(target in filepath.name.lower() for target in targets)

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
        self.directory = directory
        self.id_pattern = re.compile(r'id=(\d+)')
        
        # NESSUNA PRE-SCANSIONE. L'inizializzazione è istantanea.
    
    @property
    def name(self) -> str:
        return "Filtro Didattica (inline con pulizia retroattiva)"

    @property
    def requires_content(self) -> bool:
        return False

    def should_delete(self, filepath: Path, content: Optional[str] = None) -> bool:
        filename = filepath.name
        
        # 1. Agisce SOLO sulle URL/file della didattica
        if "-didattica-" not in filename:
            return False
            
        has_id = "id=" in filename
        has_cid = "cId=" in filename
        has_pid = "pId=" in filename
        
        # 2. Logica base: Scarta id+cId senza pId
        # (Questo scarto inline è sicuro perché non contiene link essenziali)
        if has_id and has_cid and not has_pid:
            return True
            
        # 3. Logica relazionale: Trovata la versione COMPLETA
        if has_id and has_cid and has_pid:
            match = self.id_pattern.search(filename)
            if match:
                file_id = match.group(1)
                # PULIZIA RETROATTIVA: Cerchiamo sul disco il file "solo id"
                # salvato in precedenza (che ha permesso al crawler di arrivare qui)
                # e lo eliminiamo fisicamente.
                if self.directory.exists():
                    for saved_file in self.directory.glob(f"*-id={file_id}-*.html"):
                        saved_name = saved_file.name
                        # Ci assicuriamo di cancellare solo la versione incompleta
                        if "cId=" not in saved_name and "pId=" not in saved_name:
                            try:
                                saved_file.unlink()
                            except OSError:
                                pass # Ignora errori se il file è bloccato o già rimosso
            
            # Conserviamo questa versione completa
            return False
            
        # 4. Caso file "solo id" (padre)
        # Lo facciamo passare (return False) così il crawler può estrarre i link ai figli.
        # Verrà distrutto retroattivamente allo step 3 non appena verrà analizzato un figlio.
        return False

class ExactPublicationsBaseRule(CleaningRule):
    @property
    def name(self) -> str:
        return "Pagina base pubblicazioni (senza parametri aggiuntivi)"

    @property
    def requires_content(self) -> bool:
        # Controllo rapidissimo solo sul nome
        return False

    def should_delete(self, filepath: Path, content: Optional[str] = None) -> bool:
        # endswith() garantisce che colpiamo SOLO la radice esatta,
        # salvando i file come "...ricerca-pubblicazioni-anno=2020.html"
        return filepath.name.endswith("ricerca-pubblicazioni.html")


class DepartmentBandiRule(CleaningRule):
    def __init__(self, target_department: str = "300638"):
        self.target_department = target_department
        # Intercetta sia "struttura=" che "cdsStruttura=" catturandone il valore
        self.struttura_pattern = re.compile(r'(?:cdsS|s)truttura=([^&.\s]+)')

    @property
    def name(self) -> str:
        return f"Filtro Bandi (struttura assente, =more, o diversa da {self.target_department})"

    @property
    def requires_content(self) -> bool:
        # Altamente efficiente: non apriamo il file
        return False

    def should_delete(self, filepath: Path, content: Optional[str] = None) -> bool:
        filename = filepath.name
        
        # 0. Agisce SOLO sui file che appartengono alla sezione bandi
        if "-home-bandi-" not in filename:
            return False
            
        # 1. Elimina i nomi che hanno "struttura=more" o "cdsStruttura=more"
        if "truttura=more" in filename:
            return True
            
        match = self.struttura_pattern.search(filename)
        
        # 2. Elimina i nomi che non hanno "struttura" o "cdsStruttura"
        if not match:
            return True
            
        # 3. Elimina i nomi che hanno una struttura diversa dal codice target
        struttura_val = match.group(1)
        if struttura_val != self.target_department:
            return True
                
        # In tutti gli altri casi (struttura presente e uguale al tuo codice), conserva
        return False

class CalendarRule(CleaningRule):
    @property
    def name(self) -> str:
        return "File relativo a calendari (singolare o plurale)"

    @property
    def requires_content(self) -> bool:
        return False

    def should_delete(self, filepath: Path, content: Optional[str] = None) -> bool:
        # La stringa "calendari" intercetta matematicamente sia "calendari" che "calendario"
        return "calendari" in filepath.name.lower()

class NewsRule(CleaningRule):
    @property
    def name(self) -> str:
        return "File relativo a news/notizie (contiene 'news')"

    @property
    def requires_content(self) -> bool:
        # Controllo rapido sul nome, non apre il file
        return False

    def should_delete(self, filepath: Path, content: Optional[str] = None) -> bool:
        return "news" in filepath.name.lower()