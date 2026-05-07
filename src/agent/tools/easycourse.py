"""
Tool per l'agente: accesso real-time agli orari dei corsi da EasyCourse.

Perché tool e non indicizzazione statica:
  Gli orari cambiano settimanalmente. Indicizzarli nel Vector Store produrrebbe
  chunk con informazioni obsolete nel giro di giorni. Esponendo EasyCourse come 
  tool dell'agente, l'LLM lo chiama a runtime e riceve dati sempre freschi.

Pattern: Tool (Agentic RAG) — il retrieval degli orari è un'azione dell'agente,
         non una ricerca nel Vector Store.

KPI Impact: 
- Correctness: orari sempre aggiornati, nessun dato stale.
- Scope: copre il requisito obbligatorio di integrazione easycourse.unisa.it.

NOTE TECNICHE:
  EasyCourse espone un'interfaccia web dinamica. La strategia è:
  1. Chiamare le API JSON sottostanti (se disponibili), oppure
  2. Fare scraping della pagina renderizzata.
  
  L'implementazione sotto usa l'approccio API-first con fallback a scraping.
  I parametri dei corsi DIEM (faculty_id, course_id) vanno mappati manualmente
  o estratti da una prima chiamata esplorativa.
"""

import logging
import re
from typing import Optional, List, Dict
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class ScheduleEntry:
    """Singola voce dell'orario lezioni."""
    course_name: str
    professor: str
    day: str
    start_time: str
    end_time: str
    room: str
    building: str = ""
    
    def to_text(self) -> str:
        location = f"{self.room}" + (f" ({self.building})" if self.building else "")
        return (
            f"{self.course_name} — Prof. {self.professor}\n"
            f"  {self.day} {self.start_time}-{self.end_time}, Aula: {location}"
        )


class EasyCourseClient:
    """
    Client per interrogare EasyCourse UniSA.
    
    EasyCourse ha una struttura:
      /AgendaStudenti → Facoltà → Corso di Laurea → Anno → Settimana
    
    L'approccio è API-first: tenta di usare gli endpoint JSON interni.
    Se non disponibili, fallback a scraping HTML.
    """
    
    BASE_URL = "https://easycourse.unisa.it"
    
    # Mapping statico dei corsi DIEM (da mantenere allineato)
    # Questi ID vanno verificati/aggiornati a inizio anno accademico
    DIEM_COURSES = {
        "ingegneria_informatica_triennale": {
            "label": "Ingegneria Informatica (L-8)",
            "faculty_id": None,  # Da popolare con scraping iniziale
            "course_id": None,
        },
        "ingegneria_informatica_magistrale": {
            "label": "Ingegneria Informatica (LM-32)",
            "faculty_id": None,
            "course_id": None,
        },
        "medicina_digitale": {
            "label": "Ing. dell'Informazione per la Medicina Digitale (L-8)",
            "faculty_id": None,
            "course_id": None,
        },
        "electrical_engineering": {
            "label": "Electrical Engineering for Digital Energy (LM-28)",
            "faculty_id": None,
            "course_id": None,
        },
    }
    
    def __init__(self, timeout: int = 30):
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "DIEM-RAG-Bot/1.0 (Università di Salerno)"
        })
    
    def search_schedule(
        self,
        course_name: Optional[str] = None,
        professor_name: Optional[str] = None,
    ) -> str:
        """
        Cerca orari per nome corso o docente.
        
        Strategia: scraping della pagina EasyCourse con ricerca testuale.
        Restituisce un blocco di testo formattato con gli orari trovati.
        
        Args:
            course_name: Nome (parziale) del corso/insegnamento.
            professor_name: Cognome del docente.
            
        Returns:
            Stringa formattata con gli orari trovati, oppure messaggio di errore.
        """
        if not course_name and not professor_name:
            return (
                "Per cercare gli orari serve almeno il nome del corso "
                "o il cognome del docente."
            )
        
        try:
            # Tenta l'approccio API JSON
            results = self._search_via_api(course_name, professor_name)
            
            if results is None:
                # Fallback: scraping diretto
                results = self._search_via_scraping(course_name, professor_name)
            
            if not results:
                search_term = course_name or professor_name
                return (
                    f"Non ho trovato orari per '{search_term}' su EasyCourse. "
                    f"Verifica il nome esatto su {self.BASE_URL} oppure "
                    f"contatta la segreteria del DIEM."
                )
            
            header = "Orari trovati su EasyCourse:\n\n"
            body = "\n\n".join(entry.to_text() for entry in results)
            footer = f"\n\nFonte: {self.BASE_URL}"
            
            return header + body + footer
            
        except Exception as e:
            logger.error(f"Errore interrogazione EasyCourse: {e}")
            return (
                "Non sono riuscito a recuperare gli orari da EasyCourse al momento. "
                f"Puoi consultarli direttamente su {self.BASE_URL}"
            )
    
    def _search_via_api(
        self,
        course_name: Optional[str],
        professor_name: Optional[str],
    ) -> Optional[List[ScheduleEntry]]:
        """
        Tenta di usare gli endpoint JSON interni di EasyCourse.
        Restituisce None se l'approccio API non è disponibile.
        """
        # EasyCourse non ha API pubbliche documentate.
        # Questo metodo è un placeholder per quando si identificano
        # gli endpoint XHR dalla network inspection del browser.
        # Per ora restituisce None per attivare il fallback.
        return None
    
    def _search_via_scraping(
        self,
        course_name: Optional[str],
        professor_name: Optional[str],
    ) -> List[ScheduleEntry]:
        """
        Fallback: scraping della pagina HTML di EasyCourse.
        
        Naviga alla pagina dell'agenda studenti e cerca le informazioni
        corrispondenti alla query.
        """
        results: List[ScheduleEntry] = []
        search_term = (course_name or professor_name or "").lower()
        
        try:
            # Pagina principale dell'agenda
            response = self._session.get(
                f"{self.BASE_URL}/AgendaStudenti",
                timeout=self._timeout,
            )
            
            if not response.ok:
                logger.warning(f"EasyCourse non raggiungibile: {response.status_code}")
                return results
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Cerca nelle tabelle orario (la struttura esatta dipende 
            # dalla versione corrente di EasyCourse)
            schedule_rows = soup.find_all("tr")
            
            for row in schedule_rows:
                cells = row.find_all("td")
                if len(cells) < 4:
                    continue
                
                row_text = row.get_text(separator=" ", strip=True).lower()
                
                if search_term in row_text:
                    # Parsing adattivo delle celle
                    entry = self._parse_schedule_row(cells)
                    if entry:
                        results.append(entry)
            
        except requests.RequestException as e:
            logger.error(f"Errore connessione EasyCourse: {e}")
        
        return results
    
    def _parse_schedule_row(self, cells) -> Optional[ScheduleEntry]:
        """Parsing di una riga della tabella orario. Adattabile alla struttura HTML."""
        try:
            texts = [cell.get_text(strip=True) for cell in cells]
            
            # La struttura tipica è: Giorno | Ora inizio | Ora fine | Corso | Docente | Aula
            # Ma può variare — questo è un parser best-effort
            if len(texts) >= 6:
                return ScheduleEntry(
                    day=texts[0],
                    start_time=texts[1],
                    end_time=texts[2],
                    course_name=texts[3],
                    professor=texts[4],
                    room=texts[5],
                    building=texts[6] if len(texts) > 6 else "",
                )
            elif len(texts) >= 4:
                return ScheduleEntry(
                    course_name=texts[0],
                    professor=texts[1] if len(texts) > 1 else "N/D",
                    day=texts[2] if len(texts) > 2 else "N/D",
                    start_time=texts[3] if len(texts) > 3 else "N/D",
                    end_time="",
                    room=texts[4] if len(texts) > 4 else "N/D",
                )
        except (IndexError, AttributeError):
            pass
        return None


# Istanza globale per uso nel tool dell'agente
_client = None

def get_easycourse_client() -> EasyCourseClient:
    """Singleton lazy per il client EasyCourse."""
    global _client
    if _client is None:
        _client = EasyCourseClient()
    return _client
