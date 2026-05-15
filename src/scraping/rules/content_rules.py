from bs4 import BeautifulSoup
from pathlib import Path
from typing import Optional
from scraping.core.base_rule import CleaningRule

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