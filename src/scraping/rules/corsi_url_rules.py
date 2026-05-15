"""
Regole di filtraggio URL per la sezione corsi.unisa.it.
"""

class CorsiUrlClassifier:
    """
    Classificatore URL per il dominio corsi.unisa.it.
    
    Logica:
      - "navigate": identifica pagine 'piano-di-studi' o 'regolamenti'.
                    Verranno esplorate per estrarre PDF e link, ma l'HTML NON verrÃ  salvato.
      - "pass": l'URL non corrisponde a questi pattern, procedi normalmente.
    """
    def classify(self, url: str) -> str:
        url_lower = url.lower()
        
        # Applichiamo la logica solo se siamo su corsi.unisa.it
        if "corsi.unisa.it" in url_lower:
            # Se contiene "piano-di-studi" oppure "regolamenti" -> NAVIGATE
            if "piano-di-studi" in url_lower or "regolamenti" in url_lower:
                return "navigate"
                
        return "pass"