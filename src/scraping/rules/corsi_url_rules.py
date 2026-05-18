class CorsiUrlClassifier:

    def classify(self, url: str) -> str:
        url_lower = url.lower()
        
        if "corsi.unisa.it" in url_lower:
            if "piano-di-studi" in url_lower or "regolamenti" in url_lower:
                return "navigate"
                
        return "pass"