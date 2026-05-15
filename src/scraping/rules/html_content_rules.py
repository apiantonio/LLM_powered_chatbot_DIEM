import logging
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class PublicationsHtmlFilter:
    """
    Filtra il DOM delle pagine pubblicazioni anno=0, rimuovendo le entry
    con anno < cutoff_year.
    """

    def __init__(self, cutoff_year: int = 2020):
        self.cutoff_year = cutoff_year

    def filter_html(self, html_content: str) -> str:
        soup = BeautifulSoup(html_content, "html.parser")
        removed_count = 0

        # MODIFICA 1: Miriamo direttamente ai div che contengono le singole pubblicazioni
        # per evitare di processare div inutili (header, footer, ecc.)
        for pub_div in soup.find_all("div", class_="panel-primary"):
            
            # MODIFICA 2: Rimosso 'recursive=False'. L'h4 si trova dentro 
            # <div class="panel-heading">, quindi è un nipote, non un figlio diretto.
            h4 = pub_div.find("h4")
            if not h4:
                continue
            
            small_tag = h4.find("small")
            if not small_tag:
                continue

            # La tabella si trova dentro <div class="panel-collapse">, 
            # che è contenuto in pub_div. La troverà senza problemi.
            table = pub_div.find("table")
            if not table:
                continue

            first_row = table.find("tr")
            if not first_row:
                continue

            tds = first_row.find_all("td")
            if len(tds) < 2:
                continue

            # L'anno è nella seconda cella. strip=True previene errori se ci sono spazi/newline.
            year_text = tds[1].get_text(strip=True)
            try:
                year = int(year_text)
            except ValueError:
                continue

            if year < self.cutoff_year:
                # pub_div è l'intero blocco class="panel-primary", quindi eliminiamo tutto!
                pub_div.decompose()
                removed_count += 1

        if removed_count > 0:
            logger.debug(
                f"  Pubblicazioni filtrate: {removed_count} entry "
                f"con anno < {self.cutoff_year} rimosse dal DOM"
            )

        return str(soup)