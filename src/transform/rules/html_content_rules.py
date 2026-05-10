"""
Regole di filtraggio contenuto HTML per la sezione docenti.

Incapsula la logica di filtraggio DOM per:
  - Pubblicazioni anno=0 (Req 4): rimozione nodi con anno < cutoff_year

Design: Strategy Pattern — la regola è iniettata nel crawler e invocata
sul contenuto HTML *prima* del salvataggio su disco.
"""

import logging
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class PublicationsHtmlFilter:
    """
    Filtra il DOM delle pagine pubblicazioni anno=0, rimuovendo le entry
    con anno < cutoff_year.

    Struttura DOM attesa:
      Ogni pubblicazione è un <div> contenente:
        - <h4> con <small>ID</small> e titolo
        - <table> con righe <tr>, dove la PRIMA <tr> contiene l'anno
          nella seconda <td> (es. <td>2021</td>)

    Strategia:
      1. Trova tutti i blocchi pubblicazione (div con h4 + small + table).
      2. Per ciascuno, estrai l'anno dalla prima riga della tabella.
      3. Se anno < cutoff_year, rimuovi l'intero blocco dal DOM.
      4. Restituisci l'HTML ripulito.
    """

    def __init__(self, cutoff_year: int = 2020):
        self.cutoff_year = cutoff_year

    def filter_html(self, html_content: str) -> str:
        """
        Filtra il contenuto HTML rimuovendo le pubblicazioni
        con anno < cutoff_year.

        Args:
            html_content: HTML grezzo della pagina pubblicazioni anno=0.

        Returns:
            HTML filtrato con solo le pubblicazioni >= cutoff_year.
        """
        soup = BeautifulSoup(html_content, "html.parser")
        removed_count = 0

        for pub_div in soup.find_all("div"):
            h4 = pub_div.find("h4", recursive=False)
            if not h4:
                continue
            small_tag = h4.find("small")
            if not small_tag:
                continue

            table = pub_div.find("table")
            if not table:
                continue

            first_row = table.find("tr")
            if not first_row:
                continue

            tds = first_row.find_all("td")
            if len(tds) < 2:
                continue

            year_text = tds[1].get_text(strip=True)
            try:
                year = int(year_text)
            except ValueError:
                continue

            if year < self.cutoff_year:
                pub_div.decompose()
                removed_count += 1

        if removed_count > 0:
            logger.debug(
                f"  Pubblicazioni filtrate: {removed_count} entry "
                f"con anno < {self.cutoff_year} rimosse dal DOM"
            )

        return str(soup)