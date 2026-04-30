import re
from bs4 import BeautifulSoup
from langchain_community.document_loaders import RecursiveUrlLoader

# Set globale per memorizzare i link ai PDF trovati durante lo scraping HTML.
# Verranno passati successivamente al PyPDFLoader (Modulo 1.2).
found_pdf_links = set()

def custom_unisa_extractor(html_content: str) -> str:
    """
    Estrattore personalizzato per pulire le pagine UNISA.
    Mantiene la struttura HTML utile per il successivo HTMLSectionSplitter.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    
    # 1. Pulizia chirurgica: Rimuoviamo elementi strutturali tossici per il RAG
    noise_selectors = [
        "#header",                # Carosello e header
        "#main-menu",             # Navigazione principale
        "#menu-bar",              # Barra menu
        "#unisa-left-menu",       # Menu laterale sinistro (include i social)
        "#box-agenda",            # Calendario eventi
        "#share-dropdown",        # Pulsanti condivisione
        ".breadcrumb",            # Briciole di pane
        "footer",                 # Piè di pagina principale
        ".sub-footer",            # Informazioni legali
        ".sr-only",               # Testi nascosti per screen reader (spesso duplicati)
        "div[id$='-map']",        # Div dinamici delle mappe di ateneo (es. 026557-map)
        "script", 
        "style", 
        "noscript"
    ]
    
    for selector in noise_selectors:
        for element in soup.select(selector):
            element.decompose()
        
    # 2. Intercettazione PDF: Troviamo tutti i link che puntano a documenti PDF
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        if href.lower().endswith('.pdf'):
            # In un'app reale, qui si gestiscono i link relativi convertendoli in assoluti
            found_pdf_links.add(href)
            
    # 3. Estrazione Contenuto Utile
    # Il div con id "unisa-content" è il contenitore principale del testo
    main_content = soup.find(id="unisa-content")
    
    if main_content:
        # Pulizia degli attributi ma preserviamo quelli utili alle tabelle e semantica
        allowed_attrs = ['href', 'src', 'colspan', 'rowspan']
        for tag in main_content.find_all(True):
            tag.attrs = {key: value for key, value in tag.attrs.items() if key in allowed_attrs}
            
        return main_content.decode_contents().strip()
    else:
        print(f"[Warning] Pagina senza #unisa-content\n\tURL: {soup.find('base')['href'] if soup.find('base') else 'URL sconosciuto'}")
        return ""  # Ritorna stringa vuota se non troviamo il contenuto principale

def build_diem_loaders():
    """
    Configura e restituisce i loader per i domini autorizzati.
    """
    loaders = []
    
    # --- 1. Dominio DIEM principale ---
    # prevent_outside=True assicura di non uscire da diem.unisa.it
    diem_loader = RecursiveUrlLoader(
        url="https://www.diem.unisa.it/",
        max_depth=4,  # Regolare in base alla profondità del sito
        prevent_outside=True, 
        extractor=custom_unisa_extractor,
        check_response_status=True
    )
    loaders.append(diem_loader)

    # --- 2. Dominio Corsi DIEM ---
    # ATTENZIONE: per non scaricare corsi di altri dipartimenti (fuori scope),
    # dobbiamo definire esplicitamente le base_url dei corsi DIEM.
    diem_courses_urls = [
        "https://corsi.unisa.it/ingegneria-informatica", # triennale
        "https://corsi.unisa.it/ingegneria-dell-informazione-per-la-medicina-digitale" # triennale
        "https://corsi.unisa.it/ingegneria-informatica-magistrale", # magistrale
        "https://corsi.unisa.it/information-Engineering-for-digital-medicine", # magistrale
        "https://corsi.unisa.it/ingegneria-dell-informazione" # PHD
        # Aggiungere gli altri corsi afferenti al DIEM
    ]
    
    for course_url in diem_courses_urls:
        course_loader = RecursiveUrlLoader(
            url=course_url,
            max_depth=3,
            prevent_outside=True, # Limita l'esplorazione strettamente al singolo corso
            extractor=custom_unisa_extractor,
            check_response_status=True
        )
        loaders.append(course_loader)
        
    # --- 3. Dominio Docenti DIEM ---
    # Come per i corsi, https://docenti.unisa.it/ contiene docenti di tutto l'ateneo.
    # Bisogna iniettare una lista di URL dei docenti specifici del DIEM
    diem_faculty_urls = [
        "https://docenti.unisa.it/nicola.capuano",
        "https://docenti.unisa.it/antonio.greco",
        "https://docenti.unisa.it/giovannina.albano",
        "https://docenti.unisa.it/vincenzo.auletta",
        "https://docenti.unisa.it/francesco.basile",
        "https://docenti.unisa.it/pasquale.chiacchio",
        "https://docenti.unisa.it/antonio.dellacioppa",
        "https://docenti.unisa.it/nicola.femia",
        "https://docenti.unisa.it/pasquale.foggia",
        "https://docenti.unisa.it/matteo.gaeta",
        "https://docenti.unisa.it/stefano.marano",
        "https://docenti.unisa.it/angelo.marcelli",
        "https://docenti.unisa.it/vincenzo.matta",
        "https://docenti.unisa.it/ettore.napoli",
        "https://docenti.unisa.it/gennaro.percannella",
        "https://docenti.unisa.it/giovanni.petrone",
        "https://docenti.unisa.it/pierluigi.ritrovato",
        "https://docenti.unisa.it/sabrina.senatore",
        "https://docenti.unisa.it/giovanni.spagnuolo",
        "https://docenti.unisa.it/francesco.tortorella",
        "https://docenti.unisa.it/vincenzo.tucci",
        "https://docenti.unisa.it/mario.vento",
        "https://docenti.unisa.it/paolo.addesso",
        "https://docenti.unisa.it/andrea.apicella",
        "https://docenti.unisa.it/vincenzo.carletti",
        "https://docenti.unisa.it/giuseppe.daniello",
        "https://docenti.unisa.it/tiziana.durante",
        "https://docenti.unisa.it/daniele.esposito",
        "https://docenti.unisa.it/mariarosaria.falanga",
        "https://docenti.unisa.it/diodato.ferraioli",
        "https://docenti.unisa.it/lidia.fotia",
        "https://docenti.unisa.it/diego.gragnaniello",
        "https://docenti.unisa.it/luca.greco",
        "https://docenti.unisa.it/michele.guida",
        "https://docenti.unisa.it/patrizia.lamberti",
        "https://docenti.unisa.it/francesco.moscato",
        "https://docenti.unisa.it/fabio.postiglione",
        "https://docenti.unisa.it/rocco.restaino",
        "https://docenti.unisa.it/giovanni.riccio",
        "https://docenti.unisa.it/leonardo.rundo",
        "https://docenti.unisa.it/045640/home" # Russo Giovanni ha un omonimo
        "https://docenti.unisa.it/alessia.saggese",
        "https://docenti.unisa.it/walter.zamboni",
        "https://docenti.unisa.it/vittorio.zampoli"
    ]
    
    ####
    # • Course and exam timetables of DIEM courses available from https://easycourse.unisa.it/ (optional)
    ####
    
    for faculty_url in diem_faculty_urls:
        faculty_loader = RecursiveUrlLoader(
            url=faculty_url,
            max_depth=2,
            prevent_outside=True,
            extractor=custom_unisa_extractor,
            check_response_status=True
        )
        loaders.append(faculty_loader)
        
    return loaders

def scrape_all_domains():
    """Esegue lo scraping e ritorna i documenti combinati."""
    all_docs = []
    loaders = build_diem_loaders()
    
    for loader in loaders:
        print(f"Avvio scraping per: {loader.url}")
        docs = loader.load()
        all_docs.extend(docs)
        
    print(f"Scraping completato. Documenti HTML raccolti: {len(all_docs)}")
    print(f"Link PDF intercettati (da passare al Modulo 1.2): {len(found_pdf_links)}")
    return all_docs

if __name__ == "__main__":
    # Esempio d'uso:
    docs = scrape_all_domains()