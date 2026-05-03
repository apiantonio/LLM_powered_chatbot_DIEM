import re
import os
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from langchain_community.document_loaders import RecursiveUrlLoader

# Set globale per memorizzare i link ai PDF trovati durante lo scraping HTML.
found_pdf_links = set()

def custom_unisa_extractor(html_content: str) -> str:
    """
    Estrattore personalizzato per pulire le pagine UNISA.
    Mantiene la struttura HTML utile per il successivo HTMLSectionSplitter.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    
    # 1. Pulizia chirurgica
    noise_selectors = [
        "#header", "#main-menu", "#menu-bar", "#unisa-left-menu", 
        "#box-agenda", "#share-dropdown", ".breadcrumb", "footer", 
        ".sr-only", "div[id$='-map']", "script", "style", "noscript"
    ]
    
    for selector in noise_selectors:
        for element in soup.select(selector):
            element.decompose()
        
    # 2. Intercettazione PDF
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        if href.lower().endswith('.pdf'):
            found_pdf_links.add(href)
            
    # URL della pagina
    page_url = "URL sconosciuto"
    canonical = soup.find('link', rel='canonical')
    og_url = soup.find('meta', property='og:url')
    
    if canonical and canonical.get('href'):
        page_url = canonical['href']
    elif og_url and og_url.get('content'):
        page_url = og_url['content']
            
    # 3. Catena per trovare il main content
    main_content = soup.find(id="unisa-content")
    if not main_content:
        main_content = soup.find(attrs={"role": "main"})
    if not main_content:
        main_content = soup.find(id="content")
    if not main_content:
        main_content = soup.find("main")
    if not main_content:
        main_content = soup.body
    
    if main_content:
        allowed_attrs = ['href', 'src', 'colspan', 'rowspan']
        for tag in main_content.find_all(True):
            tag.attrs = {key: value for key, value in tag.attrs.items() if key in allowed_attrs}
            
        return main_content.decode_contents().strip()
    else:
        # Silenziamo il warning per mantenere la console pulita
        return "" 

def get_dynamic_faculty_urls():
    """
    Recupera dinamicamente la lista dei docenti del DIEM.
    """
    url_personale = "https://www.diem.unisa.it/dipartimento/personale"
    print(f"Recupero dinamico dei docenti da: {url_personale}...")
    
    dynamic_urls = set()
    try:
        response = requests.get(url_personale, timeout=30)
        response.raise_for_status() 
        soup = BeautifulSoup(response.text, "html.parser")
        
        for a_tag in soup.find_all("a", href=True):
            href = a_tag['href']
            if "rubrica.unisa.it/persone?matricola=" in href:
                match = re.search(r'matricola=(\d+)', href)
                if match:
                    matricola = match.group(1)
                    docenti_url = f"https://docenti.unisa.it/{matricola}/home"
                    dynamic_urls.add(docenti_url)
                    
        print(f"Trovati {len(dynamic_urls)} docenti afferenti al DIEM.")
        return list(dynamic_urls)
    except Exception as e:
        print(f"[Errore] Impossibile recuperare i docenti dinamicamente: {e}")
        return []

def build_diem_loaders():
    """
    Configura e restituisce i loader per i domini autorizzati.
    """
    loaders = []
    safe_link_regex = r"<a\s+(?:[^>]*?\s+)?href=[\"'](?!javascript:|mailto:|tel:|[^\"']*\.(?:css|js|png|jpg|jpeg|gif|pdf|zip)(?:[\?#][^\"']*)?)([^\"']+)[\"']"
    
    # Configurazione base sicura (Timeout a 30 per evitare blocchi infiniti)
    recursive_loader_config = {
        "max_depth": 5,  # 5 è il bilanciamento perfetto per il dipartimento
        "prevent_outside": True, 
        "extractor": custom_unisa_extractor,
        "check_response_status": True,
        "link_regex": safe_link_regex,
        "timeout": 30 
    }
 
    # --- 1. Dominio DIEM principale ---
    diem_loader = RecursiveUrlLoader(
        url="https://www.diem.unisa.it/",
        **recursive_loader_config
    )
    loaders.append(diem_loader)

    # --- 2. Dominio Corsi DIEM ---
    diem_courses_urls = [
        "https://corsi.unisa.it/ingegneria-informatica", 
        "https://corsi.unisa.it/ingegneria-dell-informazione-per-la-medicina-digitale", 
        "https://corsi.unisa.it/ingegneria-informatica-magistrale", 
        "https://corsi.unisa.it/information-Engineering-for-digital-medicine", 
        "https://corsi.unisa.it/ingegneria-dell-informazione" 
    ]
    
    for course_url in diem_courses_urls:
        # Per i corsi riduciamo leggermente la depth a 4 per evitare loop in vecchi avvisi
        course_config = recursive_loader_config.copy()
        course_config["max_depth"] = 4 
        
        course_loader = RecursiveUrlLoader(
            url=course_url,
            **course_config
        )
        loaders.append(course_loader)
        
    # --- 3. Dominio Docenti DIEM ---
    diem_faculty_urls = get_dynamic_faculty_urls()
    
    for faculty_url in diem_faculty_urls:
        faculty_loader = RecursiveUrlLoader(
            url=faculty_url,
            max_depth=2, # I docenti restano a 2, è sufficiente!
            prevent_outside=True,
            extractor=custom_unisa_extractor,
            check_response_status=True,
            link_regex=safe_link_regex,
            timeout=60
        )
        loaders.append(faculty_loader)
        
    return loaders

def scrape_all_domains():
    """Esegue lo scraping e ritorna i documenti combinati."""
    global found_pdf_links
    found_pdf_links.clear() # Svuota la memoria a ogni avvio!
    
    all_docs = []
    loaders = build_diem_loaders()
    
    for loader in loaders:
        print(f"Avvio scraping per: {loader.url}")
        docs = loader.load()
        all_docs.extend(docs)
        
    print(f"\n--- RESOCONTO FINALE ---")
    print(f"Scraping completato. Documenti HTML raccolti: {len(all_docs)}")
    print(f"Link PDF intercettati (da elaborare): {len(found_pdf_links)}")
    return all_docs

def save_scraped_data(docs, sample_size=10, suffix="", output_dir="data/raw/html_samples"):
    """
    Salva un campione di documenti in file .html locali per ispezione.
    """
    os.makedirs(output_dir, exist_ok=True)
    actual_size = len(docs) if sample_size <= 0 or sample_size > len(docs) else sample_size
    print(f"Salvataggio di {actual_size} documenti per ispezione in {output_dir}...")
    
    for i, doc in enumerate(docs[:actual_size]):
        raw_url = doc.metadata.get('source', 'URL_sconosciuto')
        
        url_depth = 0
        if raw_url != 'URL_sconosciuto':
            parsed_url = urlparse(raw_url)
            path_segments = [seg for seg in parsed_url.path.split('/') if seg]
            url_depth = len(path_segments)
        
        safe_name = raw_url.replace("https://", "").replace("http://", "")
        safe_name = re.sub(r'[<>:"/\\|?*]', '-', safe_name)[:100]
        
        filename = f"sample_{i}_{suffix}_depth{url_depth}_{safe_name}.html"
        filename = filename.replace("__", "_")
        filepath = os.path.join(output_dir, filename)
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("<meta charset='utf-8'>\n")
                f.write(f"<!-- SOURCE URL: {raw_url} -->\n")
                f.write(f"<!-- URL DEPTH: {url_depth} -->\n")
                f.write(doc.page_content)
        except Exception as e:
            print(f"[Errore] Impossibile salvare il file {filename}: {e}")
            
    print("Salvataggio completato con successo!")
    return actual_size

if __name__ == "__main__":
    docs = scrape_all_domains()
    # Cambia sample_size a -1 se vuoi salvare tutto su disco per ispezione
    save_scraped_data(docs, sample_size=-1, suffix="")