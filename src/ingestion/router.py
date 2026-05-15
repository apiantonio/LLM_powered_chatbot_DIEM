"""
ingestion/router.py — Routing RIVISTO secondo audit_fattibilita_metadati.md

REFACTORING ARCHITETTURALE (audit §6 e §8):
  L'architettura passa da 4 Vector Store a 3 Vector Store:
    1. PERSONE (ex DOCENTI_DIDATTICA) — pagine docenti su docenti.unisa.it
    2. OFFERTA_FORMATIVA — pagine corsi su corsi.unisa.it + PDF regolamenti/piani
    3. DIPARTIMENTO — tutto diem.unisa.it INCLUSI i bandi (ex collection separata)

  La collection BANDI_AMMINISTRAZIONE è stata ELIMINATA e assorbita in
  DIPARTIMENTO con sotto_area = "bandi".

BUG FIX APPLICATO:
  - File tipo corsi.unisa.it-{nome_corso}-strutture-didattiche → DIPARTIMENTO
  - File tipo corsi.unisa.it-{nome_corso}-terza-missione.html → DIPARTIMENTO
  Questi file riguardano strutture fisiche e terza missione del dipartimento,
  NON l'offerta formativa del corso.

METADATI ESTRATTI (audit §6 — Schema Definitivo):
  PERSONE: matricola, nome_docente, sotto_area, nomi_insegnamenti, laboratorio_nome,
           formato_sorgente, url_originale, anno
  OFFERTA_FORMATIVA: nome_corso, corso_slug, sotto_area,
                     formato_sorgente, url_originale, anno
  DIPARTIMENTO: sotto_area, tipo_bando, anno_bando, laboratorio_nome,
                laboratorio_id, formato_sorgente, url_originale

CHUNKING CONTEXT-AWARE (audit §5, risoluzione problema chunk orfani):
  Ogni chunk DEVE contenere nei metadati le informazioni identificative
  del documento sorgente (nome_docente, matricola, nome_corso, ecc.)
  così che anche i chunk successivi al primo siano semanticamente
  autocontestualizzati. Questo risolve il problema dei "chunk orfani"
  dove solo il primo chunk contiene il nome del docente/corso.
"""

import re
import logging
from enum import Enum
from typing import Optional, List

logger = logging.getLogger(__name__)


# ============================================================
# 3 VECTOR STORE — audit §8
# ============================================================

class CollectionTarget(str, Enum):
    """
    Enum delle 3 collection — RIVISTO da audit_fattibilita_metadati.md §8.

    CAMBIO ARCHITETTURALE:
      - DOCENTI_DIDATTICA rinominato in PERSONE
      - BANDI_AMMINISTRAZIONE ELIMINATA (assorbita in DIPARTIMENTO)
      - DIPARTIMENTO_RICERCA rinominato in DIPARTIMENTO
    """
    PERSONE = "persone"
    OFFERTA_FORMATIVA = "offerta_formativa"
    DIPARTIMENTO = "dipartimento"


# ============================================================
# Mapping sottosezione docente — audit §6 "sotto_area"
# ============================================================

class DocenteSezione(str, Enum):
    """
    Sottosezioni logiche delle pagine docente — audit §6 VS PERSONE.

    Mapping deterministico URL → sotto_area:
      /home, /curriculum → profilo
      /didattica         → didattica
      /ricerca           → ricerca
      /international     → internazionale
      /risorse           → risorse
    """
    PROFILO = "profilo"
    DIDATTICA = "didattica"
    RICERCA = "ricerca"
    INTERNAZIONALE = "internazionale"
    RISORSE = "risorse"


def _classify_docente_sezione(source_url: str) -> Optional[str]:
    """
    Determina la sotto_area del docente dal path URL — audit §6 VS PERSONE.

    Mapping deterministico confermato dall'audit:
      /home, /curriculum     → "profilo"
      /didattica             → "didattica"
      /ricerca               → "ricerca"
      /international         → "internazionale"
      /risorse               → "risorse"
    """
    url_lower = source_url.lower()

    if "/home" in url_lower or "/curriculum" in url_lower:
        return DocenteSezione.PROFILO.value
    elif "/didattica" in url_lower:
        return DocenteSezione.DIDATTICA.value
    elif "/ricerca" in url_lower:
        return DocenteSezione.RICERCA.value
    elif "/international" in url_lower:
        return DocenteSezione.INTERNAZIONALE.value
    elif "/risorse" in url_lower:
        return DocenteSezione.RISORSE.value

    # Fallback per URL docente senza sottopagina specifica
    if "docenti.unisa.it/" in url_lower:
        return DocenteSezione.PROFILO.value

    return None


# ============================================================
# Mapping sotto_area DIPARTIMENTO — audit §6 VS DIPARTIMENTO
# ============================================================

def _classify_dipartimento_sottoarea(source_url: str) -> str:
    """
    Determina la sotto_area per le pagine del dipartimento — audit §6.

    Mapping deterministico da URL:
      /ricerca/              → ricerca_dipartimentale
      /terza-missione/       → terza_missione
      /dipartimento/strutture → laboratori
      /international/        → internazionale
      /home/bandi, /bandi    → bandi
      /didattica/alternanza  → alternanza
      /dipartimento/eccellenza → eccellenza
      /home/dati-di-monitoraggio → monitoraggio
      strutture-didattiche, aula, aule → aule
      laboratorio, laboratori → laboratori
      sede, edificio, campus → sedi
    """
    url_lower = source_url.lower()

    # --- Bandi (audit §5.4: bandi assorbiti in DIPARTIMENTO) ---
    if re.search(r"/bandi|/home/bandi", url_lower):
        return "bandi"

    # --- Strutture fisiche: aule ---
    if re.search(r"strutture[-_]didattiche|aul[ae]", url_lower):
        return "aule"

    # --- Strutture fisiche: laboratori ---
    if re.search(r"/dipartimento/strutture", url_lower):
        return "laboratori"
    if re.search(r"laborator[io]|laboratori|\blab\b", url_lower):
        return "laboratori"

    # --- Strutture fisiche: sedi ---
    if re.search(r"\bsed[ei]\b|edifici[o]|campus", url_lower):
        return "sedi"

    # --- Aree tematiche ---
    if "/ricerca" in url_lower:
        return "ricerca_dipartimentale"
    if "/terza-missione" in url_lower:
        return "terza_missione"
    if "/international" in url_lower:
        return "internazionale"
    if "/didattica/alternanza" in url_lower:
        return "alternanza"
    if "/dipartimento/eccellenza" in url_lower:
        return "eccellenza"
    if "/dati-di-monitoraggio" in url_lower:
        return "monitoraggio"
    if "/dipartimento/personale" in url_lower:
        return "personale"

    return "generale"


# ============================================================
# Mapping sotto_area OFFERTA_FORMATIVA — audit §6 VS OFFERTA
# ============================================================

def _classify_offerta_sottoarea_html(source_url: str) -> str:
    """sotto_area per pagine HTML di corsi.unisa.it — audit §6."""
    url_lower = source_url.lower()

    if "/strutture-didattiche" in url_lower:
        return "aule"
    if "/didattica" in url_lower:
        return "didattica"
    if "/terza-missione" in url_lower:
        return "terza_missione"

    return "informazioni_corso"


def _classify_offerta_sottoarea_pdf(pdf_url: str) -> str:
    """
    sotto_area per PDF di OFFERTA_FORMATIVA — audit §6.

    Keyword nel path URL:
      __statistiche-corsi → statistiche
      __regolamenti-cds   → regolamenti
      __piano-studi-cds   → piani_di_studio
      aule-didattiche     → aule
    """
    url_lower = pdf_url.lower()

    if "__statistiche-corsi" in url_lower:
        return "statistiche"
    if "__regolamenti-cds" in url_lower:
        return "regolamenti"
    if "__piano-studi-cds" in url_lower:
        return "piani_di_studio"
    if "aule-didattiche" in url_lower or "aule" in url_lower:
        return "aule"

    return "documentazione_corso"


# ============================================================
# ESTRAZIONE METADATI CONTEXT-AWARE — audit §6 + §5
# ============================================================

def _extract_nome_docente_from_html(html_content: str) -> Optional[str]:
    """
    Estrae il nome del docente dal tag <h1> dell'HTML — audit §6 PERSONE.

    Pattern costante: <h1>...<span>Nome COGNOME | </span>...
    Il testo prima del " | " nel primo <span> contiene il nome del docente.
    """
    match = re.search(
        r"<h1[^>]*>.*?<span[^>]*>([^<]+?)\s*\|\s*</span>",
        html_content, re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip()
    return None


def _extract_nome_corso_from_url(source_url: str) -> Optional[str]:
    """
    Estrae il nome del corso direttamente dall'URL.
    Pattern: corsi.unisa.it/{nome_corso}/...
    """
    match = re.search(
        r"corsi\.unisa\.it/([^/]+)", 
        source_url, 
        re.IGNORECASE
    )
    if match:
        nome_grezzo = match.group(1).strip()
        if nome_grezzo.lower() != "uploads":
            nome_pulito = nome_grezzo.replace("-", " ").title()
            return nome_pulito
    return None


def _extract_nome_insegnamento_from_html(html_content: str) -> Optional[str]:
    """
    Estrae il nome dell'insegnamento dal secondo <span> dell'<h1> — audit §6 PERSONE.

    Nelle pagine docente/didattica a depth 4 (singolo insegnamento),
    il secondo <span> contiene il nome dell'insegnamento.
    """
    match = re.search(
        r"<h1[^>]*>.*?<span[^>]*>[^<]+</span>\s*<span[^>]*>([^<]+)</span>",
        html_content, re.DOTALL | re.IGNORECASE
    )
    if match:
        nome = match.group(1).strip()
        # Filtra valori che sono sezioni, non insegnamenti
        sezioni_note = {"Home", "Curriculum", "Didattica", "Ricerca",
                        "International", "Risorse", "Laboratori"}
        if nome not in sezioni_note:
            return nome
    return None


def _extract_anno_from_url(source_url: str) -> Optional[str]:
    """
    Estrae l'anno dal parametro ?anno= nell'URL — audit §6.

    Usato per sotto_area didattica e internazionale (PERSONE)
    e per pagine HTML di OFFERTA_FORMATIVA.
    """
    match = re.search(r"[?&]anno=(\d{4})", source_url)
    if match:
        return match.group(1)
    return None


def _extract_anno_from_pdf_path(pdf_url: str) -> Optional[str]:
    """
    Ricalcola l'anno dai PDF — audit §5.5 (adattamento C6).

    Estrae dal path URL: __regolamenti-cds/2023/ → "2023"
                         __statistiche-corsi/2025_09/ → "2025"
    """
    match = re.search(r"/(\d{4})(?:[/_]|\.pdf)", pdf_url)
    if match:
        anno = match.group(1)
        # Sanity check: l'anno deve essere ragionevole (2000-2030)
        if 2000 <= int(anno) <= 2030:
            return anno
    return None


def _extract_laboratorio_info(html_content: str, source_url: str) -> dict:
    """
    Estrae informazioni su laboratorio — audit §6 VS DIPARTIMENTO.

    laboratorio_nome: dal secondo <span> dell'<h1>
    laboratorio_id: dal parametro ?id= nell'URL
    """
    info = {}

    # laboratorio_id dal parametro ?id=
    id_match = re.search(r"[?&]id=(\d+)", source_url)
    if id_match:
        info["laboratorio_id"] = id_match.group(1)

    # laboratorio_nome dal secondo <span> dell'<h1>
    nome_match = re.search(
        r"<h1[^>]*>.*?<span[^>]*>[^<]+</span>\s*<span[^>]*>([^<]+)</span>",
        html_content, re.DOTALL | re.IGNORECASE
    )
    if nome_match:
        info["laboratorio_nome"] = nome_match.group(1).strip()

    return info


def _extract_tipo_bando_from_html(html_content: str) -> Optional[str]:
    """
    Estrae il tipo di bando dal secondo <span> dell'<h1> — audit §6.

    Es. "Bandi Incarichi di Insegnamento" → tipo_bando
    """
    match = re.search(
        r"<h1[^>]*>.*?<span[^>]*>[^<]+</span>\s*<span[^>]*>([^<]+)</span>",
        html_content, re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip()
    return None


# ============================================================
# DOCUMENT ROUTER — Rivisto per 3 Vector Store
# ============================================================

class DocumentRouter:
    """
    Router aggiornato per 3 Vector Store — audit §8.

    Cambiamenti rispetto alla versione precedente:
      - Le regole di routing mappano a 3 collection (non 4)
      - I bandi (diem.unisa.it/bandi) vanno in DIPARTIMENTO (non in collection separata)
      - I metadati estratti seguono lo schema §6 dell'audit
      - Nuove funzioni di estrazione per chunking context-aware

    BUG FIX APPLICATO:
      - corsi.unisa.it/{corso}/strutture-didattiche → DIPARTIMENTO (strutture fisiche)
      - corsi.unisa.it/{corso}/terza-missione → DIPARTIMENTO (terza missione dipartimentale)
      Questi file NON sono offerta formativa ma riguardano strutture fisiche
      e attività istituzionali del dipartimento.
    """

    # --- Regole di routing HTML (ordine: dal più specifico al più generico) ---
    #
    # BUG FIX: Le regole per corsi.unisa.it sono ora ordinate dal più
    # specifico al più generico. Le pagine "strutture-didattiche" e
    # "terza-missione" vengono intercettate PRIMA della regola generica
    # per corsi.unisa.it/ e instradadate in DIPARTIMENTO.
    _HTML_RULES = [
        # PERSONE: tutte le pagine docente
        (r"docenti\.unisa\.it/", CollectionTarget.PERSONE),

        # ── BUG FIX: intercetta PRIMA della regola generica corsi.unisa.it ──
        # strutture-didattiche su corsi.unisa.it → DIPARTIMENTO (sono aule/strutture fisiche)
        (r"corsi\.unisa\.it/[^/]+/strutture[-_]didattiche", CollectionTarget.DIPARTIMENTO),
        # terza-missione su corsi.unisa.it → DIPARTIMENTO (sono attività istituzionali)
        (r"corsi\.unisa\.it/[^/]+/terza[-_]missione", CollectionTarget.DIPARTIMENTO),

        # OFFERTA_FORMATIVA: pagine corsi (regola generica, DOPO le eccezioni)
        (r"corsi\.unisa\.it/", CollectionTarget.OFFERTA_FORMATIVA),

        # DIPARTIMENTO: tutto diem.unisa.it (inclusi bandi)
        # Regola specifica per didattica DIEM → OFFERTA_FORMATIVA
        (r"diem\.unisa\.it/didattica(?!/alternanza)", CollectionTarget.OFFERTA_FORMATIVA),

        # Personale docente DIEM → PERSONE
        (r"diem\.unisa\.it/dipartimento/personale", CollectionTarget.PERSONE),

        # Tutto il resto di diem.unisa.it → DIPARTIMENTO (inclusi bandi!)
        (r"diem\.unisa\.it/", CollectionTarget.DIPARTIMENTO),
    ]

    # --- Regole di routing PDF ---
    _PDF_RULES = [
        # Regolamenti e piani di studio → OFFERTA_FORMATIVA
        (r"__regolamenti-cds/", CollectionTarget.OFFERTA_FORMATIVA),
        (r"__piano-studi-cds/", CollectionTarget.OFFERTA_FORMATIVA),
        (r"__statistiche-corsi/", CollectionTarget.OFFERTA_FORMATIVA),
        (r"corsi\.unisa\.it/", CollectionTarget.OFFERTA_FORMATIVA),

        # Bandi PDF → DIPARTIMENTO (audit §5.4: bandi in DIPARTIMENTO)
        (r"diem\.unisa\.it/uploads/rescue/\d+/\d+/", CollectionTarget.DIPARTIMENTO),
        (r"diem\.unisa\.it/", CollectionTarget.DIPARTIMENTO),
    ]

    @classmethod
    def route_html(cls, source_url: str) -> CollectionTarget:
        """
        Determina la collection target per un documento HTML.

        BUG FIX: Le regole sono ora ordinate dal più specifico al più generico.
        I file corsi.unisa.it/{corso}/strutture-didattiche e
        corsi.unisa.it/{corso}/terza-missione vengono ora correttamente
        instradati in DIPARTIMENTO anziché in OFFERTA_FORMATIVA.
        """
        for pattern, target in cls._HTML_RULES:
            if re.search(pattern, source_url, re.IGNORECASE):
                return target
        return CollectionTarget.DIPARTIMENTO

    @classmethod
    def route_pdf(cls, pdf_url: str) -> CollectionTarget:
        """Determina la collection target per un documento PDF."""
        for pattern, target in cls._PDF_RULES:
            if re.search(pattern, pdf_url, re.IGNORECASE):
                return target
        return CollectionTarget.DIPARTIMENTO

    @classmethod
    def use_parent_child(cls, collection: CollectionTarget, pdf_url: str) -> bool:
        """
        Determina se usare Parent-Child per il PDF.

        Solo i regolamenti e piani di studio di OFFERTA_FORMATIVA
        usano il ParentDocumentRetriever.
        """
        if collection != CollectionTarget.OFFERTA_FORMATIVA:
            return False
        return bool(re.search(r"(__regolamenti-cds|__piano-studi-cds)/", pdf_url))

    @classmethod
    def extract_metadata(cls, source_url: str, collection: CollectionTarget) -> dict:
        """
        Estrae metadati BASE dall'URL — audit §6 schema definitivo.

        Questi metadati vengono assegnati a OGNI chunk del documento,
        garantendo che nessun chunk sia "orfano" (privo di contesto).

        I metadati estratti dal CONTENUTO HTML (nome_docente, nome_corso,
        laboratorio_nome, ecc.) vengono aggiunti separatamente tramite
        extract_content_metadata() durante la fase di chunking.
        """
        metadata = {
            "formato_sorgente": "html",
            "url_originale": source_url,
            "source_domain": _extract_domain(source_url),
        }

        if collection == CollectionTarget.PERSONE:
            # --- PERSONE: matricola + sotto_area — audit §6 ---
            matricola_match = re.search(r"docenti\.unisa\.it/(\d+)/", source_url)
            if matricola_match:
                metadata["matricola"] = matricola_match.group(1)

            sezione = _classify_docente_sezione(source_url)
            if sezione:
                metadata["sotto_area"] = sezione

            # Anno (solo per didattica e internazionale)
            anno = _extract_anno_from_url(source_url)
            if anno:
                metadata["anno"] = anno

        elif collection == CollectionTarget.OFFERTA_FORMATIVA:
            # --- OFFERTA: corso_slug + sotto_area — audit §6 ---
            corso_match = re.search(r"corsi\.unisa\.it/([^/]+)", source_url)
            if corso_match:
                slug = corso_match.group(1)
                # Escludi "uploads" che non è uno slug di corso
                if slug != "uploads":
                    metadata["corso_slug"] = slug

            metadata["sotto_area"] = _classify_offerta_sottoarea_html(source_url)

            anno = _extract_anno_from_url(source_url)
            if anno:
                metadata["anno"] = anno

        elif collection == CollectionTarget.DIPARTIMENTO:
            # --- DIPARTIMENTO: sotto_area — audit §6 ---
            metadata["sotto_area"] = _classify_dipartimento_sottoarea(source_url)

            # Anno per bandi
            anno = _extract_anno_from_url(source_url)
            if anno:
                metadata["anno_bando"] = anno

        return metadata

    @classmethod
    def extract_pdf_metadata(cls, pdf_url: str, collection: CollectionTarget) -> dict:
        """
        Estrae metadati per i PDF — audit §6 + §5.5 (anno ricalcolato).
        """
        metadata = {
            "formato_sorgente": "pdf",
            "url_originale": pdf_url,
            "source_domain": _extract_domain(pdf_url),
        }

        if collection == CollectionTarget.OFFERTA_FORMATIVA:
            metadata["sotto_area"] = _classify_offerta_sottoarea_pdf(pdf_url)

            # Anno ricalcolato dal path — audit §5.5
            anno = _extract_anno_from_pdf_path(pdf_url)
            if anno:
                metadata["anno"] = anno

            # Tentativo estrazione corso_slug dal path
            corso_match = re.search(r"corsi\.unisa\.it/([^/]+)", pdf_url)
            if corso_match:
                slug = corso_match.group(1)
                if slug != "uploads":
                    metadata["corso_slug"] = slug

        elif collection == CollectionTarget.DIPARTIMENTO:
            metadata["sotto_area"] = "bandi"

            anno = _extract_anno_from_pdf_path(pdf_url)
            if anno:
                metadata["anno_bando"] = anno

        return metadata

    @classmethod
    def extract_content_metadata(
        cls,
        html_content: str,
        source_url: str,
        collection: CollectionTarget,
    ) -> dict:
        """
        Estrae metadati dal CONTENUTO HTML — audit §6.

        QUESTA È LA FUNZIONE CHIAVE PER IL CHUNKING CONTEXT-AWARE:
        I metadati estratti qui (nome_docente, nome_corso, ecc.) vengono
        iniettati in OGNI chunk del documento, risolvendo il problema
        dei chunk orfani descritto nei report di chunking.

        Metadati estratti per collection:
          PERSONE: nome_docente, nomi_insegnamenti, laboratorio_nome
          OFFERTA: nome_corso
          DIPARTIMENTO: tipo_bando, laboratorio_nome, laboratorio_id
        """
        content_meta = {}

        if collection == CollectionTarget.PERSONE:
            # nome_docente: dal primo <span> dell'<h1>
            nome = _extract_nome_docente_from_html(html_content)
            if nome:
                content_meta["nome_docente"] = nome

            # Estraiamo in modo neutro il valore presente nel secondo span
            valore_estratto = _extract_nome_insegnamento_from_html(html_content)
            
            if valore_estratto:
                url_lower = source_url.lower()
                
                # Caso 1: Laboratori (richiedono la presenza dell'ID)
                if "/laboratori" in url_lower:
                    if re.search(r"[?&]id=\d+", url_lower):
                        content_meta["laboratorio_nome"] = valore_estratto
                
                # Caso 2: Altre sezioni di ricerca (NON richiedono l'ID)
                elif "/spin-off" in url_lower:
                    content_meta["spin_off"] = valore_estratto
                
                elif "/premi-ricerca" in url_lower:
                    content_meta["premi_ricerca"] = valore_estratto
                
                elif "/brevetti" in url_lower:
                    content_meta["brevetti"] = valore_estratto
                
                elif "/pubblicazioni" in url_lower:
                    content_meta["pubblicazioni"] = valore_estratto
                
                elif "/progetti" in url_lower:
                    content_meta["progetti"] = valore_estratto
                
                # Caso 3: Qualsiasi altra pagina (es. didattica)
                else:
                    content_meta["nomi_insegnamenti"] = valore_estratto

        elif collection == CollectionTarget.OFFERTA_FORMATIVA:
            # nome_corso: dal primo <span> dell'<h1> — audit §6
            nome = _extract_nome_corso_from_url(source_url)
            if nome:
                content_meta["nome_corso"] = nome

        elif collection == CollectionTarget.DIPARTIMENTO:
            sotto_area = _classify_dipartimento_sottoarea(source_url)

            if sotto_area == "bandi":
                tipo = _extract_tipo_bando_from_html(html_content)
                if tipo:
                    content_meta["tipo_bando"] = tipo

            elif sotto_area == "laboratori":
                lab_info = _extract_laboratorio_info(html_content, source_url)
                content_meta.update(lab_info)

        return content_meta


def _extract_domain(url: str) -> str:
    """Estrae il dominio da un URL."""
    match = re.search(r"https?://([^/]+)", url)
    return match.group(1) if match else "unknown"