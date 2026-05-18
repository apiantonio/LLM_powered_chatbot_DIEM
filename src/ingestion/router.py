import re
import logging
from enum import Enum
from typing import Optional, List

logger = logging.getLogger(__name__)


class CollectionTarget(str, Enum):
    PERSONE = "persone"
    OFFERTA_FORMATIVA = "offerta_formativa"
    DIPARTIMENTO = "dipartimento"

class DocenteSezione(str, Enum):
    PROFILO = "profilo"
    DIDATTICA = "didattica"
    RICERCA = "ricerca"
    INTERNAZIONALE = "internazionale"
    RISORSE = "risorse"


def _classify_docente_sezione(source_url: str) -> Optional[str]:
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
    
    if "docenti.unisa.it/" in url_lower:
        return DocenteSezione.PROFILO.value

    return None

def _classify_dipartimento_sottoarea(source_url: str) -> str:
    url_lower = source_url.lower()

    if re.search(r"/bandi|/home/bandi", url_lower):
        return "bandi"


    if re.search(r"strutture[-_]didattiche|aul[ae]", url_lower):
        return "aule"

    if re.search(r"/dipartimento/strutture", url_lower):
        return "laboratori"
    if re.search(r"laborator[io]|laboratori|\blab\b", url_lower):
        return "laboratori"

    if re.search(r"\bsed[ei]\b|edifici[o]|campus", url_lower):
        return "sedi"


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


def _extract_nome_docente_from_html(html_content: str) -> Optional[str]:
    match = re.search(
        r"<h1[^>]*>.*?<span[^>]*>([^<]+?)\s*\|\s*</span>",
        html_content, re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip()
    return None


def _extract_nome_corso_from_url(source_url: str) -> Optional[str]:
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
    match = re.search(
        r"<h1[^>]*>.*?<span[^>]*>[^<]+</span>\s*<span[^>]*>([^<]+)</span>",
        html_content, re.DOTALL | re.IGNORECASE
    )
    if match:
        nome = match.group(1).strip()
        sezioni_note = {"Home", "Curriculum", "Didattica", "Ricerca",
                        "International", "Risorse", "Laboratori"}
        if nome not in sezioni_note:
            return nome
    return None


def _extract_anno_from_url(source_url: str) -> Optional[str]:
    match = re.search(r"[?&]anno=(\d{4})", source_url)
    if match:
        return match.group(1)
    return None


def _extract_laboratorio_info(html_content: str, source_url: str) -> dict:
    info = {}

    id_match = re.search(r"[?&]id=(\d+)", source_url)
    if id_match:
        info["laboratorio_id"] = id_match.group(1)

    nome_match = re.search(
        r"<h1[^>]*>.*?<span[^>]*>[^<]+</span>\s*<span[^>]*>([^<]+)</span>",
        html_content, re.DOTALL | re.IGNORECASE
    )
    if nome_match:
        info["laboratorio_nome"] = nome_match.group(1).strip()

    return info


def _extract_tipo_bando_from_html(html_content: str) -> Optional[str]:
    match = re.search(
        r"<h1[^>]*>.*?<span[^>]*>[^<]+</span>\s*<span[^>]*>([^<]+)</span>",
        html_content, re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip()
    return None

class DocumentRouter:
    _HTML_RULES = [
        (r"docenti\.unisa\.it/", CollectionTarget.PERSONE),

        (r"corsi\.unisa\.it/[^/]+/strutture[-_]didattiche", CollectionTarget.DIPARTIMENTO),
        (r"corsi\.unisa\.it/[^/]+/terza[-_]missione", CollectionTarget.DIPARTIMENTO),

        (r"corsi\.unisa\.it/", CollectionTarget.OFFERTA_FORMATIVA),

        (r"diem\.unisa\.it/didattica(?!/alternanza)", CollectionTarget.OFFERTA_FORMATIVA),

        (r"diem\.unisa\.it/dipartimento/personale", CollectionTarget.PERSONE),

        (r"diem\.unisa\.it/", CollectionTarget.DIPARTIMENTO),
    ]
    _PDF_RULES = [
        (r"__regolamenti-cds/", CollectionTarget.OFFERTA_FORMATIVA),
        (r"__piano-studi-cds/", CollectionTarget.OFFERTA_FORMATIVA),
        (r"__statistiche-corsi/", CollectionTarget.OFFERTA_FORMATIVA),
        (r"corsi\.unisa\.it/", CollectionTarget.OFFERTA_FORMATIVA),
        (r"diem\.unisa\.it/uploads/rescue/\d+/\d+/", CollectionTarget.DIPARTIMENTO),
        (r"diem\.unisa\.it/", CollectionTarget.DIPARTIMENTO),
    ]

    @classmethod
    def route_html(cls, source_url: str) -> CollectionTarget:
        for pattern, target in cls._HTML_RULES:
            if re.search(pattern, source_url, re.IGNORECASE):
                return target
        return CollectionTarget.DIPARTIMENTO

    @classmethod
    def route_pdf(cls, pdf_url: str) -> CollectionTarget:
        for pattern, target in cls._PDF_RULES:
            if re.search(pattern, pdf_url, re.IGNORECASE):
                return target
        return CollectionTarget.DIPARTIMENTO

    @classmethod
    def use_parent_child(cls, collection: CollectionTarget, pdf_url: str) -> bool:
        if collection != CollectionTarget.OFFERTA_FORMATIVA:
            return False
        return bool(re.search(r"(__regolamenti-cds|__piano-studi-cds)/", pdf_url))

    @classmethod
    def extract_metadata(cls, source_url: str, collection: CollectionTarget) -> dict:
        metadata = {
            "formato_sorgente": "html",
            "url_originale": source_url,
            "source_domain": _extract_domain(source_url),
        }

        if collection == CollectionTarget.PERSONE:
            matricola_match = re.search(r"docenti\.unisa\.it/(\d+)/", source_url)
            if matricola_match:
                metadata["matricola"] = matricola_match.group(1)

            sezione = _classify_docente_sezione(source_url)
            if sezione:
                metadata["sotto_area"] = sezione
            anno = _extract_anno_from_url(source_url)
            if anno:
                metadata["anno"] = anno

        elif collection == CollectionTarget.OFFERTA_FORMATIVA:
            corso_match = re.search(r"corsi\.unisa\.it/([^/]+)", source_url)
            if corso_match:
                slug = corso_match.group(1)
                if slug != "uploads":
                    metadata["corso_slug"] = slug

            metadata["sotto_area"] = _classify_offerta_sottoarea_html(source_url)

            anno = _extract_anno_from_url(source_url)
            if anno:
                metadata["anno"] = anno

        elif collection == CollectionTarget.DIPARTIMENTO:
            metadata["sotto_area"] = _classify_dipartimento_sottoarea(source_url)

            anno = _extract_anno_from_url(source_url)
            if anno:
                metadata["anno"] = anno

        return metadata

    @classmethod
    def extract_pdf_metadata(cls, pdf_url: str, collection: CollectionTarget) -> dict:
        metadata = {
            "formato_sorgente": "pdf",
            "url_originale": pdf_url,
            "source_domain": _extract_domain(pdf_url),
        }

        if collection == CollectionTarget.OFFERTA_FORMATIVA:
            metadata["sotto_area"] = _classify_offerta_sottoarea_pdf(pdf_url)
            corso_match = re.search(r"corsi\.unisa\.it/([^/]+)", pdf_url)
            if corso_match:
                slug = corso_match.group(1)
                if slug != "uploads":
                    metadata["corso_slug"] = slug

        elif collection == CollectionTarget.DIPARTIMENTO:
            metadata["sotto_area"] = "bandi"

        return metadata

    @classmethod
    def extract_content_metadata(
        cls,
        html_content: str,
        source_url: str,
        collection: CollectionTarget,
    ) -> dict:
        content_meta = {}

        if collection == CollectionTarget.PERSONE:
            nome = _extract_nome_docente_from_html(html_content)
            if nome:
                content_meta["nome_docente"] = nome
            valore_estratto = _extract_nome_insegnamento_from_html(html_content)
            
            if valore_estratto:
                url_lower = source_url.lower()
                if "/laboratori" in url_lower:
                    if re.search(r"[?&]id=\d+", url_lower):
                        content_meta["laboratorio_nome"] = valore_estratto
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
                
                else:
                    content_meta["nomi_insegnamenti"] = valore_estratto

        elif collection == CollectionTarget.OFFERTA_FORMATIVA:
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

    @classmethod
    def route_md(cls, filename: str) -> CollectionTarget:
        return CollectionTarget.DIPARTIMENTO

    @classmethod
    def extract_md_metadata(cls, filename: str, collection: CollectionTarget) -> dict:
        return {
            "formato_sorgente": "md",
            "url_originale": f"local_static:{filename}",
            "source_domain": "localhost",
            "sotto_area": "generale"
        }

    @classmethod
    def extract_content_metadata_md(cls, md_content: str, filename: str) -> dict:
        content_meta = {}
        match = re.search(r"^#\s+(.+)$", md_content, re.MULTILINE)
        if match:
            content_meta["titolo_documento"] = match.group(1).strip()
        else:
            content_meta["titolo_documento"] = filename.replace(".md", "").replace("_", " ").title()
            
        return content_meta


def _extract_domain(url: str) -> str:
    match = re.search(r"https?://([^/]+)", url)
    return match.group(1) if match else "unknown"