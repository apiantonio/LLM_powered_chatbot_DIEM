"""
Configurazione centralizzata del sistema RAG DIEM.

Principi:
- Single Source of Truth per tutti i parametri del sistema.
- Nessun valore hardcodato nei moduli applicativi.
- Type-safe tramite dataclass + factory method.
- Le API key NON vengono mai hardcodate; si leggono dall'ambiente.

REFACTORING MULTI-COLLECTION:
  Aggiunti parametri per-collection HTML chunking (docenti, offerta, bandi, dipartimento).
  Aggiunti parametri pdf_direct_chunk_size/overlap per bandi e PDF generici.
  Aggiunto get_collection_html_params() per l'indexer.
  VectorStoreConfig ora include parent_child_collection_name.

KPI Impact: Tutti. La configurazione centralizzata garantisce riproducibilità
degli esperimenti e facilita il tuning dei parametri per massimizzare le metriche RAGAS.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse


@dataclass(frozen=True)
class IngestionConfig:
    """Parametri per la pipeline di scraping e indicizzazione."""
    
    # --- Sorgenti dati ---
    html_raw_dir: str = "data/raw/html_samples"
    pdf_links_file: str = "data/raw/html_samples/pdf_links.txt"
    pdf_download_dir: str = "data/raw/pdfs"
    
    # --- Domini consentiti (bounded knowledge scope) ---
    seed_urls: tuple[str, ...] = (
        "https://www.diem.unisa.it",
        "https://corsi.unisa.it/ingegneria-informatica",
        "https://corsi.unisa.it/ingegneria-dell-informazione-per-la-medicina-digitale",
        "https://corsi.unisa.it/ingegneria-informatica-magistrale",
        "https://corsi.unisa.it/electrical-engineering-for-digital-energy",
        "https://corsi.unisa.it/information-Engineering-for-digital-medicine",
        "https://corsi.unisa.it/ingegneria-dell-informazione",
        "https://corsi.unisa.it/photovoltaics",
        "https://easycourse.unisa.it/",
    )
    
    # --- Crawler ---
    max_depth: int = 5
    batch_size: int = 1024
    crawl_delay_seconds: float = 2.0
    
    # --- Pulizia (centralizzata per scrapers e transform) ---
    cutoff_year: int = 2020
    target_department: str = "300638"
    ignored_extensions: tuple[str, ...] = (
        '.css', '.js', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp',
        '.zip', '.rar', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    )
    
    # --- Chunking HTML LEGACY (fallback per compatibilità) ---
    html_chunk_size: int = 700
    html_chunk_overlap: int = 50
    
    # --- Chunking HTML PER-COLLECTION ---
    # Ogni collection ha parametri ottimizzati per il tipo di contenuto.
    # Docenti e dipartimento: chunk più ampi (pagine con molto contesto).
    # Offerta e bandi: chunk standard (contenuto più strutturato).
    docenti_html_chunk_size: int = 800
    docenti_html_chunk_overlap: int = 100
    
    offerta_html_chunk_size: int = 700
    offerta_html_chunk_overlap: int = 50
    
    bandi_html_chunk_size: int = 700
    bandi_html_chunk_overlap: int = 50
    
    dipartimento_html_chunk_size: int = 800
    dipartimento_html_chunk_overlap: int = 100
    
    # --- Chunking PDF Parent-Child (SOLO per regolamenti/piani di studio) ---
    pdf_parent_chunk_size: int = 3000
    pdf_parent_chunk_overlap: int = 500
    pdf_child_chunk_size: int = 400
    pdf_child_chunk_overlap: int = 50
    
    # --- Chunking PDF diretto (bandi e PDF generici, NO Parent-Child) ---
    pdf_direct_chunk_size: int = 1500
    pdf_direct_chunk_overlap: int = 200
    
    # --- Registro incrementale (hash-based deduplication) ---
    index_registry_path: str = "data/vectorstore/index_registry.json"
    
    # --- Derived helpers ---
    
    def get_allowed_domains(self) -> set[str]:
        """Deriva i domini consentiti dalle seed_urls."""
        domains = set()
        for url in self.seed_urls:
            parsed = urlparse(url)
            if parsed.netloc:
                domains.add(parsed.netloc)
        domains.add("docenti.unisa.it")
        return domains
    
    def get_allowed_prefixes(self) -> tuple[str, ...]:
        """Deriva i prefissi consentiti dalle seed_urls (esclude easycourse)."""
        return tuple(
            url for url in self.seed_urls 
            if "easycourse" not in url
        )
    
    def get_collection_html_params(self, collection_name: str) -> tuple[int, int]:
        """
        Restituisce (chunk_size, chunk_overlap) per una data collection.
        
        L'indexer chiama questo metodo per costruire lo splitter HTML
        specifico di ogni collection, garantendo Single Source of Truth.
        
        Args:
            collection_name: Il value dell'enum CollectionTarget
                             (es. "docenti_e_didattica").
        
        Returns:
            Tupla (chunk_size, chunk_overlap).
        """
        mapping = {
            "docenti_e_didattica": (
                self.docenti_html_chunk_size,
                self.docenti_html_chunk_overlap,
            ),
            "offerta_formativa_e_corsi": (
                self.offerta_html_chunk_size,
                self.offerta_html_chunk_overlap,
            ),
            "bandi_e_amministrazione": (
                self.bandi_html_chunk_size,
                self.bandi_html_chunk_overlap,
            ),
            "dipartimento_e_ricerca": (
                self.dipartimento_html_chunk_size,
                self.dipartimento_html_chunk_overlap,
            ),
        }
        return mapping.get(
            collection_name,
            (self.html_chunk_size, self.html_chunk_overlap),
        )


@dataclass(frozen=True)
class EmbeddingConfig:
    """Parametri per il modello di embedding."""
    
    model_name: str = "BAAI/bge-m3"
    normalize_embeddings: bool = True
    expected_dim: int = 1024


@dataclass(frozen=True)
class VectorStoreConfig:
    """
    Parametri per il database vettoriale.
    
    REFACTORING: collection_name mantenuto per backward compatibility.
    Le 4 collection usano nomi derivati dall'enum CollectionTarget.
    parent_child_collection_name è la collection Chroma dedicata ai
    child chunks del ParentDocumentRetriever.
    """
    
    persist_directory: str = "data/vectorstore/chroma"
    collection_name: str = "diem_knowledge_base"  # legacy, non più usato
    parent_store_directory: str = "data/vectorstore/parent_docstore"
    search_type: str = "similarity"
    search_k: int = 20
    
    # Collection Chroma dedicata ai child chunks del Parent-Child
    parent_child_collection_name: str = "offerta_formativa_pdf_childs"


@dataclass(frozen=True)
class RerankerConfig:
    """Parametri per il Cross-Encoder di re-ranking post-retrieval."""
    
    model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    top_n: int = 5


@dataclass(frozen=True)
class LLMConfig:
    """Parametri per il Large Language Model (Strategy Pattern ready)."""
    
    provider: str = "huggingface"
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    temperature: float = 0.1
    max_tokens: int = 1024
    
    huggingface_api_token: Optional[str] = field(default=None)
    openai_api_key: Optional[str] = field(default=None)
    ollama_base_url: str = "http://localhost:11434"


@dataclass(frozen=True)
class EasyCourseConfig:
    """Parametri per il tool EasyCourse."""
    
    base_url: str = "https://easycourse.unisa.it"
    timeout: int = 30
    user_agent: str = "DIEM-RAG-Bot/1.0 (Università di Salerno)"


@dataclass(frozen=True)
class GuardrailsConfig:
    """Parametri per i guardrails di sicurezza."""
    
    allowed_scope_description: str = (
        "Domande relative al Dipartimento DIEM dell'Università degli Studi di Salerno: "
        "corsi di laurea, docenti, orari, esami, regolamenti, tesi, borse di studio, "
        "laboratori, servizi dipartimentali, dottorato di ricerca."
    )
    max_agent_iterations: int = 10
    enable_pii_filter: bool = True


@dataclass(frozen=True)
class ObservabilityConfig:
    """Parametri per logging e tracciamento della pipeline."""
    
    enable_verbose_callbacks: bool = True
    log_retrieved_chunks: bool = True
    log_tool_invocations: bool = True
    max_chunk_preview_chars: int = 200


@dataclass(frozen=True)
class AppSettings:
    """Aggregatore di tutte le configurazioni."""
    
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vectorstore: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    easycourse: EasyCourseConfig = field(default_factory=EasyCourseConfig)
    guardrails: GuardrailsConfig = field(default_factory=GuardrailsConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)


def load_settings() -> AppSettings:
    """
    Factory method che costruisce le impostazioni leggendo le variabili d'ambiente.
    Ogni parametro ha un default sensato; le env var permettono l'override.
    """
    return AppSettings(
        ingestion=IngestionConfig(
            html_raw_dir=os.getenv("HTML_RAW_DIR", "data/raw/html_samples"),
            pdf_links_file=os.getenv("PDF_LINKS_FILE", "data/raw/html_samples/pdf_links.txt"),
            pdf_download_dir=os.getenv("PDF_DOWNLOAD_DIR", "data/raw/pdfs"),
            cutoff_year=int(os.getenv("CUTOFF_YEAR", "2020")),
            target_department=os.getenv("TARGET_DEPARTMENT", "300638"),
        ),
        llm=LLMConfig(
            provider=os.getenv("LLM_PROVIDER", "ollama"),
            model_name=os.getenv("LLM_MODEL", "llama3.2"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            huggingface_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        ),
        embedding=EmbeddingConfig(
            model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        ),
        vectorstore=VectorStoreConfig(
            persist_directory=os.getenv("CHROMA_PERSIST_DIR", "data/vectorstore/chroma"),
            parent_store_directory=os.getenv("PARENT_STORE_DIR", "data/vectorstore/parent_docstore"),
        ),
        easycourse=EasyCourseConfig(
            base_url=os.getenv("EASYCOURSE_BASE_URL", "https://easycourse.unisa.it"),
            timeout=int(os.getenv("EASYCOURSE_TIMEOUT", "30")),
        ),
        observability=ObservabilityConfig(
            enable_verbose_callbacks=os.getenv("ENABLE_VERBOSE_CALLBACKS", "true").lower() == "true",
        ),
    )