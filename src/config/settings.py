"""
Configurazione centralizzata del sistema RAG DIEM.

Principi:
- Single Source of Truth per tutti i parametri del sistema.
- Nessun valore hardcodato nei moduli applicativi.
- Type-safe tramite dataclass + factory method.
- Le API key NON vengono mai hardcodate; si leggono dall'ambiente.

KPI Impact: Tutti. La configurazione centralizzata garantisce riproducibilità
degli esperimenti e facilita il tuning dei parametri per massimizzare le metriche RAGAS.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


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
    
    # --- Chunking HTML (diretto, no Parent-Child) ---
    html_chunk_size: int = 700
    html_chunk_overlap: int = 50
    
    # --- Chunking PDF (Parent-Child) ---
    pdf_parent_chunk_size: int = 3000
    pdf_parent_chunk_overlap: int = 500
    pdf_child_chunk_size: int = 400
    pdf_child_chunk_overlap: int = 50
    
    # --- Registro incrementale (hash-based deduplication) ---
    index_registry_path: str = "data/vectorstore/index_registry.json"


@dataclass(frozen=True)
class EmbeddingConfig:
    """Parametri per il modello di embedding."""
    
    model_name: str = "BAAI/bge-m3"
    normalize_embeddings: bool = True
    expected_dim: int = 1024


@dataclass(frozen=True)
class VectorStoreConfig:
    """Parametri per il database vettoriale."""
    
    persist_directory: str = "data/vectorstore/chroma"
    collection_name: str = "diem_knowledge_base"
    parent_store_directory: str = "data/vectorstore/parent_docstore"
    search_type: str = "similarity"
    search_k: int = 20


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
class AppSettings:
    """Aggregatore di tutte le configurazioni."""
    
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vectorstore: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    guardrails: GuardrailsConfig = field(default_factory=GuardrailsConfig)


def load_settings() -> AppSettings:
    """
    Factory method che costruisce le impostazioni leggendo le variabili d'ambiente.
    """
    return AppSettings(
        ingestion=IngestionConfig(
            html_raw_dir=os.getenv("HTML_RAW_DIR", "data/raw/html_samples"),
            pdf_links_file=os.getenv("PDF_LINKS_FILE", "data/raw/html_samples/pdf_links.txt"),
            pdf_download_dir=os.getenv("PDF_DOWNLOAD_DIR", "data/raw/pdfs"),
        ),
        llm=LLMConfig(
            provider=os.getenv("LLM_PROVIDER", "huggingface"),
            model_name=os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
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
    )
