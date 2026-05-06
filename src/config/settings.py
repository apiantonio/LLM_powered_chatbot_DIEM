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
    html_raw_dir: str = "data/raw/html_samples_v7"
    pdf_links_file: str = "data/raw/html_samples_v7/pdf_links_cleaned_new.txt"
    
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
        "https://easycourse.unisa.it/",  # REQUISITO: Orari corsi/esami
    )
    
    # --- Crawler ---
    max_depth: int = 5
    batch_size: int = 1024
    crawl_delay_seconds: float = 2.0
    
    # --- Chunking ---
    chunk_size: int = 700
    chunk_overlap: int = 50


@dataclass(frozen=True)
class EmbeddingConfig:
    """Parametri per il modello di embedding."""
    
    model_name: str = "BAAI/bge-m3"
    normalize_embeddings: bool = True
    # Dimensionalità attesa (per validazione runtime)
    expected_dim: int = 1024  # bge-m3 produce vettori a 1024 dimensioni


@dataclass(frozen=True)
class VectorStoreConfig:
    """Parametri per il database vettoriale."""
    
    persist_directory: str = "data/vectorstore/chroma"
    collection_name: str = "diem_knowledge_base"
    search_type: str = "similarity"  # "similarity" | "mmr"
    search_k: int = 20  # Top-K iniziale (prima del reranking)


@dataclass(frozen=True)
class RerankerConfig:
    """Parametri per il Cross-Encoder di re-ranking post-retrieval."""
    
    model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    top_n: int = 5  # Documenti finali dopo il re-ranking


@dataclass(frozen=True)
class LLMConfig:
    """Parametri per il Large Language Model (Strategy Pattern ready)."""
    
    # Provider: "ollama" | "huggingface" | "openai"
    provider: str = "huggingface"
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    temperature: float = 0.1
    max_tokens: int = 1024
    
    # API keys (lette dall'ambiente, mai hardcodate)
    huggingface_api_token: Optional[str] = field(default=None)
    openai_api_key: Optional[str] = field(default=None)
    
    # Ollama (se provider == "ollama")
    ollama_base_url: str = "http://localhost:11434"


@dataclass(frozen=True)
class GuardrailsConfig:
    """Parametri per i guardrails di sicurezza."""
    
    # Scope Awareness: lista di topic consentiti (whitelist semantica)
    allowed_scope_description: str = (
        "Domande relative al Dipartimento DIEM dell'Università degli Studi di Salerno: "
        "corsi di laurea, docenti, orari, esami, regolamenti, tesi, borse di studio, "
        "laboratori, servizi dipartimentali, dottorato di ricerca."
    )
    
    # Anti-injection: pattern pericolosi da bloccare
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
    
    Pattern: Factory Method (GoF) — centralizza la creazione dell'oggetto di configurazione.
    Vantaggio: un singolo punto di modifica per passare da dev a prod.
    """
    return AppSettings(
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
        ),
    )
