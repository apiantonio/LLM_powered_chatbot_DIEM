"""Configurazione centralizzata del sistema RAG DIEM.

Contiene tutti i dataclass di configurazione dell'applicazione e la
funzione di caricamento da variabili d'ambiente.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse


@dataclass(frozen=True)
class LoggingConfig:
    """Configurazione del sistema di logging centralizzato."""

    level: str = "INFO"
    log_file: Optional[str] = None
    log_to_console: bool = True
    log_format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class IngestionConfig:
    """Parametri per la pipeline di scraping e indicizzazione."""

    html_raw_dir: str = "data/raw/html_samples"
    pdf_links_file: str = "data/raw/html_samples/pdf_links.txt"
    pdf_download_dir: str = "data/raw/pdfs"
    md_static_dir: str = "data/raw/static_md"
    md_chunk_size: int = 800
    md_chunk_overlap: int = 100

    seed_urls: tuple[str, ...] = (
        "https://www.diem.unisa.it",
        "https://corsi.unisa.it/ingegneria-informatica",
        "https://corsi.unisa.it/ingegneria-dell-informazione-per-la-medicina-digitale",
        "https://corsi.unisa.it/ingegneria-informatica-magistrale",
        "https://corsi.unisa.it/electrical-engineering-for-digital-energy",
        "https://corsi.unisa.it/information-Engineering-for-digital-medicine",
        "https://corsi.unisa.it/ingegneria-dell-informazione",
        "https://corsi.unisa.it/photovoltaics",
    )

    max_depth: int = 5
    batch_size: int = 1024
    crawl_delay_seconds: float = 2.0

    cutoff_year: int = 2020
    target_department: str = "300638"
    ignored_extensions: tuple[str, ...] = (
        '.css', '.js', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp',
        '.zip', '.rar', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    )

    html_chunk_size: int = 700
    html_chunk_overlap: int = 50

    persone_html_chunk_size: int = 800
    persone_html_chunk_overlap: int = 100
    offerta_html_chunk_size: int = 700
    offerta_html_chunk_overlap: int = 50
    dipartimento_html_chunk_size: int = 800
    dipartimento_html_chunk_overlap: int = 100

    pdf_parent_chunk_size: int = 3000
    pdf_parent_chunk_overlap: int = 500
    pdf_child_chunk_size: int = 400
    pdf_child_chunk_overlap: int = 50

    pdf_direct_chunk_size: int = 1500
    pdf_direct_chunk_overlap: int = 200

    index_registry_path: str = "data/vectorstore/index_registry.json"

    def get_allowed_domains(self) -> set[str]:
        """Deriva i domini autorizzati a partire dai seed URL."""
        domains = set()
        for url in self.seed_urls:
            parsed = urlparse(url)
            if parsed.netloc:
                domains.add(parsed.netloc)
        domains.add("docenti.unisa.it")
        return domains

    def get_allowed_prefixes(self) -> tuple[str, ...]:
        """Restituisce i prefissi URL autorizzati."""
        return tuple(self.seed_urls)

    def get_collection_html_params(self, collection_name: str) -> tuple[int, int]:
        """Restituisce chunk_size e chunk_overlap specifici per la collezione indicata.

        Args:
            collection_name: Nome della collezione (persone, offerta_formativa, dipartimento).

        Returns:
            Tupla (chunk_size, chunk_overlap) per la collezione richiesta,
            oppure i valori di default se la collezione non e' mappata.
        """
        mapping = {
            "persone": (self.persone_html_chunk_size, self.persone_html_chunk_overlap),
            "offerta_formativa": (self.offerta_html_chunk_size, self.offerta_html_chunk_overlap),
            "dipartimento": (self.dipartimento_html_chunk_size, self.dipartimento_html_chunk_overlap),
        }
        return mapping.get(collection_name, (self.html_chunk_size, self.html_chunk_overlap))


@dataclass(frozen=True)
class CrawlerConfig:
    """Parametri operativi del crawler (thread, fattori di scala)."""

    thread_cpu_factor: float = 0.75
    thread_min_workers: int = 2
    thread_max_workers: int = 16
    write_buffer_size: int = 0

    def compute_max_workers(self) -> int:
        """Calcola il numero di worker in base alle CPU disponibili.

        Il valore e' limitato tra thread_min_workers e thread_max_workers.
        """
        cpu = os.cpu_count() or 4
        computed = max(self.thread_min_workers, int(cpu * self.thread_cpu_factor))
        return min(computed, self.thread_max_workers)


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configurazione del modello di embedding."""

    model_name: str = "BAAI/bge-m3"
    normalize_embeddings: bool = True
    expected_dim: int = 1024


@dataclass(frozen=True)
class VectorStoreConfig:
    """Configurazione del vector store (Chroma) e del parent store."""

    persist_directory: str = "data/vectorstore/chroma"
    collection_name: str = "diem_knowledge_base"
    parent_store_directory: str = "data/vectorstore/parent_docstore"
    search_type: str = "similarity"
    search_k: int = 20
    parent_child_collection_name: str = "offerta_formativa_pdf_childs"


@dataclass(frozen=True)
class RerankerConfig:
    """Configurazione del reranker per il riordinamento dei risultati."""

    score_treshold: float = 0.0
    model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    top_n: int = 5


@dataclass(frozen=True)
class LLMConfig:
    """Configurazione del modello linguistico (provider, credenziali, parametri)."""

    provider: str = "groq"
    model_name: str = "llama-3.3-70b-versatile"
    temperature: float = 0.1
    max_tokens: int = 1024
    huggingface_api_token: Optional[str] = field(default=None)
    openai_api_key: Optional[str] = field(default=None)
    ollama_base_url: str = "http://localhost:11434"
    groq_api_key: Optional[str] = field(default=None)
    groq_chat_api_key: Optional[str] = field(default=None)


@dataclass(frozen=True)
class GuardrailsConfig:
    """Configurazione dei guardrail per la validazione delle richieste."""

    allowed_scope_description: str = (
        "Domande relative al Dipartimento DIEM dell'Universita degli Studi di Salerno: "
        "corsi di laurea, docenti, orari, esami, regolamenti, tesi, borse di studio, "
        "laboratori, servizi dipartimentali, dottorato di ricerca."
    )
    max_agent_iterations: int = 50
    enable_pii_filter: bool = True


@dataclass(frozen=True)
class ObservabilityConfig:
    """Configurazione per l'osservabilita e il debug dell'applicazione."""

    enable_verbose_callbacks: bool = True
    log_retrieved_chunks: bool = True
    log_tool_invocations: bool = True
    max_chunk_preview_chars: int = 200


@dataclass(frozen=True)
class AppSettings:
    """Aggregatore di tutte le sezioni di configurazione dell'applicazione."""

    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    crawler: CrawlerConfig = field(default_factory=CrawlerConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vectorstore: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    guardrails: GuardrailsConfig = field(default_factory=GuardrailsConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_settings() -> AppSettings:
    """Carica e restituisce la configurazione completa dell'applicazione.

    Le variabili d'ambiente sovrascrivono i valori di default. Se il
    pacchetto python-dotenv e' disponibile, il file .env viene caricato
    automaticamente prima della lettura delle variabili.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    return AppSettings(
        ingestion=IngestionConfig(
            html_raw_dir=os.getenv("HTML_RAW_DIR", "data/raw/html_samples"),
            pdf_links_file=os.getenv("PDF_LINKS_FILE", "data/raw/html_samples/pdf_links.txt"),
            pdf_download_dir=os.getenv("PDF_DOWNLOAD_DIR", "data/raw/pdfs"),
            md_static_dir=os.getenv("MD_STATIC_DIR", "data/raw/static_md"),
            cutoff_year=int(os.getenv("CUTOFF_YEAR", "2020")),
            target_department=os.getenv("TARGET_DEPARTMENT", "300638"),
            index_registry_path=os.getenv("INDEX_REGISTRY_PATH", "data/vectorstore_qwen/index_registry.json"),
            max_depth=int(os.getenv("MAX_DEPTH", "5")),
            batch_size=int(os.getenv("BATCH_SIZE", "1024")),
        ),
        crawler=CrawlerConfig(
            thread_cpu_factor=float(os.getenv("CRAWLER_THREAD_FACTOR", "0.75")),
            thread_max_workers=int(os.getenv("CRAWLER_MAX_WORKERS", "16")),
        ),
        llm=LLMConfig(
            provider=os.getenv("LLM_PROVIDER", "groq"),
            model_name=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            huggingface_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            groq_chat_api_key=os.getenv("GROQ_CHAT_API_KEY"),
        ),
        embedding=EmbeddingConfig(
            model_name=os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
        ),
        vectorstore=VectorStoreConfig(
            persist_directory=os.getenv("CHROMA_PERSIST_DIR", "data/vectorstore_qwen/chroma"),
            parent_store_directory=os.getenv("PARENT_STORE_DIR", "data/vectorstore_qwen/parent_docstore"),
        ),
        observability=ObservabilityConfig(
            enable_verbose_callbacks=os.getenv("ENABLE_VERBOSE_CALLBACKS", "true").lower() == "true",
        ),
        reranker=RerankerConfig(
            model_name=os.getenv("RERANKER_MODEL", "Qwen/Qwen3-Reranker-0.6B"),
            score_treshold=float(os.getenv("SCORE_TRESHOLD", "0.0")),
        ),
        guardrails=GuardrailsConfig(
            max_agent_iterations=int(os.getenv("MAX_AGENT_ITER", "50")),
        ),
        logging=LoggingConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE"),
            log_to_console=os.getenv("LOG_TO_CONSOLE", "true").lower() == "true",
            log_format=os.getenv(
                "LOG_FORMAT",
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            ),
            date_format=os.getenv("LOG_DATE_FORMAT", "%Y-%m-%d %H:%M:%S"),
        ),
    )