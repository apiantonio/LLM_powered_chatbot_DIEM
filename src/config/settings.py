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

    html_rule_names: tuple[str, ...] = (
        "publication_tip",
        "exact_publications",
        "department_bandi",
        "calendar",
        "news",
        "404",
        "nocontent",
        "empty_body",
        "filename",
    )

    pdf_rule_names: tuple[str, ...] = (
        "domain_whitelist",
        "semantic_trap",
        "obsolete_year",
        "english_pdf",
    )

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
        """Restituisce chunk_size e chunk_overlap specifici per la collezione indicata."""
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
        """Calcola il numero di worker in base alle CPU disponibili."""
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
    log_top_n: int = 5


@dataclass(frozen=True)
class QueryOptimizerConfig:
    """Configurazione del QueryOptimizer per riscrittura e espansione query."""

    rewrite_max_context_chars: int = 300
    rewrite_max_expansion_factor: int = 8
    multi_query_max_variants: int = 3
    dedup_hash_chars: int = 200

    rewrite_system_prompt: str = (
        "You are a coreference resolver for an Italian university Q&A system. "
        "You receive the last interaction (user question + assistant answer) and a new query. "
        "Replace any pronouns or implicit references (lui, lei, suo, suoi, questo, quella, "
        "lì, ci, ne, quali sono i suoi, ecc.) with the explicit entity from the last interaction. "
        "The entity can be anything: a person, a course, a classroom, a lab, a scholarship, etc. "
        "If the new query is already self-contained, return it unchanged. "
        "Do not answer the question. Output only the rewritten query.\n\n"
        "Last Q: \"Chi è il prof. Rossi?\" Last A: \"Il prof. Rossi insegna...\" → "
        "New query: \"Qual è il suo ricevimento?\" → "
        "Output: Qual è il ricevimento del prof. Rossi?\n\n"
        "Last Q: \"Parlami del corso di Informatica triennale\" Last A: \"Il corso prevede...\" → "
        "New query: \"Quali sono i suoi contenuti?\" → "
        "Output: Quali sono i contenuti del corso di Informatica triennale?\n\n"
        "Last Q: \"Dove si trova l'aula 10?\" Last A: \"L'aula 10 è nel campus...\" → "
        "New query: \"Quanti posti ha?\" → "
        "Output: Quanti posti ha l'aula 10?\n\n"
        "Last Q: \"Parlami di Ingegneria Informatica\" Last A: \"...\" → "
        "New query: \"Dove si trova l'aula 10?\" → "
        "Output: Dove si trova l'aula 10?"
    )

    rewrite_human_template: str = (
        "Last Q: \"{last_user_query}\"\n"
        "Last A: \"{last_assistant_answer}\"\n\n"
        "New query: \"{question}\""
    )

    multi_query_system_prompt: str = (
        "You generate Italian rephrasings of a question for semantic search in the knowledge base "
        "of the DIEM department (Dipartimento di Ingegneria dell'Informazione ed Elettrica e "
        "Matematica Applicata) at Universita degli Studi di Salerno.\n\n"
        "STRICT RULES:\n"
        "1. Generate exactly 3 rephrasings in Italian.\n"
        "2. Preserve ALL proper nouns, acronyms (DIEM, UniSA, CFU), professor names, "
        "course names, and the original intent EXACTLY.\n"
        "3. Do NOT change, expand, or guess institutional names. "
        "'DIEM' stays 'DIEM', NOT 'Dipartimento di Scienze dell'Informazione'. "
        "Do NOT add university names like 'Bologna', 'Roma', 'Milano'.\n"
        "4. Only vary the sentence structure and synonyms, not the entities.\n"
        "5. Output one variant per line, no numbering, no explanations, no preamble.\n\n"
        "Example input: 'Quali sono i corsi di laurea triennale offerti dal DIEM?'\n"
        "Example output:\n"
        "Quali corsi di laurea triennale sono disponibili presso il DIEM?\n"
        "Elenco dei corsi triennali del DIEM\n"
        "Offerta formativa triennale del DIEM"
    )

    multi_query_human_template: str = "{question}"


@dataclass(frozen=True)
class MemoryConfig:
    """Configurazione della memoria conversazionale SmartConversationMemory."""

    max_turns: int = 10
    similarity_threshold: float = 0.55
    max_token_limit: int = 1500
    history_preview_chars: int = 150
    langchain_history_max_chars: int = 300
    summary_human_prefix: str = "Utente"
    summary_ai_prefix: str = "Assistente"


@dataclass(frozen=True)
class LLMConfig:
    """Configurazione del modello linguistico (provider, credenziali, parametri)."""

    provider: str = "ollama"
    model_name: str = "nemotron-3-super:cloud"
    temperature: float = 0.0
    max_tokens: int = 1024
    ollama_base_url: str = "http://localhost:11434"

    fallback_model: str = "qwen2.5"
    fallback_base_url: str = "http://localhost:11434"

    rewriter_provider: str = "groq"
    rewriter_model: str = "llama-3.3-70b-versatile"
    rewriter_max_tokens: int = 256
    rewriter_temperature: float = 0.0
    rewriter_hf_temperature: float = 0.01
    groq_rewriter_api_key: Optional[str] = field(default=None)

    groq_guardrails_api_key: Optional[str] = field(default=None)
    guardrails_model: str = "llama-3.3-70b-versatile"
    guardrails_max_tokens: int = 10

    groq_chat_api_key: Optional[str] = field(default=None)

    groq_validation_model: str = "llama-3.3-70b-versatile"
    groq_validation_max_tokens: int = 5

    huggingface_api_token: Optional[str] = field(default=None)
    openai_api_key: Optional[str] = field(default=None)


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
    max_tool_calls: int = 3

    blocked_input_message: str = (
        "Mi dispiace, non posso elaborare questa richiesta. "
        "Sono l'assistente virtuale del Dipartimento DIEM "
        "dell'Universita degli Studi di Salerno. "
        "Posso aiutarti con informazioni su corsi, docenti, esami, "
        "regolamenti, laboratori e servizi universitari."
    )

    blocked_output_message: str = (
        "Mi scuso, non posso fornire questa risposta per motivi di policy. "
        "Posso aiutarti con informazioni sul Dipartimento DIEM."
    )

    meta_fallback_message: str = (
        "Ciao! Sono l'assistente virtuale del Dipartimento DIEM "
        "dell'Universita degli Studi di Salerno. "
        "Posso aiutarti con informazioni su corsi, docenti, esami, "
        "regolamenti, laboratori e servizi universitari. "
        "Come posso esserti utile?"
    )

    error_generic_message: str = (
        "Mi scuso, si e' verificato un errore. "
        "Riprova tra qualche istante."
    )

    error_loop_message: str = (
        "Mi scuso, ho riscontrato difficolta nell'elaborare la tua "
        "domanda. Prova a riformularla in modo piu specifico."
    )

    error_tool_limit_message: str = (
        "Mi scuso, non sono riuscito a trovare informazioni sufficienti "
        "per rispondere alla tua domanda. Ti consiglio di consultare "
        "il sito web del DIEM o contattare la segreteria."
    )

    error_empty_response_message: str = (
        "Mi scuso, ho riscontrato un problema nell'elaborazione "
        "della risposta. Per favore, riprova o riformula la domanda."
    )

    synthesis_system_prompt: str = (
        "Sei l'assistente virtuale del DIEM, Universita degli Studi di Salerno. "
        "Ti vengono forniti dei documenti recuperati dalla knowledge base. "
        "Rispondi alla domanda dell'utente basandoti ESCLUSIVAMENTE su questi documenti. "
        "Se i documenti non contengono l'informazione richiesta, dillo chiaramente. "
        "Rispondi nella stessa lingua usata dall'utente."
    )


@dataclass(frozen=True)
class ObservabilityConfig:
    """Configurazione per l'osservabilita e il debug dell'applicazione."""

    enable_verbose_callbacks: bool = True
    log_retrieved_chunks: bool = True
    log_tool_invocations: bool = True
    max_chunk_preview_chars: int = 200

    report_path: str = "data/reports/ingestion_report.json"
    interaction_log_dir: str = "logs/interactions"
    log_query_preview_chars: int = 80


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
    query_optimizer: QueryOptimizerConfig = field(default_factory=QueryOptimizerConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)


def load_settings() -> AppSettings:
    """Carica e restituisce la configurazione completa dell'applicazione."""
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
            index_registry_path=os.getenv("INDEX_REGISTRY_PATH", "data/vectorstore/index_registry.json"),
            max_depth=int(os.getenv("MAX_DEPTH", "5")),
            batch_size=int(os.getenv("BATCH_SIZE", "1024")),
        ),
        crawler=CrawlerConfig(
            thread_cpu_factor=float(os.getenv("CRAWLER_THREAD_FACTOR", "0.75")),
            thread_max_workers=int(os.getenv("CRAWLER_MAX_WORKERS", "16")),
        ),
        llm=LLMConfig(
            provider=os.getenv("LLM_PROVIDER", "ollama"),
            model_name=os.getenv("LLM_MODEL", "nemotron-3-super:cloud"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            fallback_model=os.getenv("FALLBACK_MODEL", "qwen2.5"),
            fallback_base_url=os.getenv("FALLBACK_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")),
            rewriter_provider=os.getenv("REWRITER_PROVIDER", "groq"),
            rewriter_model=os.getenv("REWRITER_MODEL", "llama-3.3-70b-versatile"),
            rewriter_max_tokens=int(os.getenv("REWRITER_MAX_TOKENS", "256")),
            rewriter_temperature=float(os.getenv("REWRITER_TEMPERATURE", "0.0")),
            rewriter_hf_temperature=float(os.getenv("REWRITER_HF_TEMPERATURE", "0.01")),
            groq_rewriter_api_key=os.getenv("GROQ_REWRITER_API_KEY"),
            groq_guardrails_api_key=os.getenv("GROQ_GUARDRAILS_API_KEY"),
            guardrails_model=os.getenv("GUARDRAILS_MODEL", "llama-3.3-70b-versatile"),
            guardrails_max_tokens=int(os.getenv("GUARDRAILS_MAX_TOKENS", "10")),
            groq_chat_api_key=os.getenv("GROQ_CHAT_API_KEY"),
            groq_validation_model=os.getenv("GROQ_VALIDATION_MODEL", "llama-3.3-70b-versatile"),
            groq_validation_max_tokens=int(os.getenv("GROQ_VALIDATION_MAX_TOKENS", "5")),
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
        observability=ObservabilityConfig(
            enable_verbose_callbacks=os.getenv("ENABLE_VERBOSE_CALLBACKS", "true").lower() == "true",
            report_path=os.getenv("INGESTION_REPORT_PATH", "data/reports/ingestion_report.json"),
            interaction_log_dir=os.getenv("INTERACTION_LOG_DIR", "logs/interactions"),
            log_query_preview_chars=int(os.getenv("LOG_QUERY_PREVIEW_CHARS", "80")),
        ),
        reranker=RerankerConfig(
            model_name=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2"),
            score_treshold=float(os.getenv("SCORE_TRESHOLD", "0.0")),
            top_n=int(os.getenv("RERANKER_TOP_N", "5")),
            log_top_n=int(os.getenv("RERANKER_LOG_TOP_N", "5")),
        ),
        guardrails=GuardrailsConfig(
            max_agent_iterations=int(os.getenv("MAX_AGENT_ITER", "50")),
            max_tool_calls=int(os.getenv("MAX_TOOL_CALLS", "3")),
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
        query_optimizer=QueryOptimizerConfig(
            rewrite_max_context_chars=int(os.getenv("REWRITE_MAX_CONTEXT_CHARS", "300")),
            rewrite_max_expansion_factor=int(os.getenv("REWRITE_MAX_EXPANSION_FACTOR", "8")),
            multi_query_max_variants=int(os.getenv("MULTI_QUERY_MAX_VARIANTS", "3")),
            dedup_hash_chars=int(os.getenv("DEDUP_HASH_CHARS", "200")),
        ),
        memory=MemoryConfig(
            max_turns=int(os.getenv("MEMORY_MAX_TURNS", "10")),
            similarity_threshold=float(os.getenv("MEMORY_SIMILARITY_THRESHOLD", "0.55")),
            max_token_limit=int(os.getenv("MEMORY_MAX_TOKEN_LIMIT", "1500")),
            history_preview_chars=int(os.getenv("MEMORY_HISTORY_PREVIEW_CHARS", "150")),
            langchain_history_max_chars=int(os.getenv("MEMORY_LANGCHAIN_HISTORY_MAX_CHARS", "300")),
            summary_human_prefix=os.getenv("MEMORY_SUMMARY_HUMAN_PREFIX", "Utente"),
            summary_ai_prefix=os.getenv("MEMORY_SUMMARY_AI_PREFIX", "Assistente"),
        ),
    )