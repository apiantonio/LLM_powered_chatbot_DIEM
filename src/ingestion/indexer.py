"""
Modulo di indicizzazione: Chunking → Embedding → ChromaDB.

Responsabilità: Trasformare i documenti puliti (HTML/PDF) in vettori
indicizzati nel Vector Store, pronti per il retrieval.

Pattern: 
- Template Method per il flusso Load → Split → Embed → Store.
- Dependency Injection per embedding model e vector store.

KPI Impact: 
- Context Precision: chunk più piccoli e semanticamente coesi → meno rumore.
- Context Recall: overlap garantisce che i confini concettuali non vengano recisi.
"""

import os
import logging
from typing import List, Optional
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, HTMLSectionSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.settings import AppSettings

logger = logging.getLogger(__name__)


class KnowledgeBaseIndexer:
    """
    Motore di indicizzazione che costruisce e aggiorna il Vector Store.
    
    Responsabilità unica: prendere documenti grezzi e trasformarli in vettori ricercabili.
    Non si occupa di scraping, pulizia o retrieval.
    """
    
    def __init__(self, settings: AppSettings):
        self._settings = settings
        
        # Inizializzazione lazy del modello di embedding
        self._embedding_model = HuggingFaceEmbeddings(
            model_name=settings.embedding.model_name,
            encode_kwargs={"normalize_embeddings": settings.embedding.normalize_embeddings},
        )
        
        # Inizializzazione del Vector Store con persistenza
        os.makedirs(settings.vectorstore.persist_directory, exist_ok=True)
        self._vectorstore = Chroma(
            collection_name=settings.vectorstore.collection_name,
            embedding_function=self._embedding_model,
            persist_directory=settings.vectorstore.persist_directory,
        )
        
        # Text Splitters
        self._html_splitter = HTMLSectionSplitter(
            headers_to_split_on=[
                ("h1", "Titolo"),
                ("h2", "Sezione"),
                ("h3", "Sottosezione"),
            ]
        )
        
        self._recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.ingestion.chunk_size,
            chunk_overlap=settings.ingestion.chunk_overlap,
            add_start_index=True,
        )
        
        logger.info(
            f"Indexer inizializzato — Embedding: {settings.embedding.model_name}, "
            f"Chunk: {settings.ingestion.chunk_size}/{settings.ingestion.chunk_overlap}"
        )
    
    def index_html_directory(self, directory: str) -> int:
        """
        Indicizza tutti i file HTML puliti da una directory.
        
        Flow: Load HTML → Split per sezioni → Split ricorsivo → Embed → Store
        
        Returns:
            Numero di chunk indicizzati.
        """
        html_files = list(Path(directory).glob("*.html"))
        if not html_files:
            logger.warning(f"Nessun file HTML trovato in {directory}")
            return 0
        
        all_chunks: List[Document] = []
        
        for filepath in html_files:
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                
                # Estrazione URL sorgente dai metadati del crawler
                source_url = self._extract_source_url(content)
                
                # Step 1: Split strutturale per sezioni HTML
                section_docs = self._html_splitter.split_text(content)
                
                # Step 2: Split ricorsivo per dimensione
                for doc in section_docs:
                    if isinstance(doc, str):
                        doc = Document(page_content=doc, metadata={})
                    
                    doc.metadata["source_url"] = source_url or filepath.name
                    doc.metadata["source_file"] = filepath.name
                
                chunks = self._recursive_splitter.split_documents(
                    [d if isinstance(d, Document) else Document(page_content=str(d), metadata={}) 
                     for d in section_docs]
                )
                
                all_chunks.extend(chunks)
                
            except Exception as e:
                logger.error(f"Errore indicizzazione {filepath.name}: {e}")
        
        # Step 3: Batch insert nel Vector Store
        if all_chunks:
            self._vectorstore.add_documents(all_chunks)
            logger.info(f"Indicizzati {len(all_chunks)} chunk da {len(html_files)} file HTML")
        
        return len(all_chunks)
    
    def index_pdf_list(self, pdf_links_file: str, download_dir: str = "data/raw/pdfs") -> int:
        """
        Scarica e indicizza i PDF dalla lista di link.
        
        Flow: Download → Load PDF → Split ricorsivo → Embed → Store
        """
        if not os.path.exists(pdf_links_file):
            logger.warning(f"File lista PDF non trovato: {pdf_links_file}")
            return 0
        
        with open(pdf_links_file, "r") as f:
            pdf_urls = [line.strip() for line in f if line.strip()]
        
        os.makedirs(download_dir, exist_ok=True)
        all_chunks: List[Document] = []
        
        for url in pdf_urls:
            try:
                filename = url.split("/")[-1]
                local_path = os.path.join(download_dir, filename)
                
                # Download se non esiste già (incrementale)
                if not os.path.exists(local_path):
                    import requests
                    response = requests.get(url, timeout=30)
                    if response.ok:
                        with open(local_path, "wb") as pf:
                            pf.write(response.content)
                    else:
                        continue
                
                # Load e split
                loader = PyPDFLoader(local_path)
                pages = loader.load()
                
                for page in pages:
                    page.metadata["source_url"] = url
                
                chunks = self._recursive_splitter.split_documents(pages)
                all_chunks.extend(chunks)
                
            except Exception as e:
                logger.error(f"Errore processing PDF {url}: {e}")
        
        if all_chunks:
            self._vectorstore.add_documents(all_chunks)
            logger.info(f"Indicizzati {len(all_chunks)} chunk da {len(pdf_urls)} PDF")
        
        return len(all_chunks)
    
    def get_retriever(self, search_type: Optional[str] = None, k: Optional[int] = None):
        """
        Restituisce un LangChain Retriever configurato dal settings.
        
        Il retriever è il punto di connessione tra Vector Store e RAG chain.
        """
        return self._vectorstore.as_retriever(
            search_type=search_type or self._settings.vectorstore.search_type,
            search_kwargs={"k": k or self._settings.vectorstore.search_k},
        )
    
    @property
    def vectorstore(self) -> Chroma:
        """Accesso diretto al vector store (per query manuali o debug)."""
        return self._vectorstore
    
    @staticmethod
    def _extract_source_url(html_content: str) -> Optional[str]:
        """Estrae l'URL sorgente dai metadati inseriti dal crawler."""
        import re
        match = re.search(r"<!--\s*SOURCE:\s*(https?://[^\s]+)\s*-->", html_content)
        return match.group(1).strip() if match else None
