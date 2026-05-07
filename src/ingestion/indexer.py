"""
Modulo di indicizzazione duale: HTML (diretto) + PDF (Parent-Child via LangChain).

REFACTORING:
  - ELIMINATO: ParentDocStore custom
  - SOSTITUITO CON: create_kv_docstore(LocalFileStore(...)) + ParentDocumentRetriever
  - La risalita Child→Parent è ora gestita nativamente da LangChain

Architettura:
  HTML → HTMLSectionSplitter → RecursiveCharacterTextSplitter(700, 50) → Chroma
  PDF  → ParentDocumentRetriever (LangChain nativo)
         Parent: RCS(3000, 500) → create_kv_docstore(LocalFileStore)
         Child:  RCS(400, 50)   → Chroma (vettorializzato)

KPI Impact:
  - Ingegneria del Software: uso idiomatico del framework, zero reinvenzione.
  - Context Precision: child piccoli → embedding focalizzato.
  - Context Recall: parent ampi restituiti al LLM via risalita nativa.
"""

import os
import re
import logging
import requests
from typing import List, Optional
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, HTMLSectionSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore

from config.settings import AppSettings
from ingestion.registry import IndexRegistry, IndexEntry

logger = logging.getLogger(__name__)


class KnowledgeBaseIndexer:
    """
    Motore di indicizzazione duale con aggiornamento incrementale.
    
    HTML: indicizzazione diretta in Chroma.
    PDF: ParentDocumentRetriever di LangChain (Parent-Child nativo).
    """
    
    def __init__(self, settings: AppSettings):
        self._settings = settings
        
        # --- Embedding condiviso ---
        self._embedding_model = HuggingFaceEmbeddings(
            model_name=settings.embedding.model_name,
            encode_kwargs={"normalize_embeddings": settings.embedding.normalize_embeddings},
        )
        
        # --- Pipeline HTML: Chroma diretto ---
        os.makedirs(settings.vectorstore.persist_directory, exist_ok=True)
        self._html_vectorstore = Chroma(
            collection_name=f"{settings.vectorstore.collection_name}_html",
            embedding_function=self._embedding_model,
            persist_directory=settings.vectorstore.persist_directory,
        )
        
        # --- Pipeline PDF: ParentDocumentRetriever nativo ---
        self._pdf_child_vectorstore = Chroma(
            collection_name=f"{settings.vectorstore.collection_name}_pdf_childs",
            embedding_function=self._embedding_model,
            persist_directory=settings.vectorstore.persist_directory,
        )
        
        os.makedirs(settings.vectorstore.parent_store_directory, exist_ok=True)
        self._pdf_parent_docstore = create_kv_docstore(
            LocalFileStore(settings.vectorstore.parent_store_directory)
        )
        
        self._pdf_retriever = ParentDocumentRetriever(
            vectorstore=self._pdf_child_vectorstore,
            docstore=self._pdf_parent_docstore,
            child_splitter=RecursiveCharacterTextSplitter(
                chunk_size=settings.ingestion.pdf_child_chunk_size,
                chunk_overlap=settings.ingestion.pdf_child_chunk_overlap,
                add_start_index=True,
            ),
            parent_splitter=RecursiveCharacterTextSplitter(
                chunk_size=settings.ingestion.pdf_parent_chunk_size,
                chunk_overlap=settings.ingestion.pdf_parent_chunk_overlap,
            ),
        )
        
        # --- Splitters HTML ---
        self._html_section_splitter = HTMLSectionSplitter(
            headers_to_split_on=[
                ("h1", "Titolo"),
                ("h2", "Sezione"),
                ("h3", "Sottosezione"),
            ]
        )
        self._html_chunk_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.ingestion.html_chunk_size,
            chunk_overlap=settings.ingestion.html_chunk_overlap,
            add_start_index=True,
        )
        
        # --- Registro incrementale ---
        self._registry = IndexRegistry(settings.ingestion.index_registry_path)
        
        logger.info(
            f"Indexer duale — HTML: {settings.ingestion.html_chunk_size}/{settings.ingestion.html_chunk_overlap}, "
            f"PDF Parent: {settings.ingestion.pdf_parent_chunk_size}, "
            f"PDF Child: {settings.ingestion.pdf_child_chunk_size}"
        )

    # ==========================================================
    # PIPELINE HTML
    # ==========================================================

    def index_html_directory(self, directory: Optional[str] = None) -> dict:
        directory = directory or self._settings.ingestion.html_raw_dir
        html_files = list(Path(directory).glob("*.html"))
        
        if not html_files:
            logger.warning(f"Nessun file HTML in {directory}")
            return {"indexed": 0, "skipped": 0, "updated": 0, "orphans_removed": 0}
        
        stats = {"indexed": 0, "skipped": 0, "updated": 0, "orphans_removed": 0}
        current_sources = set()
        
        for filepath in html_files:
            source_id = f"html:{filepath.name}"
            current_sources.add(source_id)
            
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                
                content_hash = IndexRegistry.compute_hash(content)
                existing = self._registry.get(source_id)
                
                if existing and existing.content_hash == content_hash:
                    stats["skipped"] += 1
                    continue
                
                if existing:
                    self._delete_from_vectorstore(self._html_vectorstore, existing.chroma_ids)
                    stats["updated"] += 1
                else:
                    stats["indexed"] += 1
                
                source_url = self._extract_source_url(content)
                chroma_ids = self._chunk_and_index_html(content, source_url, filepath.name)
                
                self._registry.upsert(source_id, IndexEntry(
                    content_hash=content_hash,
                    chroma_ids=chroma_ids,
                ))
                
            except Exception as e:
                logger.error(f"Errore HTML {filepath.name}: {e}")
        
        html_orphans = {
            sid for sid in self._registry.all_source_ids()
            if sid.startswith("html:") and sid not in current_sources
        }
        for orphan_id in html_orphans:
            entry = self._registry.remove(orphan_id)
            if entry:
                self._delete_from_vectorstore(self._html_vectorstore, entry.chroma_ids)
                stats["orphans_removed"] += 1
        
        self._registry.save()
        logger.info(f"HTML indexing: {stats}")
        return stats

    def _chunk_and_index_html(
        self, content: str, source_url: Optional[str], filename: str
    ) -> List[str]:
        section_docs = self._html_section_splitter.split_text(content)
        
        normalized = []
        for doc in section_docs:
            if isinstance(doc, str):
                doc = Document(page_content=doc, metadata={})
            doc.metadata["source_url"] = source_url or filename
            doc.metadata["source_file"] = filename
            doc.metadata["doc_type"] = "html"
            normalized.append(doc)
        
        chunks = self._html_chunk_splitter.split_documents(normalized)
        if not chunks:
            return []
        
        chroma_ids = [f"html_{filename}_{i}" for i in range(len(chunks))]
        self._html_vectorstore.add_documents(chunks, ids=chroma_ids)
        return chroma_ids

    # ==========================================================
    # PIPELINE PDF (ParentDocumentRetriever nativo)
    # ==========================================================

    def index_pdf_list(self, pdf_links_file: Optional[str] = None) -> dict:
        pdf_links_file = pdf_links_file or self._settings.ingestion.pdf_links_file
        download_dir = self._settings.ingestion.pdf_download_dir
        
        if not os.path.exists(pdf_links_file):
            logger.warning(f"File lista PDF non trovato: {pdf_links_file}")
            return {"indexed": 0, "skipped": 0, "updated": 0, "orphans_removed": 0}
        
        with open(pdf_links_file, "r") as f:
            pdf_urls = [line.strip() for line in f if line.strip()]
        
        os.makedirs(download_dir, exist_ok=True)
        stats = {"indexed": 0, "skipped": 0, "updated": 0, "orphans_removed": 0}
        current_sources = set()
        
        for url in pdf_urls:
            source_id = f"pdf:{url}"
            current_sources.add(source_id)
            
            try:
                filename = self._safe_filename(url)
                local_path = os.path.join(download_dir, filename)
                
                if not self._download_if_needed(url, local_path):
                    stats["skipped"] += 1
                    continue
                
                file_hash = IndexRegistry.compute_file_hash(local_path)
                existing = self._registry.get(source_id)
                
                if existing and existing.content_hash == file_hash:
                    stats["skipped"] += 1
                    continue
                
                if existing and existing.chroma_ids:
                    self._delete_from_vectorstore(self._pdf_child_vectorstore, existing.chroma_ids)
                    stats["updated"] += 1
                else:
                    stats["indexed"] += 1
                
                chroma_ids = self._index_single_pdf(local_path, url)
                
                self._registry.upsert(source_id, IndexEntry(
                    content_hash=file_hash,
                    chroma_ids=chroma_ids,
                ))
                
            except Exception as e:
                logger.error(f"Errore PDF {url}: {e}")
        
        pdf_orphans = {
            sid for sid in self._registry.all_source_ids()
            if sid.startswith("pdf:") and sid not in current_sources
        }
        for orphan_id in pdf_orphans:
            entry = self._registry.remove(orphan_id)
            if entry:
                self._delete_from_vectorstore(self._pdf_child_vectorstore, entry.chroma_ids)
                stats["orphans_removed"] += 1
        
        self._registry.save()
        logger.info(f"PDF indexing: {stats}")
        return stats

    def _index_single_pdf(self, local_path: str, source_url: str) -> List[str]:
        """
        Indicizza un PDF tramite ParentDocumentRetriever.add_documents().
        
        Internamente LangChain:
        1. Applica parent_splitter → Parent chunk
        2. Salva Parent nel docstore (LocalFileStore) con UUID
        3. Applica child_splitter su ogni Parent → Child chunk
        4. Inietta doc_id (riferimento Parent) nei metadati Child
        5. Vettorializza e salva Child in Chroma
        """
        loader = PyPDFLoader(local_path)
        pages = loader.load()
        
        for page in pages:
            page.metadata["source_url"] = source_url
            page.metadata["doc_type"] = "pdf"
        
        # Cattura ID prima dell'inserimento per il registro incrementale
        existing_data = self._pdf_child_vectorstore.get()
        ids_before = set(existing_data["ids"]) if existing_data["ids"] else set()
        
        # Delega interamente a LangChain
        self._pdf_retriever.add_documents(pages)
        
        ids_after = set(self._pdf_child_vectorstore.get()["ids"])
        new_ids = list(ids_after - ids_before)
        
        logger.debug(
            f"PDF '{os.path.basename(local_path)}': "
            f"{len(new_ids)} child chunks via ParentDocumentRetriever"
        )
        return new_ids

    # ==========================================================
    # RETRIEVER ACCESSORS
    # ==========================================================

    def get_html_retriever(self, search_type: Optional[str] = None, k: Optional[int] = None):
        """Retriever per HTML (chunk diretti)."""
        return self._html_vectorstore.as_retriever(
            search_type=search_type or self._settings.vectorstore.search_type,
            search_kwargs={"k": k or self._settings.vectorstore.search_k},
        )
    
    def get_pdf_retriever(self, k: Optional[int] = None):
        """
        Retriever per PDF (Parent-Child nativo LangChain).
        invoke() restituisce direttamente i Parent.
        """
        self._pdf_retriever.search_kwargs = {
            "k": k or self._settings.vectorstore.search_k
        }
        return self._pdf_retriever

    # ==========================================================
    # UTILITÀ
    # ==========================================================

    @staticmethod
    def _delete_from_vectorstore(vectorstore: Chroma, ids: List[str]) -> None:
        if not ids:
            return
        try:
            vectorstore.delete(ids=ids)
        except Exception as e:
            logger.warning(f"Errore eliminazione da Chroma: {e}")

    @staticmethod
    def _download_if_needed(url: str, local_path: str) -> bool:
        if os.path.exists(local_path):
            return True
        try:
            response = requests.get(url, timeout=60)
            if response.ok:
                with open(local_path, "wb") as f:
                    f.write(response.content)
                return True
            return False
        except Exception as e:
            logger.error(f"Errore download {url}: {e}")
            return False
    
    @staticmethod
    def _extract_source_url(html_content: str) -> Optional[str]:
        match = re.search(r"<!--\s*SOURCE:\s*(https?://[^\s]+)\s*-->", html_content)
        return match.group(1).strip() if match else None
    
    @staticmethod
    def _safe_filename(url: str) -> str:
        name = re.sub(r'[<>:"/\\|?*]', '_', url.replace("https://", "").replace("http://", ""))
        return name[:120]