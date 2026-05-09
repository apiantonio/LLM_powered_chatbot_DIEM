"""
ingestion/indexer.py — Refactoring multi-collection.

Architettura:
  4 Collection Chroma, ciascuna con strategia di chunking dedicata.
  DocumentRouter assegna ogni documento alla collection corretta.
  Parent-Child solo per regolamenti/piani di studio (Collection 2).
  Chunking diretto per bandi (Collection 3) — risolve il problema
  dei falsi positivi semantici.
"""

import os
import re
import logging
from typing import List, Optional, Dict
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
from ingestion.router import DocumentRouter, CollectionTarget

logger = logging.getLogger(__name__)


class KnowledgeBaseIndexer:
    """
    Motore di indicizzazione multi-collection con routing semantico.
    
    Gestisce 4 collection Chroma con strategie di chunking differenziate
    e un ParentDocumentRetriever dedicato ai soli PDF che lo necessitano.
    """

    def __init__(self, settings: AppSettings):
        self._settings = settings

        # --- Embedding condiviso (unico modello per tutte le collection) ---
        self._embedding_model = HuggingFaceEmbeddings(
            model_name=settings.embedding.model_name,
            encode_kwargs={
                "normalize_embeddings": settings.embedding.normalize_embeddings
            },
        )

        # --- Istanziazione delle 4 collection Chroma ---
        persist_dir = settings.vectorstore.persist_directory
        os.makedirs(persist_dir, exist_ok=True)

        self._collections: Dict[CollectionTarget, Chroma] = {}
        for target in CollectionTarget:
            self._collections[target] = Chroma(
                collection_name=target.value,
                embedding_function=self._embedding_model,
                persist_directory=persist_dir,
            )

        # --- Parent-Child SOLO per regolamenti/piani di studio ---
        parent_dir = settings.vectorstore.parent_store_directory
        os.makedirs(parent_dir, exist_ok=True)

        self._pc_child_vectorstore = Chroma(
            collection_name="offerta_formativa_pdf_childs",
            embedding_function=self._embedding_model,
            persist_directory=persist_dir,
        )
        self._pc_parent_docstore = create_kv_docstore(
            LocalFileStore(parent_dir)
        )
        self._parent_child_retriever = ParentDocumentRetriever(
            vectorstore=self._pc_child_vectorstore,
            docstore=self._pc_parent_docstore,
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

        # --- Splitters per tipo ---
        self._html_section_splitter = HTMLSectionSplitter(
            headers_to_split_on=[
                ("h1", "Titolo"),
                ("h2", "Sezione"),
                ("h3", "Sottosezione"),
            ]
        )
        self._html_splitters = {
            CollectionTarget.DOCENTI_DIDATTICA: RecursiveCharacterTextSplitter(
                chunk_size=800, chunk_overlap=100, add_start_index=True
            ),
            CollectionTarget.OFFERTA_FORMATIVA: RecursiveCharacterTextSplitter(
                chunk_size=700, chunk_overlap=50, add_start_index=True
            ),
            CollectionTarget.BANDI_AMMINISTRAZIONE: RecursiveCharacterTextSplitter(
                chunk_size=700, chunk_overlap=50, add_start_index=True
            ),
            CollectionTarget.DIPARTIMENTO_RICERCA: RecursiveCharacterTextSplitter(
                chunk_size=800, chunk_overlap=100, add_start_index=True
            ),
        }
        # Splitter diretto per PDF che NON usano Parent-Child
        self._pdf_direct_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=200, add_start_index=True
        )

        # --- Registro incrementale ---
        self._registry = IndexRegistry(settings.ingestion.index_registry_path)

        logger.info(
            f"Indexer multi-collection inizializzato: "
            f"{[t.value for t in CollectionTarget]}"
        )

    # ==========================================================
    # PIPELINE HTML (con routing)
    # ==========================================================

    def index_html_directory(self, directory: Optional[str] = None) -> dict:
        directory = directory or self._settings.ingestion.html_raw_dir
        html_files = list(Path(directory).glob("*.html"))

        if not html_files:
            logger.warning(f"Nessun file HTML in {directory}")
            return {"indexed": 0, "skipped": 0, "updated": 0, "orphans_removed": 0}

        stats = {"indexed": 0, "skipped": 0, "updated": 0, "orphans_removed": 0}
        current_sources = set()
        routing_stats: Dict[str, int] = {t.value: 0 for t in CollectionTarget}

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

                # --- ROUTING: determina la collection target ---
                source_url = self._extract_source_url(content) or filepath.name
                collection_target = DocumentRouter.route_html(source_url)
                vectorstore = self._collections[collection_target]

                # Cleanup vecchi chunk se update
                if existing:
                    self._delete_from_vectorstore(
                        self._collections.get(
                            CollectionTarget(existing.collection_name),
                            vectorstore
                        ) if hasattr(existing, 'collection_name') else vectorstore,
                        existing.chroma_ids
                    )
                    stats["updated"] += 1
                else:
                    stats["indexed"] += 1

                # Metadati arricchiti
                extra_meta = DocumentRouter.extract_metadata(
                    source_url, collection_target
                )

                # Chunking e indicizzazione
                chroma_ids = self._chunk_and_index_html(
                    content, source_url, filepath.name,
                    vectorstore, collection_target, extra_meta
                )

                self._registry.upsert(source_id, IndexEntry(
                    content_hash=content_hash,
                    chroma_ids=chroma_ids,
                    collection_name=collection_target.value,
                ))
                routing_stats[collection_target.value] += 1

            except Exception as e:
                logger.error(f"Errore HTML {filepath.name}: {e}")

        # Pulizia orfani
        html_orphans = {
            sid for sid in self._registry.all_source_ids()
            if sid.startswith("html:") and sid not in current_sources
        }
        for orphan_id in html_orphans:
            entry = self._registry.remove(orphan_id)
            if entry:
                target_vs = self._collections.get(
                    CollectionTarget(entry.collection_name)
                    if hasattr(entry, 'collection_name') else None,
                    self._collections[CollectionTarget.DIPARTIMENTO_RICERCA]
                )
                self._delete_from_vectorstore(target_vs, entry.chroma_ids)
                stats["orphans_removed"] += 1

        self._registry.save()
        logger.info(f"HTML indexing: {stats}")
        logger.info(f"HTML routing: {routing_stats}")
        return stats

    def _chunk_and_index_html(
        self,
        content: str,
        source_url: str,
        filename: str,
        vectorstore: Chroma,
        collection: CollectionTarget,
        extra_metadata: dict,
    ) -> List[str]:
        """Chunk e indicizza un documento HTML nella collection target."""
        section_docs = self._html_section_splitter.split_text(content)

        normalized = []
        for doc in section_docs:
            if isinstance(doc, str):
                doc = Document(page_content=doc, metadata={})
            doc.metadata.update({
                "source_url": source_url,
                "source_file": filename,
                "doc_type": "html",
                **extra_metadata,
            })
            normalized.append(doc)

        splitter = self._html_splitters[collection]
        chunks = splitter.split_documents(normalized)
        if not chunks:
            return []

        chroma_ids = [
            f"{collection.value}_html_{filename}_{i}"
            for i in range(len(chunks))
        ]
        vectorstore.add_documents(chunks, ids=chroma_ids)
        return chroma_ids

    # ==========================================================
    # PIPELINE PDF (con routing + chunking differenziato)
    # ==========================================================

    def index_pdf_list(self, pdf_links_file: Optional[str] = None) -> dict:
        pdf_links_file = pdf_links_file or self._settings.ingestion.pdf_links_file
        download_dir = self._settings.ingestion.pdf_download_dir

        if not os.path.exists(pdf_links_file):
            logger.warning(f"File lista PDF non trovato: {pdf_links_file}")
            return {"indexed": 0, "skipped": 0, "updated": 0}

        with open(pdf_links_file, "r") as f:
            pdf_urls = [line.strip() for line in f if line.strip()]

        os.makedirs(download_dir, exist_ok=True)
        stats = {"indexed": 0, "skipped": 0, "updated": 0}
        routing_stats = {t.value: 0 for t in CollectionTarget}

        for url in pdf_urls:
            source_id = f"pdf:{url}"

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

                # --- ROUTING ---
                collection_target = DocumentRouter.route_pdf(url)
                use_pc = DocumentRouter.use_parent_child(collection_target, url)

                # Cleanup vecchi chunk
                if existing and existing.chroma_ids:
                    target_vs = (
                        self._pc_child_vectorstore if use_pc
                        else self._collections[collection_target]
                    )
                    self._delete_from_vectorstore(target_vs, existing.chroma_ids)
                    stats["updated"] += 1
                else:
                    stats["indexed"] += 1

                # Metadati arricchiti
                extra_meta = DocumentRouter.extract_metadata(url, collection_target)

                # --- CHUNKING DIFFERENZIATO ---
                if use_pc:
                    # Parent-Child solo per regolamenti/piani di studio
                    chroma_ids = self._index_pdf_parent_child(
                        local_path, url, extra_meta
                    )
                else:
                    # Chunking diretto per bandi e altri PDF
                    chroma_ids = self._index_pdf_direct(
                        local_path, url, collection_target, extra_meta
                    )

                self._registry.upsert(source_id, IndexEntry(
                    content_hash=file_hash,
                    chroma_ids=chroma_ids,
                    collection_name=collection_target.value,
                ))
                routing_stats[collection_target.value] += 1

            except Exception as e:
                logger.error(f"Errore PDF {url}: {e}")

        self._registry.save()
        logger.info(f"PDF indexing: {stats}")
        logger.info(f"PDF routing: {routing_stats}")
        return stats

    def _index_pdf_parent_child(
        self, local_path: str, source_url: str, extra_meta: dict
    ) -> List[str]:
        """
        Parent-Child per regolamenti e piani di studio.
        Identico al vecchio comportamento, ma limitato ai documenti che
        effettivamente beneficiano di questo pattern.
        """
        loader = PyPDFLoader(local_path)
        pages = loader.load()

        for page in pages:
            page.metadata.update({
                "source_url": source_url,
                "doc_type": "pdf",
                **extra_meta,
            })

        ids_before = set(self._pc_child_vectorstore.get()["ids"] or [])
        self._parent_child_retriever.add_documents(pages)
        ids_after = set(self._pc_child_vectorstore.get()["ids"])

        new_ids = list(ids_after - ids_before)
        logger.debug(
            f"PDF Parent-Child '{os.path.basename(local_path)}': "
            f"{len(new_ids)} child chunks"
        )
        return new_ids

    def _index_pdf_direct(
        self,
        local_path: str,
        source_url: str,
        collection: CollectionTarget,
        extra_meta: dict,
    ) -> List[str]:
        """
        Chunking diretto per bandi e PDF che NON necessitano di Parent-Child.
        
        Usa chunk ampi (1500 chars) per catturare il contesto del bando
        senza la risalita al Parent che causa i falsi positivi.
        """
        loader = PyPDFLoader(local_path)
        pages = loader.load()

        for page in pages:
            page.metadata.update({
                "source_url": source_url,
                "doc_type": "pdf",
                **extra_meta,
            })

        chunks = self._pdf_direct_splitter.split_documents(pages)
        if not chunks:
            return []

        vectorstore = self._collections[collection]
        filename = os.path.basename(local_path)
        chroma_ids = [
            f"{collection.value}_pdf_{filename}_{i}"
            for i in range(len(chunks))
        ]
        vectorstore.add_documents(chunks, ids=chroma_ids)

        logger.debug(
            f"PDF diretto '{filename}': "
            f"{len(chunks)} chunks → {collection.value}"
        )
        return chroma_ids

    # ==========================================================
    # RETRIEVER ACCESSORS (uno per collection + merge)
    # ==========================================================

    def get_collection_retriever(
        self,
        collection: CollectionTarget,
        search_type: Optional[str] = None,
        k: Optional[int] = None,
    ):
        """Retriever per una singola collection (chunk diretti)."""
        return self._collections[collection].as_retriever(
            search_type=search_type or self._settings.vectorstore.search_type,
            search_kwargs={"k": k or self._settings.vectorstore.search_k},
        )

    def get_parent_child_retriever(self, k: Optional[int] = None):
        """Retriever Parent-Child per regolamenti/piani di studio."""
        self._parent_child_retriever.search_kwargs = {
            "k": k or self._settings.vectorstore.search_k
        }
        return self._parent_child_retriever

    def get_all_retrievers(self):
        """Restituisce tutti i retriever per query cross-collection."""
        retrievers = {}
        for target in CollectionTarget:
            retrievers[target] = self.get_collection_retriever(target)
        retrievers["parent_child"] = self.get_parent_child_retriever()
        return retrievers

    # ==========================================================
    # UTILITÀ (invariate)
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
            import requests
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
        match = re.search(
            r"<!--\s*SOURCE:\s*(https?://[^\s]+)\s*-->", html_content
        )
        return match.group(1).strip() if match else None

    @staticmethod
    def _safe_filename(url: str) -> str:
        name = re.sub(
            r'[<>:"/\\|?*]', '_',
            url.replace("https://", "").replace("http://", "")
        )
        return name[:120]