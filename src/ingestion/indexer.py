"""
ingestion/indexer.py — Motore di indicizzazione multi-collection.

Architettura:
  4 Collection Chroma, ciascuna con strategia di chunking dedicata.
  DocumentRouter assegna ogni documento alla collection corretta.
  Parent-Child solo per regolamenti/piani di studio (Collection 2).
  Chunking diretto per bandi (Collection 3) — risolve il problema
  dei falsi positivi semantici.

Coerenza con gli altri moduli:
  - config/settings.py: usa IngestionConfig.get_collection_html_params() per chunk sizes.
                         Usa pdf_direct_chunk_size/overlap per bandi.
                         Usa parent_child_collection_name per la collection PC.
  - ingestion/router.py: usa DocumentRouter per routing e metadati.
  - ingestion/registry.py: usa IndexEntry.collection_name per cleanup mirato.
  - retrieval/engine.py: espone get_collection_retriever(CollectionTarget) e
                          get_parent_child_retriever() per il wiring.
"""

import os
import re
import logging
import requests
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
            logger.info(f"  Collection Chroma inizializzata: {target.value}")

        # --- Parent-Child SOLO per regolamenti/piani di studio ---
        parent_dir = settings.vectorstore.parent_store_directory
        os.makedirs(parent_dir, exist_ok=True)

        # Nome collection PC letto da settings (Single Source of Truth)
        self._pc_child_vectorstore = Chroma(
            collection_name=settings.vectorstore.parent_child_collection_name,
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
        logger.info(
            f"  Parent-Child Retriever: child={settings.ingestion.pdf_child_chunk_size}, "
            f"parent={settings.ingestion.pdf_parent_chunk_size}"
        )

        # --- Splitters HTML per-collection ---
        # LEGGE I PARAMETRI DA SETTINGS (non hardcodati!)
        self._html_section_splitter = HTMLSectionSplitter(
            headers_to_split_on=[
                ("h1", "Titolo"),
                ("h2", "Sezione"),
                ("h3", "Sottosezione"),
            ]
        )
        
        self._html_splitters: Dict[CollectionTarget, RecursiveCharacterTextSplitter] = {}
        for target in CollectionTarget:
            chunk_size, chunk_overlap = settings.ingestion.get_collection_html_params(
                target.value
            )
            self._html_splitters[target] = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                add_start_index=True,
            )
            logger.info(
                f"  HTML splitter [{target.value}]: {chunk_size}/{chunk_overlap}"
            )

        # --- Splitter diretto per PDF (bandi e PDF generici, NO Parent-Child) ---
        # Parametri letti da settings (non hardcodati!)
        self._pdf_direct_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.ingestion.pdf_direct_chunk_size,
            chunk_overlap=settings.ingestion.pdf_direct_chunk_overlap,
            add_start_index=True,
        )
        logger.info(
            f"  PDF direct splitter: {settings.ingestion.pdf_direct_chunk_size}/"
            f"{settings.ingestion.pdf_direct_chunk_overlap}"
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
        """
        Indicizza tutti i file HTML dalla directory, instradando ciascuno
        nella collection corretta tramite DocumentRouter.
        
        Returns:
            Dict con statistiche: indexed, skipped, updated, orphans_removed,
            errors, routing (breakdown per collection).
        """
        directory = directory or self._settings.ingestion.html_raw_dir
        html_files = list(Path(directory).glob("*.html"))

        if not html_files:
            logger.warning(f"Nessun file HTML in {directory}")
            return {
                "indexed": 0, "skipped": 0, "updated": 0,
                "orphans_removed": 0, "errors": 0,
                "routing": {t.value: 0 for t in CollectionTarget},
            }

        stats = {
            "indexed": 0, "skipped": 0, "updated": 0,
            "orphans_removed": 0, "errors": 0,
        }
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
                    old_collection = self._resolve_collection_for_cleanup(existing)
                    self._delete_from_vectorstore(old_collection, existing.chroma_ids)
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
                logger.error(f"Errore HTML {filepath.name}: {e}", exc_info=True)
                stats["errors"] += 1

        # Pulizia orfani: documenti nel registro ma non più presenti su disco
        html_orphans = {
            sid for sid in self._registry.all_source_ids()
            if sid.startswith("html:") and sid not in current_sources
        }
        for orphan_id in html_orphans:
            entry = self._registry.remove(orphan_id)
            if entry:
                old_collection = self._resolve_collection_for_cleanup(entry)
                self._delete_from_vectorstore(old_collection, entry.chroma_ids)
                stats["orphans_removed"] += 1
                logger.info(f"Orfano rimosso: {orphan_id} da {entry.collection_name}")

        self._registry.save()
        stats["routing"] = routing_stats
        logger.info(f"HTML indexing completato: {stats}")
        logger.info(f"HTML routing breakdown: {routing_stats}")
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
        
        logger.debug(
            f"HTML '{filename}': {len(chunks)} chunks → {collection.value}"
        )
        return chroma_ids

    # ==========================================================
    # PIPELINE PDF (con routing + chunking differenziato)
    # ==========================================================

    def index_pdf_list(self, pdf_links_file: Optional[str] = None) -> dict:
        """
        Indicizza tutti i PDF dalla lista di link, instradando ciascuno
        nella collection corretta e applicando la strategia di chunking
        appropriata (Parent-Child vs diretto).
        
        Returns:
            Dict con statistiche: indexed, skipped, updated, errors,
            parent_child_count, direct_count, routing (breakdown per collection).
        """
        pdf_links_file = pdf_links_file or self._settings.ingestion.pdf_links_file
        download_dir = self._settings.ingestion.pdf_download_dir

        if not os.path.exists(pdf_links_file):
            logger.warning(f"File lista PDF non trovato: {pdf_links_file}")
            return {
                "indexed": 0, "skipped": 0, "updated": 0, "errors": 0,
                "parent_child_count": 0, "direct_count": 0,
                "routing": {t.value: 0 for t in CollectionTarget},
            }

        with open(pdf_links_file, "r") as f:
            pdf_urls = [line.strip() for line in f if line.strip()]

        os.makedirs(download_dir, exist_ok=True)
        stats = {"indexed": 0, "skipped": 0, "updated": 0, "errors": 0}
        routing_stats: Dict[str, int] = {t.value: 0 for t in CollectionTarget}
        pc_count = 0
        direct_count = 0

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
                    if use_pc:
                        self._delete_from_vectorstore(
                            self._pc_child_vectorstore, existing.chroma_ids
                        )
                    else:
                        old_collection = self._resolve_collection_for_cleanup(existing)
                        self._delete_from_vectorstore(old_collection, existing.chroma_ids)
                    stats["updated"] += 1
                else:
                    stats["indexed"] += 1

                # Metadati arricchiti
                extra_meta = DocumentRouter.extract_metadata(url, collection_target)

                # --- CHUNKING DIFFERENZIATO ---
                if use_pc:
                    chroma_ids = self._index_pdf_parent_child(
                        local_path, url, extra_meta
                    )
                    pc_count += 1
                else:
                    chroma_ids = self._index_pdf_direct(
                        local_path, url, collection_target, extra_meta
                    )
                    direct_count += 1

                self._registry.upsert(source_id, IndexEntry(
                    content_hash=file_hash,
                    chroma_ids=chroma_ids,
                    collection_name=collection_target.value,
                ))
                routing_stats[collection_target.value] += 1

            except Exception as e:
                logger.error(f"Errore PDF {url}: {e}", exc_info=True)
                stats["errors"] += 1

        self._registry.save()
        stats["parent_child_count"] = pc_count
        stats["direct_count"] = direct_count
        stats["routing"] = routing_stats
        logger.info(f"PDF indexing completato: {stats}")
        logger.info(f"PDF routing breakdown: {routing_stats}")
        logger.info(f"PDF chunking: {pc_count} Parent-Child, {direct_count} diretto")
        return stats

    def _index_pdf_parent_child(
        self, local_path: str, source_url: str, extra_meta: dict
    ) -> List[str]:
        """Parent-Child per regolamenti e piani di studio."""
        loader = PyPDFLoader(local_path)
        pages = loader.load()

        for page in pages:
            page.metadata.update({
                "source_url": source_url,
                "doc_type": "pdf",
                **extra_meta,
            })

        # Cattura ID prima dell'inserimento per il registro incrementale
        existing_data = self._pc_child_vectorstore.get()
        ids_before = set(existing_data["ids"]) if existing_data and existing_data.get("ids") else set()

        self._parent_child_retriever.add_documents(pages)

        ids_after = set(self._pc_child_vectorstore.get()["ids"])
        new_ids = list(ids_after - ids_before)

        logger.info(
            f"  PDF Parent-Child: '{os.path.basename(local_path)}' → "
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
        """Chunking diretto per bandi e PDF che NON necessitano di Parent-Child."""
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

        logger.info(
            f"  PDF diretto: '{filename}' → "
            f"{len(chunks)} chunks → {collection.value}"
        )
        return chroma_ids

    # ==========================================================
    # RETRIEVER ACCESSORS
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
    # PRIVATE HELPERS
    # ==========================================================

    def _resolve_collection_for_cleanup(self, entry: IndexEntry) -> Chroma:
        """
        Risolve il vectorstore Chroma corretto per eliminare i chunk
        di un'entry del registro, basandosi su entry.collection_name.
        """
        if entry.collection_name:
            try:
                target = CollectionTarget(entry.collection_name)
                return self._collections[target]
            except ValueError:
                logger.warning(
                    f"collection_name non riconosciuto: '{entry.collection_name}', "
                    f"fallback a DIPARTIMENTO_RICERCA"
                )
        return self._collections[CollectionTarget.DIPARTIMENTO_RICERCA]

    @staticmethod
    def _delete_from_vectorstore(vectorstore: Chroma, ids: List[str]) -> None:
        """Elimina chunk da Chroma per ID, con gestione errori."""
        if not ids:
            return
        try:
            vectorstore.delete(ids=ids)
        except Exception as e:
            logger.warning(f"Errore eliminazione da Chroma ({len(ids)} ids): {e}")

    @staticmethod
    def _download_if_needed(url: str, local_path: str) -> bool:
        """Scarica un file se non esiste già su disco."""
        if os.path.exists(local_path):
            return True
        try:
            response = requests.get(url, timeout=60)
            if response.ok:
                with open(local_path, "wb") as f:
                    f.write(response.content)
                return True
            logger.warning(f"Download fallito ({response.status_code}): {url}")
            return False
        except Exception as e:
            logger.error(f"Errore download {url}: {e}")
            return False

    @staticmethod
    def _extract_source_url(html_content: str) -> Optional[str]:
        """Estrae il source URL dal commento <!-- SOURCE: ... --> nell'HTML."""
        match = re.search(r"<!--\s*SOURCE:\s*(https?://[^\s]+)\s*-->", html_content)
        return match.group(1).strip() if match else None

    @staticmethod
    def _safe_filename(url: str) -> str:
        """Genera un nome file sicuro da un URL."""
        name = re.sub(
            r'[<>:"/\\|?*]', '_',
            url.replace("https://", "").replace("http://", "")
        )
        return name[:120]