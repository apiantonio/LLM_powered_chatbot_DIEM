"""Indicizzatore multi-collezione per la Knowledge Base.

Gestisce il chunking, l'embedding e l'inserimento nel vector store Chroma
di documenti HTML, PDF e Markdown, con supporto alla strategia
Parent-Child per PDF regolamentari.
"""

import os
import re
import logging
from typing import List, Optional, Dict
from pathlib import Path

import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    HTMLSectionSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore

from src.config.settings import AppSettings
from src.ingestion.registry import IndexRegistry, IndexEntry
from src.ingestion.router import DocumentRouter, CollectionTarget

logger = logging.getLogger(__name__)


class KnowledgeBaseIndexer:
    """Indicizzatore multi-collezione per documenti HTML, PDF e Markdown.

    Coordina il ciclo di vita completo dell'indicizzazione: caricamento,
    chunking, arricchimento con metadati contestuali, embedding e
    persistenza nel vector store Chroma. Supporta indicizzazione
    incrementale tramite IndexRegistry.
    """

    def __init__(self, settings: AppSettings):
        """Inizializza l'indicizzatore con tutte le componenti necessarie.

        Args:
            settings: Configurazione completa dell'applicazione.
        """
        self._settings = settings
        self._embedding_model = self._build_embedding_model(settings)
        self._collections = self._build_collections(settings)
        self._pc_child_vectorstore, self._pc_parent_docstore = (
            self._build_parent_child_stores(settings)
        )
        self._parent_child_retriever = self._build_parent_child_retriever(settings)
        self._html_section_splitter = self._build_html_section_splitter()
        self._html_splitters = self._build_html_splitters(settings)
        self._pdf_direct_splitter = self._build_pdf_direct_splitter(settings)
        self._md_header_splitter, self._md_text_splitter = (
            self._build_md_splitters(settings)
        )
        self._registry = IndexRegistry(settings.ingestion.index_registry_path)

        logger.info(
            "Indexer multi-collection inizializzato: %s",
            [t.value for t in CollectionTarget],
        )

    def _build_embedding_model(self, settings: AppSettings) -> HuggingFaceEmbeddings:
        """Costruisce il modello di embedding HuggingFace.

        Args:
            settings: Configurazione dell'applicazione.

        Returns:
            Istanza di HuggingFaceEmbeddings configurata.
        """
        return HuggingFaceEmbeddings(
            model_name=settings.embedding.model_name,
            encode_kwargs={
                "normalize_embeddings": settings.embedding.normalize_embeddings,
            },
        )

    def _build_collections(
        self, settings: AppSettings
    ) -> Dict[CollectionTarget, Chroma]:
        """Costruisce le collezioni Chroma per ciascun CollectionTarget.

        Args:
            settings: Configurazione dell'applicazione.

        Returns:
            Dizionario CollectionTarget -> Chroma.
        """
        persist_dir = settings.vectorstore.persist_directory
        os.makedirs(persist_dir, exist_ok=True)

        collections: Dict[CollectionTarget, Chroma] = {}
        for target in CollectionTarget:
            collections[target] = Chroma(
                collection_name=target.value,
                embedding_function=self._embedding_model,
                persist_directory=persist_dir,
            )
            logger.info("Collection Chroma inizializzata: %s", target.value)
        return collections

    def _build_parent_child_stores(
        self, settings: AppSettings
    ) -> tuple[Chroma, object]:
        """Costruisce il vector store child e il docstore parent per Parent-Child.

        Args:
            settings: Configurazione dell'applicazione.

        Returns:
            Tupla (child_vectorstore, parent_docstore).
        """
        persist_dir = settings.vectorstore.persist_directory
        parent_dir = settings.vectorstore.parent_store_directory
        os.makedirs(parent_dir, exist_ok=True)

        child_vectorstore = Chroma(
            collection_name=settings.vectorstore.parent_child_collection_name,
            embedding_function=self._embedding_model,
            persist_directory=persist_dir,
        )
        parent_docstore = create_kv_docstore(LocalFileStore(parent_dir))
        return child_vectorstore, parent_docstore

    def _build_parent_child_retriever(
        self, settings: AppSettings
    ) -> ParentDocumentRetriever:
        """Costruisce il ParentDocumentRetriever con gli splitter configurati.

        Args:
            settings: Configurazione dell'applicazione.

        Returns:
            Istanza di ParentDocumentRetriever.
        """
        retriever = ParentDocumentRetriever(
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
            "Parent-Child Retriever: child=%d, parent=%d",
            settings.ingestion.pdf_child_chunk_size,
            settings.ingestion.pdf_parent_chunk_size,
        )
        return retriever

    def _build_html_section_splitter(self) -> HTMLSectionSplitter:
        """Costruisce lo splitter per sezioni HTML basato su header h1-h3.

        Returns:
            Istanza di HTMLSectionSplitter.
        """
        return HTMLSectionSplitter(
            headers_to_split_on=[
                ("h1", "Titolo"),
                ("h2", "Sezione"),
                ("h3", "Sottosezione"),
            ]
        )

    def _build_html_splitters(
        self, settings: AppSettings
    ) -> Dict[CollectionTarget, RecursiveCharacterTextSplitter]:
        """Costruisce gli splitter di testo HTML per ciascuna collezione.

        Args:
            settings: Configurazione dell'applicazione.

        Returns:
            Dizionario CollectionTarget -> RecursiveCharacterTextSplitter.
        """
        splitters: Dict[CollectionTarget, RecursiveCharacterTextSplitter] = {}
        for target in CollectionTarget:
            chunk_size, chunk_overlap = settings.ingestion.get_collection_html_params(
                target.value
            )
            splitters[target] = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                add_start_index=True,
            )
            logger.info(
                "HTML splitter [%s]: %d/%d", target.value, chunk_size, chunk_overlap
            )
        return splitters

    def _build_pdf_direct_splitter(
        self, settings: AppSettings
    ) -> RecursiveCharacterTextSplitter:
        """Costruisce lo splitter per il chunking diretto dei PDF.

        Args:
            settings: Configurazione dell'applicazione.

        Returns:
            Istanza di RecursiveCharacterTextSplitter.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.ingestion.pdf_direct_chunk_size,
            chunk_overlap=settings.ingestion.pdf_direct_chunk_overlap,
            add_start_index=True,
        )
        logger.info(
            "PDF direct splitter: %d/%d",
            settings.ingestion.pdf_direct_chunk_size,
            settings.ingestion.pdf_direct_chunk_overlap,
        )
        return splitter

    def _build_md_splitters(
        self, settings: AppSettings
    ) -> tuple[MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter]:
        """Costruisce gli splitter per documenti Markdown.

        Args:
            settings: Configurazione dell'applicazione.

        Returns:
            Tupla (header_splitter, text_splitter).
        """
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
        )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.ingestion.md_chunk_size,
            chunk_overlap=settings.ingestion.md_chunk_overlap,
            add_start_index=True,
        )
        return header_splitter, text_splitter

    @staticmethod
    def _build_context_prefix(
        content_metadata: dict, collection: CollectionTarget
    ) -> str:
        """Costruisce il prefisso contestuale da anteporre al contenuto dei chunk.

        Args:
            content_metadata: Metadati del documento sorgente.
            collection: Collezione target del documento.

        Returns:
            Stringa di prefisso contestuale o stringa vuota.
        """
        clean_metadata = {
            k: v.lower() if isinstance(v, str) else v
            for k, v in content_metadata.items()
        }

        parts = []

        if collection == CollectionTarget.PERSONE:
            if "nome_docente" in clean_metadata:
                parts.append(f"Docente: {clean_metadata['nome_docente']}")
            if "matricola" in clean_metadata:
                parts.append(f"Matricola: {clean_metadata['matricola']}")
            if "sotto_area" in clean_metadata:
                parts.append(f"Sezione: {clean_metadata['sotto_area']}")
            if "laboratorio_nome" in clean_metadata:
                parts.append(f"Laboratorio: {clean_metadata['laboratorio_nome']}")
            if "spin_off" in clean_metadata:
                parts.append(f"Spin-off: {clean_metadata['spin_off']}")
            if "premi_ricerca" in clean_metadata:
                parts.append(f"Premio: {clean_metadata['premi_ricerca']}")
            if "brevetti" in clean_metadata:
                parts.append(f"Brevetto: {clean_metadata['brevetti']}")
            if "pubblicazioni" in clean_metadata:
                parts.append(f"Pubblicazione: {clean_metadata['pubblicazioni']}")
            if "progetti" in clean_metadata:
                parts.append(f"Progetto: {clean_metadata['progetti']}")
            if "nomi_insegnamenti" in clean_metadata:
                parts.append(f"Insegnamento: {clean_metadata['nomi_insegnamenti']}")

        elif collection == CollectionTarget.OFFERTA_FORMATIVA:
            if "nome_corso" in clean_metadata:
                parts.append(f"Corso di Laurea: {clean_metadata['nome_corso']}")
            if "sotto_area" in clean_metadata:
                parts.append(f"Sezione: {clean_metadata['sotto_area']}")

        elif collection == CollectionTarget.DIPARTIMENTO:
            if "laboratorio_nome" in clean_metadata:
                parts.append(f"Laboratorio: {clean_metadata['laboratorio_nome']}")
            if "tipo_bando" in clean_metadata:
                parts.append(f"Tipo bando: {clean_metadata['tipo_bando']}")
            if "sotto_area" in clean_metadata:
                parts.append(f"Sezione: {clean_metadata['sotto_area']}")
            if "titolo_documento" in clean_metadata:
                parts.append(
                    f"Documento: {clean_metadata['titolo_documento'].title()}"
                )

        if "doc_id" in clean_metadata:
            parts.append(f"Codice ID: {clean_metadata['doc_id']}")
        if "anno" in clean_metadata:
            parts.append(f"Anno: {clean_metadata['anno']}")

        if not parts:
            return ""

        return " | ".join(parts) + "\n\n"

    @staticmethod
    def _inject_context_in_chunks(
        chunks: List[Document],
        context_prefix: str,
        all_metadata: dict,
    ) -> List[Document]:
        """Inietta il prefisso contestuale e i metadati in ciascun chunk.

        Args:
            chunks: Lista di chunk Document da arricchire.
            context_prefix: Prefisso testuale da anteporre al contenuto.
            all_metadata: Metadati da aggiungere a ciascun chunk.

        Returns:
            Lista di chunk arricchiti.
        """
        for chunk in chunks:
            chunk.metadata.update(all_metadata)
            if context_prefix:
                chunk.page_content = context_prefix + chunk.page_content
        return chunks

    def index_html_directory(self, directory: Optional[str] = None) -> dict:
        """Indicizza incrementalmente tutti i file HTML nella directory specificata.

        Args:
            directory: Percorso della directory contenente i file HTML.
                       Se None, usa il valore da settings.

        Returns:
            Dizionario con statistiche di indicizzazione.
        """
        directory = directory or self._settings.ingestion.html_raw_dir
        html_files = list(Path(directory).glob("*.html"))

        if not html_files:
            logger.warning("Nessun file HTML in %s", directory)
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

                source_url = self._extract_source_url(content) or filepath.name
                collection_target = DocumentRouter.route_html(source_url)
                vectorstore = self._collections[collection_target]

                if existing:
                    old_collection = self._resolve_collection_for_cleanup(existing)
                    self._delete_from_vectorstore(old_collection, existing.chroma_ids)
                    stats["updated"] += 1
                else:
                    stats["indexed"] += 1

                url_metadata = DocumentRouter.extract_metadata(
                    source_url, collection_target
                )
                content_metadata = DocumentRouter.extract_content_metadata(
                    content, source_url, collection_target
                )
                all_metadata = {**url_metadata, **content_metadata}

                chroma_ids = self._chunk_and_index_html(
                    content, source_url, filepath.name,
                    vectorstore, collection_target, all_metadata,
                )

                self._registry.upsert(source_id, IndexEntry(
                    content_hash=content_hash,
                    chroma_ids=chroma_ids,
                    collection_name=collection_target.value,
                ))
                routing_stats[collection_target.value] += 1

            except Exception as e:
                logger.error("Errore HTML %s: %s", filepath.name, e, exc_info=True)
                stats["errors"] += 1

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
                logger.info(
                    "Orfano rimosso: %s da %s", orphan_id, entry.collection_name
                )

        self._registry.save()
        stats["routing"] = routing_stats
        logger.info("HTML indexing completato: %s", stats)
        logger.info("HTML routing breakdown: %s", routing_stats)
        return stats

    def _chunk_and_index_html(
        self,
        content: str,
        source_url: str,
        filename: str,
        vectorstore: Chroma,
        collection: CollectionTarget,
        all_metadata: dict,
    ) -> List[str]:
        """Esegue il chunking e l'indicizzazione di un singolo documento HTML.

        Args:
            content: Contenuto HTML grezzo.
            source_url: URL sorgente della pagina.
            filename: Nome del file su disco.
            vectorstore: Istanza Chroma di destinazione.
            collection: CollectionTarget di appartenenza.
            all_metadata: Metadati completi da iniettare nei chunk.

        Returns:
            Lista degli ID Chroma generati per i chunk.
        """
        section_docs = self._html_section_splitter.split_text(content)

        normalized = []
        for doc in section_docs:
            if isinstance(doc, str):
                doc = Document(page_content=doc, metadata={})
            doc.metadata.update({
                "source_url": source_url,
                "source_file": filename,
                "doc_type": "html",
            })
            normalized.append(doc)

        splitter = self._html_splitters[collection]
        chunks = splitter.split_documents(normalized)
        if not chunks:
            return []

        context_prefix = self._build_context_prefix(all_metadata, collection)
        chunks = self._inject_context_in_chunks(chunks, context_prefix, all_metadata)

        chroma_ids = [
            f"{collection.value}_html_{filename}_{i}"
            for i in range(len(chunks))
        ]
        vectorstore.add_documents(chunks, ids=chroma_ids)

        logger.debug(
            "HTML '%s': %d chunks -> %s (context prefix: %s)",
            filename, len(chunks), collection.value, bool(context_prefix),
        )
        return chroma_ids

    def index_pdf_list(self, pdf_links_file: Optional[str] = None) -> dict:
        """Indicizza incrementalmente tutti i PDF elencati nel file di link.

        Args:
            pdf_links_file: Percorso del file contenente gli URL dei PDF.
                            Se None, usa il valore da settings.

        Returns:
            Dizionario con statistiche di indicizzazione.
        """
        pdf_links_file = pdf_links_file or self._settings.ingestion.pdf_links_file
        download_dir = self._settings.ingestion.pdf_download_dir

        if not os.path.exists(pdf_links_file):
            logger.warning("File lista PDF non trovato: %s", pdf_links_file)
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

                collection_target = DocumentRouter.route_pdf(url)
                use_pc = DocumentRouter.use_parent_child(collection_target, url)

                if existing and existing.chroma_ids:
                    if use_pc:
                        self._delete_from_vectorstore(
                            self._pc_child_vectorstore, existing.chroma_ids
                        )
                    else:
                        old_collection = self._resolve_collection_for_cleanup(existing)
                        self._delete_from_vectorstore(
                            old_collection, existing.chroma_ids
                        )
                    stats["updated"] += 1
                else:
                    stats["indexed"] += 1

                extra_meta = DocumentRouter.extract_pdf_metadata(url, collection_target)

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
                logger.error("Errore PDF %s: %s", url, e, exc_info=True)
                stats["errors"] += 1

        self._registry.save()
        stats["parent_child_count"] = pc_count
        stats["direct_count"] = direct_count
        stats["routing"] = routing_stats
        logger.info("PDF indexing completato: %s", stats)
        logger.info("PDF routing breakdown: %s", routing_stats)
        logger.info(
            "PDF chunking: %d Parent-Child, %d diretto", pc_count, direct_count
        )
        return stats

    def _index_pdf_parent_child(
        self, local_path: str, source_url: str, extra_meta: dict
    ) -> List[str]:
        """Indicizza un PDF con strategia Parent-Child.

        Args:
            local_path: Percorso locale del file PDF.
            source_url: URL sorgente del PDF.
            extra_meta: Metadati aggiuntivi da iniettare.

        Returns:
            Lista degli ID Chroma dei child chunk generati.
        """
        loader = PyPDFLoader(local_path)
        pages = loader.load()

        for page in pages:
            self._normalize_pdf_creation_date(page)
            page.metadata.update({
                "source_url": source_url,
                "doc_type": "pdf",
                **extra_meta,
            })

            context_prefix = self._build_context_prefix(
                page.metadata, CollectionTarget.OFFERTA_FORMATIVA
            )
            if context_prefix:
                page.page_content = context_prefix + page.page_content

        existing_data = self._pc_child_vectorstore.get()
        ids_before = (
            set(existing_data["ids"])
            if existing_data and existing_data.get("ids")
            else set()
        )

        self._parent_child_retriever.add_documents(pages)

        ids_after = set(self._pc_child_vectorstore.get()["ids"])
        new_ids = list(ids_after - ids_before)

        logger.info(
            "PDF Parent-Child: '%s' -> %d child chunks",
            os.path.basename(local_path), len(new_ids),
        )
        return new_ids

    def _index_pdf_direct(
        self,
        local_path: str,
        source_url: str,
        collection: CollectionTarget,
        extra_meta: dict,
    ) -> List[str]:
        """Indicizza un PDF con chunking diretto nella collezione specificata.

        Args:
            local_path: Percorso locale del file PDF.
            source_url: URL sorgente del PDF.
            collection: CollectionTarget di destinazione.
            extra_meta: Metadati aggiuntivi da iniettare.

        Returns:
            Lista degli ID Chroma generati per i chunk.
        """
        loader = PyPDFLoader(local_path)
        pages = loader.load()

        for page in pages:
            self._normalize_pdf_creation_date(page)
            page.metadata.update({
                "source_url": source_url,
                "doc_type": "pdf",
                **extra_meta,
            })

        chunks = self._pdf_direct_splitter.split_documents(pages)
        if not chunks:
            return []

        for chunk in chunks:
            chunk.metadata.update(extra_meta)
            context_prefix = self._build_context_prefix(chunk.metadata, collection)
            if context_prefix:
                chunk.page_content = context_prefix + chunk.page_content

        vectorstore = self._collections[collection]
        filename = os.path.basename(local_path)
        chroma_ids = [
            f"{collection.value}_pdf_{filename}_{i}"
            for i in range(len(chunks))
        ]
        vectorstore.add_documents(chunks, ids=chroma_ids)

        logger.info(
            "PDF diretto: '%s' -> %d chunks -> %s",
            filename, len(chunks), collection.value,
        )
        return chroma_ids

    def index_markdown_directory(self, directory: Optional[str] = None) -> dict:
        """Indicizza incrementalmente tutti i file Markdown nella directory specificata.

        Args:
            directory: Percorso della directory contenente i file Markdown.
                       Se None, usa il valore da settings.

        Returns:
            Dizionario con statistiche di indicizzazione.
        """
        directory = directory or self._settings.ingestion.md_static_dir
        if not os.path.exists(directory):
            logger.warning("Directory MD non trovata: %s", directory)
            return {
                "indexed": 0, "skipped": 0, "updated": 0,
                "orphans_removed": 0, "errors": 0,
            }

        md_files = list(Path(directory).glob("*.md"))
        stats = {
            "indexed": 0, "skipped": 0, "updated": 0,
            "orphans_removed": 0, "errors": 0,
        }
        current_sources = set()

        for filepath in md_files:
            source_id = f"md:{filepath.name}"
            current_sources.add(source_id)

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                content_hash = IndexRegistry.compute_hash(content)
                existing = self._registry.get(source_id)

                if existing and existing.content_hash == content_hash:
                    stats["skipped"] += 1
                    continue

                collection_target = DocumentRouter.route_md(filepath.name)
                vectorstore = self._collections[collection_target]

                if existing:
                    old_collection = self._resolve_collection_for_cleanup(existing)
                    self._delete_from_vectorstore(old_collection, existing.chroma_ids)
                    stats["updated"] += 1
                else:
                    stats["indexed"] += 1

                url_metadata = DocumentRouter.extract_md_metadata(
                    filepath.name, collection_target
                )
                content_metadata = DocumentRouter.extract_content_metadata_md(
                    content, filepath.name
                )
                all_metadata = {**url_metadata, **content_metadata}

                chroma_ids = self._chunk_and_index_md(
                    content, filepath.name, vectorstore,
                    collection_target, all_metadata,
                )

                self._registry.upsert(source_id, IndexEntry(
                    content_hash=content_hash,
                    chroma_ids=chroma_ids,
                    collection_name=collection_target.value,
                ))

            except Exception as e:
                logger.error("Errore MD %s: %s", filepath.name, e, exc_info=True)
                stats["errors"] += 1

        md_orphans = {
            sid for sid in self._registry.all_source_ids()
            if sid.startswith("md:") and sid not in current_sources
        }
        for orphan_id in md_orphans:
            entry = self._registry.remove(orphan_id)
            if entry:
                old_collection = self._resolve_collection_for_cleanup(entry)
                self._delete_from_vectorstore(old_collection, entry.chroma_ids)
                stats["orphans_removed"] += 1

        self._registry.save()
        logger.info("MD indexing completato: %s", stats)
        return stats

    def _chunk_and_index_md(
        self,
        content: str,
        filename: str,
        vectorstore: Chroma,
        collection: CollectionTarget,
        all_metadata: dict,
    ) -> List[str]:
        """Esegue il chunking e l'indicizzazione di un singolo documento Markdown.

        Args:
            content: Contenuto testuale del file Markdown.
            filename: Nome del file su disco.
            vectorstore: Istanza Chroma di destinazione.
            collection: CollectionTarget di appartenenza.
            all_metadata: Metadati completi da iniettare nei chunk.

        Returns:
            Lista degli ID Chroma generati per i chunk.
        """
        header_splits = self._md_header_splitter.split_text(content)

        normalized = []
        for doc in header_splits:
            doc.metadata.update({
                "source_url": filename,
                "doc_type": "md",
            })
            normalized.append(doc)

        chunks = self._md_text_splitter.split_documents(normalized)
        if not chunks:
            return []

        context_prefix = self._build_context_prefix(all_metadata, collection)
        chunks = self._inject_context_in_chunks(chunks, context_prefix, all_metadata)

        chroma_ids = [
            f"{collection.value}_md_{filename}_{i}"
            for i in range(len(chunks))
        ]
        vectorstore.add_documents(chunks, ids=chroma_ids)

        return chroma_ids

    def get_collection_retriever(
        self,
        collection: CollectionTarget,
        search_type: Optional[str] = None,
        k: Optional[int] = None,
    ):
        """Restituisce un retriever per la collezione specificata.

        Args:
            collection: CollectionTarget da interrogare.
            search_type: Tipo di ricerca (default da settings).
            k: Numero di risultati (default da settings).

        Returns:
            Retriever LangChain configurato.
        """
        return self._collections[collection].as_retriever(
            search_type=search_type or self._settings.vectorstore.search_type,
            search_kwargs={"k": k or self._settings.vectorstore.search_k},
        )

    def get_parent_child_retriever(self, k: Optional[int] = None):
        """Restituisce il retriever Parent-Child configurato.

        Args:
            k: Numero di risultati (default da settings).

        Returns:
            ParentDocumentRetriever configurato.
        """
        self._parent_child_retriever.search_kwargs = {
            "k": k or self._settings.vectorstore.search_k,
        }
        return self._parent_child_retriever

    def get_all_retrievers(self) -> dict:
        """Restituisce tutti i retriever disponibili, incluso il Parent-Child.

        Returns:
            Dizionario con retriever per ogni collezione e per Parent-Child.
        """
        retrievers = {}
        for target in CollectionTarget:
            retrievers[target] = self.get_collection_retriever(target)
        retrievers["parent_child"] = self.get_parent_child_retriever()
        return retrievers

    def _resolve_collection_for_cleanup(self, entry: IndexEntry) -> Chroma:
        """Risolve la collezione Chroma per la pulizia di un'entry esistente.

        Gestisce anche nomi di collezione legacy con mapping verso le
        collezioni attuali.

        Args:
            entry: Voce del registry da cui estrarre il nome collezione.

        Returns:
            Istanza Chroma corrispondente.
        """
        if entry.collection_name:
            try:
                target = CollectionTarget(entry.collection_name)
                return self._collections[target]
            except ValueError:
                legacy_mapping = {
                    "docenti_e_didattica": CollectionTarget.PERSONE,
                    "bandi_e_amministrazione": CollectionTarget.DIPARTIMENTO,
                    "dipartimento_e_ricerca": CollectionTarget.DIPARTIMENTO,
                }
                mapped = legacy_mapping.get(entry.collection_name)
                if mapped:
                    return self._collections[mapped]
                logger.warning(
                    "collection_name non riconosciuto: '%s', fallback a DIPARTIMENTO",
                    entry.collection_name,
                )
        return self._collections[CollectionTarget.DIPARTIMENTO]

    @staticmethod
    def _normalize_pdf_creation_date(page: Document) -> None:
        """Normalizza il campo creationdate di un documento PDF in anno.

        Estrae l'anno dal campo creationdate dei metadati e lo memorizza
        nel campo 'anno', rimuovendo il campo originale.

        Args:
            page: Documento LangChain rappresentante una pagina PDF.
        """
        if "creationdate" not in page.metadata:
            return

        raw_date = str(page.metadata.pop("creationdate"))
        date_match = re.search(r"(\d{4})", raw_date)

        if date_match:
            page.metadata["anno"] = date_match.group(1)
        else:
            page.metadata["anno"] = raw_date[:4]

    @staticmethod
    def _delete_from_vectorstore(vectorstore: Chroma, ids: List[str]) -> None:
        """Elimina chunk da Chroma per ID, con gestione errori.

        Args:
            vectorstore: Istanza Chroma da cui eliminare.
            ids: Lista degli ID da rimuovere.
        """
        if not ids:
            return
        try:
            vectorstore.delete(ids=ids)
        except Exception as e:
            logger.warning(
                "Errore eliminazione da Chroma (%d ids): %s", len(ids), e
            )

    @staticmethod
    def _download_if_needed(url: str, local_path: str) -> bool:
        """Scarica un file se non esiste gia su disco.

        Args:
            url: URL del file da scaricare.
            local_path: Percorso locale di destinazione.

        Returns:
            True se il file e' disponibile localmente, False altrimenti.
        """
        if os.path.exists(local_path):
            return True
        try:
            response = requests.get(url, timeout=60)
            if response.ok:
                with open(local_path, "wb") as f:
                    f.write(response.content)
                return True
            logger.warning("Download fallito (%d): %s", response.status_code, url)
            return False
        except Exception as e:
            logger.error("Errore download %s: %s", url, e)
            return False

    @staticmethod
    def _extract_source_url(html_content: str) -> Optional[str]:
        """Estrae l'URL sorgente dal commento HTML embedded nel contenuto.

        Args:
            html_content: Contenuto HTML in cui cercare il commento SOURCE.

        Returns:
            URL sorgente o None se non trovato.
        """
        match = re.search(r"<!--\s*SOURCE:\s*(https?://[^\s]+)\s*-->", html_content)
        return match.group(1).strip() if match else None

    @staticmethod
    def _safe_filename(url: str) -> str:
        """Genera un nome file sicuro a partire da un URL.

        Args:
            url: URL da convertire in nome file.

        Returns:
            Stringa sicura per l'uso come nome file, troncata a 120 caratteri.
        """
        name = re.sub(
            r'[<>:"/\\|?*]', '_',
            url.replace("https://", "").replace("http://", ""),
        )
        return name[:120]