"""
ingestion/indexer.py — Motore di indicizzazione RIVISTO per 3 Vector Store.

REFACTORING COMPLETO secondo audit_fattibilita_metadati.md:

  ARCHITETTURA 3 VECTOR STORE (audit §8):
    1. PERSONE — pagine docente (docenti.unisa.it)
    2. OFFERTA_FORMATIVA — corsi + PDF regolamenti/piani
    3. DIPARTIMENTO — diem.unisa.it (inclusi bandi, laboratori, ricerca)

  CHUNKING CONTEXT-AWARE (risoluzione problema chunk orfani):
    Il problema critico identificato nei report di chunking era che solo
    il primo chunk conteneva il nome del docente/corso, mentre i chunk
    successivi erano "orfani" senza contesto identificativo.

    SOLUZIONE IMPLEMENTATA:
      1. Prima del chunking, si estraggono i metadati dal CONTENUTO HTML
         (nome_docente, matricola, nome_corso, ecc.) tramite
         DocumentRouter.extract_content_metadata().
      2. Questi metadati vengono iniettati in OGNI chunk come metadata
         Chroma, così ogni chunk è semanticamente autosufficiente.
      3. Inoltre, un PREFISSO CONTESTUALE viene preposto al page_content
         di ogni chunk (es. "Docente: Mario Rossi | Sezione: didattica |")
         così che anche la ricerca semantica per embedding tenga conto
         del contesto identificativo del documento sorgente.

  SCHEMA METADATI (audit §6):
    Ogni chunk porta i metadati dello schema definitivo dell'audit.
    Vedi il modulo router.py per i dettagli di estrazione.
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
    Motore di indicizzazione multi-collection con chunking context-aware.

    3 collection Chroma + 1 ParentDocumentRetriever dedicato ai PDF
    di regolamenti/piani di studio.

    La strategia di chunking context-aware garantisce che OGNI chunk
    contenga nei metadati e nel prefisso testuale le informazioni
    identificative del documento sorgente, eliminando il problema
    dei chunk orfani.
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

        # --- Istanziazione delle 3 collection Chroma (audit §8) ---
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

        # --- HTMLSectionSplitter per split strutturale ---
        self._html_section_splitter = HTMLSectionSplitter(
            headers_to_split_on=[
                ("h1", "Titolo"),
                ("h2", "Sezione"),
                ("h3", "Sottosezione"),
            ]
        )

        # --- Splitters HTML per-collection (parametri da settings) ---
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

        # --- Splitter diretto per PDF (bandi e PDF generici) ---
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
    # CHUNKING CONTEXT-AWARE — Risoluzione problema chunk orfani
    # ==========================================================

    @staticmethod
    def _build_context_prefix(content_metadata: dict, collection: CollectionTarget) -> str:
        """
        Costruisce il prefisso contestuale da iniettare in ogni chunk.

        QUESTA È LA SOLUZIONE AL PROBLEMA DEI CHUNK ORFANI:
        Ogni chunk viene prefissato con le informazioni identificative
        del documento sorgente, così che anche l'embedding vettoriale
        includa il contesto (docente, corso, laboratorio, ecc.).

        Esempi di prefisso generato:
          PERSONE: "Docente: Mario ROSSI | Matricola: 026104 | Sezione: didattica | "
          OFFERTA: "Corso di Laurea: Ingegneria Informatica | "
          DIPARTIMENTO: "Laboratorio: Embedded Intelligent Systems | "
        """
        parts = []

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
        """
        Inietta il contesto in ogni chunk — SOLUZIONE CHUNK ORFANI.

        Due livelli di iniezione:
          1. METADATA: ogni chunk riceve i metadati completi (matricola,
             nome_docente, sotto_area, ecc.) come campi metadata Chroma.
             Questo abilita il pre-filtering per metadati nel retrieval.
          2. PREFISSO TESTUALE: il prefisso contestuale viene preposto
             al page_content, così che l'embedding vettoriale del chunk
             contenga le informazioni identificative e il retrieval
             semantico funzioni anche per query che menzionano il docente
             o il corso.
        """
        for chunk in chunks:
            # Livello 1: metadati Chroma
            chunk.metadata.update(all_metadata)

            # Livello 2: prefisso nel contenuto testuale
            if context_prefix:
                chunk.page_content = context_prefix + chunk.page_content

        return chunks

    # ==========================================================
    # PIPELINE HTML (con routing + chunking context-aware)
    # ==========================================================

    def index_html_directory(self, directory: Optional[str] = None) -> dict:
        """
        Indicizza tutti i file HTML, instradando ciascuno nella collection
        corretta e applicando il chunking context-aware.

        Returns:
            Dict con statistiche di indicizzazione.
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

                # --- ROUTING: determina la collection target (audit §8) ---
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

                # --- ESTRAZIONE METADATI (audit §6) ---
                # Metadati dall'URL (matricola, sotto_area, corso_slug, ecc.)
                url_metadata = DocumentRouter.extract_metadata(
                    source_url, collection_target
                )
                # Metadati dal CONTENUTO HTML (nome_docente, nome_corso, ecc.)
                # CHIAVE PER IL CHUNKING CONTEXT-AWARE
                content_metadata = DocumentRouter.extract_content_metadata(
                    content, source_url, collection_target
                )

                # Unione metadati: URL + contenuto
                all_metadata = {**url_metadata, **content_metadata}

                # --- CHUNKING CONTEXT-AWARE ---
                chroma_ids = self._chunk_and_index_html(
                    content, source_url, filepath.name,
                    vectorstore, collection_target, all_metadata
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

        # Pulizia orfani
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
        all_metadata: dict,
    ) -> List[str]:
        """
        Chunking context-aware e indicizzazione di un documento HTML.

        FLUSSO:
          1. HTMLSectionSplitter: split strutturale per headers
          2. RecursiveCharacterTextSplitter: split per dimensione
          3. Iniezione contesto: prefisso testuale + metadati in ogni chunk
          4. Indicizzazione in Chroma

        Il prefisso contestuale (es. "Docente: Mario ROSSI | Sezione: didattica |")
        viene preposto al page_content di ogni chunk, risolvendo il problema
        dei chunk orfani: anche i chunk lontani dal <h1> conterranno
        l'informazione di chi è il docente o quale corso è.
        """
        # Step 1: split strutturale
        section_docs = self._html_section_splitter.split_text(content)

        normalized = []
        for doc in section_docs:
            if isinstance(doc, str):
                doc = Document(page_content=doc, metadata={})
            # Metadati base (sovrascrivibili da all_metadata)
            doc.metadata.update({
                "source_url": source_url,
                "source_file": filename,
                "doc_type": "html",
            })
            normalized.append(doc)

        # Step 2: split per dimensione (parametri per-collection)
        splitter = self._html_splitters[collection]
        chunks = splitter.split_documents(normalized)
        if not chunks:
            return []

        # Step 3: INIEZIONE CONTESTO (soluzione chunk orfani)
        # Costruisce il prefisso dal contesto estratto
        context_prefix = self._build_context_prefix(all_metadata, collection)
        # Inietta metadati e prefisso in OGNI chunk
        chunks = self._inject_context_in_chunks(chunks, context_prefix, all_metadata)

        # Step 4: indicizzazione in Chroma
        chroma_ids = [
            f"{collection.value}_html_{filename}_{i}"
            for i in range(len(chunks))
        ]
        vectorstore.add_documents(chunks, ids=chroma_ids)

        logger.debug(
            f"HTML '{filename}': {len(chunks)} chunks → {collection.value} "
            f"(context prefix: {bool(context_prefix)})"
        )
        return chroma_ids

    # ==========================================================
    # PIPELINE PDF (con routing + chunking differenziato)
    # ==========================================================

    def index_pdf_list(self, pdf_links_file: Optional[str] = None) -> dict:
        """
        Indicizza tutti i PDF dalla lista di link, instradando ciascuno
        nella collection corretta con chunking appropriato.

        Returns:
            Dict con statistiche di indicizzazione.
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

                # --- ROUTING (audit §8) ---
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

                # Metadati PDF — audit §6 + §5.5 (anno ricalcolato)
                extra_meta = DocumentRouter.extract_pdf_metadata(url, collection_target)

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
            # 1. Controlliamo se esiste il metadato nativo del PDF
            if "creationdate" in page.metadata:
                # 2. Rimuoviamo la vecchia chiave e prendiamo il valore
                raw_date = str(page.metadata.pop("creationdate"))
                # 3. Estraiamo l'anno a 4 cifre
                date_match = re.search(r"(\d{4})", raw_date)
                
                if date_match:
                    anno_estratto = date_match.group(1)
                else:
                    anno_estratto = raw_date[:4]
                
                # 4. Essendo parent-child per offerta formativa, usiamo sempre la chiave 'anno'
                page.metadata["anno"] = anno_estratto

            page.metadata.update({
                "source_url": source_url,
                "doc_type": "pdf",
                **extra_meta,
            })
            
            # Generiamo e iniettiamo il prefisso sulla pagina intera prima dello split interno
            context_prefix = self._build_context_prefix(page.metadata, CollectionTarget.OFFERTA_FORMATIVA)
            if context_prefix:
                page.page_content = context_prefix + page.page_content

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
        """Chunking diretto per bandi e PDF generici."""
        loader = PyPDFLoader(local_path)
        pages = loader.load()

        for page in pages:
            # 1. Controlliamo se esiste il metadato nativo del PDF
            if "creationdate" in page.metadata:
                # 2. Rimuoviamo la vecchia chiave e prendiamo il valore come stringa
                raw_date = str(page.metadata.pop("creationdate"))
                # 3. Estraiamo solo le prime 4 cifre consecutive (l'anno)
                date_match = re.search(r"(\d{4})", raw_date)
                
                if date_match:
                    anno_estratto = date_match.group(1)
                else:
                    anno_estratto = raw_date[:4]
                
                
                page.metadata["anno"] = anno_estratto

            # 5. Applichiamo i metadati calcolati dal router
            page.metadata.update({
                "source_url": source_url,
                "doc_type": "pdf",
                **extra_meta,
            })

        chunks = self._pdf_direct_splitter.split_documents(pages)
        if not chunks:
            return []

        # Iniezione contesto per PDF (metadati in ogni chunk)
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
        """Risolve il vectorstore Chroma per eliminare chunk di un'entry."""
        if entry.collection_name:
            try:
                target = CollectionTarget(entry.collection_name)
                return self._collections[target]
            except ValueError:
                # Supporto retrocompatibilità nomi vecchi
                legacy_mapping = {
                    "docenti_e_didattica": CollectionTarget.PERSONE,
                    "bandi_e_amministrazione": CollectionTarget.DIPARTIMENTO,
                    "dipartimento_e_ricerca": CollectionTarget.DIPARTIMENTO,
                }
                mapped = legacy_mapping.get(entry.collection_name)
                if mapped:
                    return self._collections[mapped]
                logger.warning(
                    f"collection_name non riconosciuto: '{entry.collection_name}', "
                    f"fallback a DIPARTIMENTO"
                )
        return self._collections[CollectionTarget.DIPARTIMENTO]

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