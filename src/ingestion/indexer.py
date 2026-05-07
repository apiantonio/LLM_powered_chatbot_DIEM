"""
Modulo di indicizzazione duale: HTML (diretto) + PDF (Parent-Child).

Architettura differenziata per tipo di sorgente:

  HTML → HTMLSectionSplitter → RecursiveCharacterTextSplitter(700, 50) → Chroma
         I documenti HTML del DIEM sono brevi e ben strutturati.
         Ogni chunk è autosufficiente → indicizzazione diretta.

  PDF  → RecursiveCharacterTextSplitter(3000, 500) → Parent (LocalFileStore)
       → RecursiveCharacterTextSplitter(400, 50)  → Child  (Chroma, vettorializzato)
         I PDF contengono regolamenti di 50+ pagine dove una formula cruciale
         occupa una riga. Child piccoli = embedding focalizzato sulla formula.
         Parent grandi = contesto completo per il LLM alla generazione.

Pattern:
- Template Method: flusso Load → Split → Embed → Store con varianti per HTML/PDF.
- Strategy: chunking strategy selezionata in base al tipo di documento.
- Repository: IndexRegistry per aggiornamenti incrementali hash-based.

KPI Impact:
- Context Precision: child chunk piccoli nei PDF → embedding focalizzato → meno rumore.
- Context Recall: parent chunk ampi → LLM riceve contesto completo → nessuna info persa.
- Faithfulness: contesto pulito e completo → meno allucinazioni.
- Knowledge Freshness: registro incrementale → vector store sempre allineato.
"""

import os
import re
import uuid
import logging
import requests
from typing import List, Optional, Tuple
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, HTMLSectionSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.settings import AppSettings
from ingestion.registry import IndexRegistry, IndexEntry

logger = logging.getLogger(__name__)


class ParentDocStore:
    """
    Key-Value store su filesystem per i Parent chunk dei PDF.
    
    Ogni Parent viene salvato come file di testo con il suo ID come nome.
    Alla retrieval, il sistema recupera il Child da Chroma, legge il parent_id
    dai metadati, e carica il Parent da qui per passarlo al LLM.
    
    Perché non InMemoryStore: i Parent devono persistere tra restart.
    Perché non Chroma: i Parent NON vengono vettorializzati, servono solo alla generazione.
    """
    
    def __init__(self, store_dir: str):
        self._dir = store_dir
        os.makedirs(self._dir, exist_ok=True)
    
    def store(self, doc_id: str, content: str, metadata: dict) -> None:
        """Salva un Parent chunk su disco."""
        import json
        filepath = os.path.join(self._dir, f"{doc_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"content": content, "metadata": metadata}, f, ensure_ascii=False)
    
    def retrieve(self, doc_id: str) -> Optional[Document]:
        """Recupera un Parent chunk dal disco."""
        import json
        filepath = os.path.join(self._dir, f"{doc_id}.json")
        if not os.path.exists(filepath):
            logger.warning(f"Parent doc non trovato: {doc_id}")
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Document(page_content=data["content"], metadata=data.get("metadata", {}))
    
    def delete(self, doc_id: str) -> None:
        """Elimina un Parent chunk dal disco."""
        filepath = os.path.join(self._dir, f"{doc_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
    
    def delete_many(self, doc_ids: List[str]) -> None:
        """Elimina un batch di Parent chunk."""
        for doc_id in doc_ids:
            self.delete(doc_id)


class KnowledgeBaseIndexer:
    """
    Motore di indicizzazione duale con aggiornamento incrementale.
    
    Responsabilità unica: trasformare documenti grezzi in vettori ricercabili
    e mantenere il Vector Store sincronizzato con le sorgenti.
    """
    
    def __init__(self, settings: AppSettings):
        self._settings = settings
        
        # --- Embedding model (condiviso tra HTML e PDF) ---
        self._embedding_model = HuggingFaceEmbeddings(
            model_name=settings.embedding.model_name,
            encode_kwargs={"normalize_embeddings": settings.embedding.normalize_embeddings},
        )
        
        # --- Vector Store (Chroma) ---
        os.makedirs(settings.vectorstore.persist_directory, exist_ok=True)
        self._vectorstore = Chroma(
            collection_name=settings.vectorstore.collection_name,
            embedding_function=self._embedding_model,
            persist_directory=settings.vectorstore.persist_directory,
        )
        
        # --- Parent Doc Store (solo per PDF) ---
        self._parent_store = ParentDocStore(settings.vectorstore.parent_store_directory)
        
        # --- Registro incrementale ---
        self._registry = IndexRegistry(settings.ingestion.index_registry_path)
        
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
        
        # --- Splitters PDF (Parent e Child) ---
        self._pdf_parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.ingestion.pdf_parent_chunk_size,
            chunk_overlap=settings.ingestion.pdf_parent_chunk_overlap,
        )
        self._pdf_child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.ingestion.pdf_child_chunk_size,
            chunk_overlap=settings.ingestion.pdf_child_chunk_overlap,
            add_start_index=True,
        )
        
        logger.info(
            f"Indexer duale inizializzato — "
            f"HTML: {settings.ingestion.html_chunk_size}/{settings.ingestion.html_chunk_overlap}, "
            f"PDF Parent: {settings.ingestion.pdf_parent_chunk_size}, "
            f"PDF Child: {settings.ingestion.pdf_child_chunk_size}"
        )

    # ==========================================================
    # PIPELINE HTML: Indicizzazione diretta
    # ==========================================================

    def index_html_directory(self, directory: Optional[str] = None) -> dict:
        """
        Indicizza i file HTML con aggiornamento incrementale.
        
        Returns:
            Stats dict: {"indexed": N, "skipped": N, "updated": N, "orphans_removed": N}
        """
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
                
                # Skip se hash identico
                if existing and existing.content_hash == content_hash:
                    stats["skipped"] += 1
                    continue
                
                # Se hash diverso, elimina vecchi chunk prima
                if existing:
                    self._delete_chroma_ids(existing.chroma_ids)
                    stats["updated"] += 1
                else:
                    stats["indexed"] += 1
                
                # Chunking + indicizzazione
                source_url = self._extract_source_url(content)
                chroma_ids = self._chunk_and_index_html(content, source_url, filepath.name)
                
                # Aggiorna registro
                self._registry.upsert(source_id, IndexEntry(
                    content_hash=content_hash,
                    chroma_ids=chroma_ids,
                ))
                
            except Exception as e:
                logger.error(f"Errore indicizzazione HTML {filepath.name}: {e}")
        
        # Pulizia orfani: documenti nel registro ma non più nel filesystem
        html_orphans = {
            sid for sid in self._registry.all_source_ids()
            if sid.startswith("html:") and sid not in current_sources
        }
        for orphan_id in html_orphans:
            entry = self._registry.remove(orphan_id)
            if entry:
                self._delete_chroma_ids(entry.chroma_ids)
                stats["orphans_removed"] += 1
        
        self._registry.save()
        logger.info(f"HTML indexing: {stats}")
        return stats

    def _chunk_and_index_html(
        self, content: str, source_url: Optional[str], filename: str
    ) -> List[str]:
        """Chunking HTML e inserimento in Chroma. Restituisce i chroma_ids generati."""
        
        # Step 1: Split strutturale per sezioni HTML
        section_docs = self._html_section_splitter.split_text(content)
        
        # Normalizza in oggetti Document
        normalized = []
        for doc in section_docs:
            if isinstance(doc, str):
                doc = Document(page_content=doc, metadata={})
            doc.metadata["source_url"] = source_url or filename
            doc.metadata["source_file"] = filename
            doc.metadata["doc_type"] = "html"
            normalized.append(doc)
        
        # Step 2: Split ricorsivo per dimensione
        chunks = self._html_chunk_splitter.split_documents(normalized)
        
        if not chunks:
            return []
        
        # Step 3: Genera ID deterministici e inserisci
        chroma_ids = [f"html_{filename}_{i}" for i in range(len(chunks))]
        self._vectorstore.add_documents(chunks, ids=chroma_ids)
        
        return chroma_ids

    # ==========================================================
    # PIPELINE PDF: Parent-Child Chunking
    # ==========================================================

    def index_pdf_list(self, pdf_links_file: Optional[str] = None) -> dict:
        """
        Scarica e indicizza i PDF con architettura Parent-Child.
        
        Flusso per ogni PDF:
        1. Download (se non presente o hash cambiato).
        2. Load pagine con PyPDFLoader.
        3. Split Parent (3000 chars) → salvati nel ParentDocStore.
        4. Split Child (400 chars) da ogni Parent → vettorializzati in Chroma.
        5. Ogni Child porta nei metadati il parent_id per la risalita.
        
        Returns:
            Stats dict.
        """
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
                # --- Download incrementale ---
                filename = self._safe_filename(url)
                local_path = os.path.join(download_dir, filename)
                
                if not self._download_if_needed(url, local_path):
                    stats["skipped"] += 1
                    continue
                
                # --- Hash check ---
                file_hash = IndexRegistry.compute_file_hash(local_path)
                existing = self._registry.get(source_id)
                
                if existing and existing.content_hash == file_hash:
                    stats["skipped"] += 1
                    continue
                
                # Cleanup vecchi chunk se update
                if existing:
                    self._delete_chroma_ids(existing.chroma_ids)
                    self._parent_store.delete_many(existing.parent_ids)
                    stats["updated"] += 1
                else:
                    stats["indexed"] += 1
                
                # --- Parent-Child Chunking ---
                chroma_ids, parent_ids = self._chunk_and_index_pdf(local_path, url)
                
                self._registry.upsert(source_id, IndexEntry(
                    content_hash=file_hash,
                    chroma_ids=chroma_ids,
                    parent_ids=parent_ids,
                ))
                
            except Exception as e:
                logger.error(f"Errore indicizzazione PDF {url}: {e}")
        
        # Pulizia orfani PDF
        pdf_orphans = {
            sid for sid in self._registry.all_source_ids()
            if sid.startswith("pdf:") and sid not in current_sources
        }
        for orphan_id in pdf_orphans:
            entry = self._registry.remove(orphan_id)
            if entry:
                self._delete_chroma_ids(entry.chroma_ids)
                self._parent_store.delete_many(entry.parent_ids)
                stats["orphans_removed"] += 1
        
        self._registry.save()
        logger.info(f"PDF indexing: {stats}")
        return stats

    def _chunk_and_index_pdf(
        self, local_path: str, source_url: str
    ) -> Tuple[List[str], List[str]]:
        """
        Parent-Child chunking di un PDF.
        
        1. Carica tutte le pagine.
        2. Crea Parent chunk (3000 chars) → salvati nel ParentDocStore.
        3. Per ogni Parent, crea Child chunk (400 chars) → vettorializzati in Chroma.
        4. Ogni Child porta parent_id nei metadati.
        
        Returns:
            (chroma_ids dei Child, parent_ids dei Parent)
        """
        # Load
        loader = PyPDFLoader(local_path)
        pages = loader.load()
        
        for page in pages:
            page.metadata["source_url"] = source_url
            page.metadata["doc_type"] = "pdf"
        
        # Step 1: Parent chunking (blocchi ampi)
        parent_chunks = self._pdf_parent_splitter.split_documents(pages)
        
        all_chroma_ids: List[str] = []
        all_parent_ids: List[str] = []
        
        for parent_idx, parent_doc in enumerate(parent_chunks):
            # Genera ID univoco per il Parent
            parent_id = f"parent_{self._safe_filename(source_url)}_{parent_idx}"
            all_parent_ids.append(parent_id)
            
            # Salva il Parent nel docstore (NON in Chroma)
            self._parent_store.store(
                doc_id=parent_id,
                content=parent_doc.page_content,
                metadata=parent_doc.metadata,
            )
            
            # Step 2: Child chunking (sotto-frammenti focalizzati)
            child_chunks = self._pdf_child_splitter.split_documents([parent_doc])
            
            for child_idx, child_doc in enumerate(child_chunks):
                # Inietta il parent_id nei metadati del Child
                child_doc.metadata["parent_id"] = parent_id
                child_doc.metadata["child_index"] = child_idx
            
            # Genera ID deterministici per i Child
            child_ids = [
                f"child_{self._safe_filename(source_url)}_{parent_idx}_{i}"
                for i in range(len(child_chunks))
            ]
            
            if child_chunks:
                self._vectorstore.add_documents(child_chunks, ids=child_ids)
                all_chroma_ids.extend(child_ids)
        
        logger.debug(
            f"PDF '{os.path.basename(local_path)}': "
            f"{len(parent_chunks)} parent → {len(all_chroma_ids)} child chunks"
        )
        
        return all_chroma_ids, all_parent_ids

    # ==========================================================
    # RETRIEVER con risalita Parent
    # ==========================================================

    def get_retriever(self, search_type: Optional[str] = None, k: Optional[int] = None):
        """
        Restituisce il VectorStoreRetriever standard di LangChain.
        
        NOTA: Questo retriever restituisce i Child chunk. Per i PDF,
        il RetrievalEngine deve risalire ai Parent tramite resolve_parents().
        """
        return self._vectorstore.as_retriever(
            search_type=search_type or self._settings.vectorstore.search_type,
            search_kwargs={"k": k or self._settings.vectorstore.search_k},
        )
    
    def resolve_parents(self, documents: List[Document]) -> List[Document]:
        """
        Risale dai Child ai Parent per i documenti PDF.
        
        Per i documenti HTML (che non hanno parent_id), li restituisce invariati.
        Per i Child PDF, recupera il Parent dal docstore e lo restituisce al suo posto,
        preservando il relevance_score del Child nei metadati.
        
        Questo è il punto chiave del Parent-Child pattern:
        - Il Child ha trovato la formula del voto (embedding preciso).
        - Ma al LLM passiamo il Parent con la sezione completa del regolamento
          (contesto per una risposta accurata).
        """
        resolved = []
        seen_parent_ids = set()
        
        for doc in documents:
            parent_id = doc.metadata.get("parent_id")
            
            if parent_id is None:
                # Documento HTML: nessuna risalita necessaria
                resolved.append(doc)
                continue
            
            # Documento PDF: risalita al Parent (deduplicazione per parent_id)
            if parent_id in seen_parent_ids:
                continue
            seen_parent_ids.add(parent_id)
            
            parent_doc = self._parent_store.retrieve(parent_id)
            if parent_doc:
                # Preserva il relevance_score del Child che ha triggerato il match
                if "relevance_score" in doc.metadata:
                    parent_doc.metadata["relevance_score"] = doc.metadata["relevance_score"]
                parent_doc.metadata["matched_via_child"] = True
                resolved.append(parent_doc)
            else:
                # Fallback: se il Parent non esiste, usa il Child
                logger.warning(f"Parent {parent_id} non trovato, fallback al Child")
                resolved.append(doc)
        
        return resolved

    # ==========================================================
    # UTILITÀ
    # ==========================================================

    def _download_if_needed(self, url: str, local_path: str) -> bool:
        """Scarica il PDF solo se non esiste localmente. Restituisce True se scaricato."""
        if os.path.exists(local_path):
            return True  # File presente, procedi con hash check
        
        try:
            response = requests.get(url, timeout=60)
            if response.ok:
                with open(local_path, "wb") as f:
                    f.write(response.content)
                return True
            else:
                logger.warning(f"Download fallito ({response.status_code}): {url}")
                return False
        except Exception as e:
            logger.error(f"Errore download PDF {url}: {e}")
            return False
    
    def _delete_chroma_ids(self, ids: List[str]) -> None:
        """Elimina chunk specifici da Chroma per ID."""
        if not ids:
            return
        try:
            self._vectorstore.delete(ids=ids)
            logger.debug(f"Eliminati {len(ids)} chunk da Chroma")
        except Exception as e:
            logger.warning(f"Errore eliminazione chunk da Chroma: {e}")
    
    @staticmethod
    def _extract_source_url(html_content: str) -> Optional[str]:
        """Estrae l'URL sorgente dai metadati inseriti dal crawler."""
        match = re.search(r"<!--\s*SOURCE:\s*(https?://[^\s]+)\s*-->", html_content)
        return match.group(1).strip() if match else None
    
    @staticmethod
    def _safe_filename(url: str) -> str:
        """Converte un URL in un nome di file sicuro per il filesystem."""
        name = re.sub(r'[<>:"/\\|?*]', '_', url.replace("https://", "").replace("http://", ""))
        return name[:120]  # Troncamento per sicurezza
    
    @property
    def vectorstore(self) -> Chroma:
        """Accesso diretto al vector store."""
        return self._vectorstore
    
    @property
    def parent_store(self) -> ParentDocStore:
        """Accesso diretto al parent doc store."""
        return self._parent_store
