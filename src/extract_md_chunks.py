#!/usr/bin/env python3
"""
extract_md_chunks.py — Utility di ispezione per i Vector Store.

Estrae tutti i chunk generati dai file Markdown statici (es. diem_info_generali.md)
e li formatta in un file di output testuale per verificarne l'integrità, 
la context-injection e i metadati.
"""

import os
import sys
import json
import logging
from typing import Dict, Any

# ============================================================
# PATH SETUP
# ============================================================
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from config.settings import load_settings
from ingestion.indexer import KnowledgeBaseIndexer
from ingestion.router import CollectionTarget

# Configurazione logging base per lo script
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def export_markdown_chunks(output_filepath: str = "md_chunks_export.txt") -> None:
    """
    Si connette alla collection DIPARTIMENTO e recupera i chunk
    filtrati per doc_type == "md". Salva il risultato su file.
    """
    logger.info("Caricamento configurazioni e connessione al Vector Store...")
    settings = load_settings()
    
    # Inizializziamo l'indexer (che si occuperà di fare il bind con ChromaDB)
    # Nota: Questo caricherà in memoria anche il modello di embedding, 
    # è normale un leggero delay di startup.
    indexer = KnowledgeBaseIndexer(settings)
    
    # Sappiamo dal router che i file .md vanno nella collection DIPARTIMENTO
    target_collection = CollectionTarget.DIPARTIMENTO
    vectorstore = indexer._collections[target_collection]
    
    logger.info(f"Query su ChromaDB collection '{target_collection.value}' per doc_type='md'...")
    
    try:
        # Metodo nativo di Langchain/Chroma per estrarre documenti filtrati
        results: Dict[str, Any] = vectorstore.get(
            where={"doc_type": "md"}
        )
    except Exception as e:
        logger.error(f"Errore durante l'interrogazione del Vector Store: {e}")
        sys.exit(1)

    ids = results.get("ids", [])
    metadatas = results.get("metadatas", [])
    documents = results.get("documents", [])

    if not ids:
        logger.warning("Nessun chunk di tipo Markdown trovato nel Vector Store.")
        logger.info("Assicurati di aver eseguito l'ingestion_main.py dopo aver inserito il file .md!")
        return

    logger.info(f"Trovati {len(ids)} chunk Markdown. Scrittura in corso su {output_filepath}...")

    # Scrittura del file di output formattato per human-reading
    try:
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"ESPORTAZIONE CHUNK MARKDOWN\n")
            f.write(f"Totale chunk recuperati: {len(ids)}\n")
            f.write(f"Collection target: {target_collection.value}\n")
            f.write("=" * 80 + "\n\n")

            for i in range(len(ids)):
                chunk_id = ids[i]
                meta = metadatas[i]
                content = documents[i]

                f.write(f"--- CHUNK [{i+1}/{len(ids)}] | ID: {chunk_id} ---\n\n")
                
                f.write("[METADATI]\n")
                f.write(json.dumps(meta, indent=2, ensure_ascii=False) + "\n\n")
                
                f.write("[CONTENUTO DEL CHUNK]\n")
                f.write(content + "\n\n")
                
                f.write("-" * 80 + "\n\n")
                
        logger.info(f"Esportazione completata con successo! File salvato in: {os.path.abspath(output_filepath)}")
        
    except IOError as e:
        logger.error(f"Errore di I/O durante il salvataggio del file: {e}")


if __name__ == "__main__":
    # Esegui lo script generando il file nella directory corrente da cui è lanciato
    export_markdown_chunks(output_filepath="md_chunks_export.txt")