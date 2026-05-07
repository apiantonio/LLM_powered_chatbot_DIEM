"""
Registro incrementale per l'indicizzazione.

Mantiene una mappa persistente: source_identifier → (content_hash, [chroma_ids], [parent_ids])
Questo permette di:
- Saltare documenti non modificati (hash identico).
- Eliminare chunk orfani quando un documento cambia o scompare.
- Eseguire upsert logici senza ri-indicizzare l'intero corpus.

Pattern: Repository (DDD) — incapsula l'accesso al registro persistente.

KPI Impact: Knowledge Freshness (aggiornamento continuo senza downtime).
"""

import json
import hashlib
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class IndexEntry:
    """Record di un singolo documento indicizzato."""
    content_hash: str
    chroma_ids: List[str] = field(default_factory=list)
    parent_ids: List[str] = field(default_factory=list)


class IndexRegistry:
    """
    Registro persistente su file JSON per tracciare lo stato di indicizzazione.
    
    Flusso di aggiornamento incrementale:
    1. Calcola hash del documento corrente.
    2. Confronta con l'hash nel registro.
    3a. Hash identico → skip (il documento non è cambiato).
    3b. Hash diverso → elimina vecchi chunk da Chroma/ParentStore, re-indicizza, aggiorna registro.
    3c. URL assente nel registro → prima indicizzazione.
    4. URL presente nel registro ma assente nel crawl → elimina chunk orfani.
    """
    
    def __init__(self, registry_path: str):
        self._path = registry_path
        self._entries: Dict[str, IndexEntry] = {}
        self._load()
    
    def _load(self) -> None:
        """Carica il registro da disco."""
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self._entries = {
                    key: IndexEntry(**val) for key, val in raw.items()
                }
                logger.info(f"Registro caricato: {len(self._entries)} documenti tracciati")
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Registro corrotto, reset: {e}")
                self._entries = {}
        else:
            logger.info("Nessun registro esistente, inizializzazione vuota")
    
    def save(self) -> None:
        """Persiste il registro su disco."""
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(
                {key: asdict(entry) for key, entry in self._entries.items()},
                f, indent=2, ensure_ascii=False,
            )
    
    def get(self, source_id: str) -> Optional[IndexEntry]:
        """Recupera l'entry per un dato source identifier."""
        return self._entries.get(source_id)
    
    def upsert(self, source_id: str, entry: IndexEntry) -> None:
        """Inserisce o aggiorna un'entry."""
        self._entries[source_id] = entry
    
    def remove(self, source_id: str) -> Optional[IndexEntry]:
        """Rimuove un'entry e la restituisce (per cleanup dei chunk)."""
        return self._entries.pop(source_id, None)
    
    def all_source_ids(self) -> set:
        """Restituisce tutti i source_id attualmente tracciati."""
        return set(self._entries.keys())
    
    @staticmethod
    def compute_hash(content: str) -> str:
        """Calcola SHA-256 del contenuto testuale."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
    
    @staticmethod
    def compute_file_hash(filepath: str) -> str:
        """Calcola SHA-256 di un file binario (per i PDF)."""
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                h.update(block)
        return h.hexdigest()
