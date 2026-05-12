"""
Registro incrementale per l'indicizzazione.

Mantiene una mappa persistente:
  source_identifier → (content_hash, [chroma_ids], [parent_ids], collection_name)

Pattern: Repository (DDD) — incapsula l'accesso al registro persistente.
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
    """
    Record di un singolo documento indicizzato.

    Campi:
      content_hash: SHA-256 del contenuto (per deduplication incrementale).
      chroma_ids: Lista degli ID dei chunk in Chroma (per cleanup).
      parent_ids: Lista degli ID dei parent nel docstore (per cleanup Parent-Child).
      collection_name: Nome della collection Chroma in cui il documento è indicizzato.
                       Corrisponde a CollectionTarget.value (es. "persone").
    """
    content_hash: str
    chroma_ids: List[str] = field(default_factory=list)
    parent_ids: List[str] = field(default_factory=list)
    collection_name: str = ""


class IndexRegistry:
    """
    Registro persistente su file JSON per tracciare lo stato di indicizzazione.
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
        parent_dir = os.path.dirname(self._path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(
                {key: asdict(entry) for key, entry in self._entries.items()},
                f, indent=2, ensure_ascii=False,
            )

    def get(self, source_id: str) -> Optional[IndexEntry]:
        return self._entries.get(source_id)

    def upsert(self, source_id: str, entry: IndexEntry) -> None:
        self._entries[source_id] = entry

    def remove(self, source_id: str) -> Optional[IndexEntry]:
        return self._entries.pop(source_id, None)

    def all_source_ids(self) -> set:
        return set(self._entries.keys())

    def clear_all(self) -> None:
        self._entries.clear()
        logger.info("Registro svuotato completamente")

    @staticmethod
    def compute_hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def compute_file_hash(filepath: str) -> str:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                h.update(block)
        return h.hexdigest()