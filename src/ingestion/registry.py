"""Registry incrementale per il tracciamento dei documenti indicizzati.

Gestisce la persistenza su disco dello stato di indicizzazione,
consentendo operazioni incrementali di upsert, rimozione e
verifica tramite hash del contenuto.
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
    """Singola voce del registro di indicizzazione.

    Attributes:
        content_hash: Hash SHA-256 del contenuto indicizzato.
        chroma_ids: Lista degli ID dei chunk nel vector store Chroma.
        parent_ids: Lista degli ID dei documenti parent (per Parent-Child).
        collection_name: Nome della collezione di appartenenza.
    """

    content_hash: str
    chroma_ids: List[str] = field(default_factory=list)
    parent_ids: List[str] = field(default_factory=list)
    collection_name: str = ""


class IndexRegistry:
    """Registro persistente per il tracciamento incrementale dei documenti indicizzati.

    Mantiene una mappa source_id -> IndexEntry su disco in formato JSON,
    permettendo di identificare documenti nuovi, aggiornati o rimossi
    tra esecuzioni successive della pipeline di ingestion.
    """

    def __init__(self, registry_path: str):
        """Inizializza il registro caricando lo stato da disco se presente.

        Args:
            registry_path: Percorso del file JSON di persistenza.
        """
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
                logger.info(
                    "Registro caricato: %d documenti tracciati",
                    len(self._entries),
                )
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning("Registro corrotto, reset: %s", e)
                self._entries = {}
        else:
            logger.info("Nessun registro esistente, inizializzazione vuota")

    def save(self) -> None:
        """Persiste il registro su disco in formato JSON."""
        parent_dir = os.path.dirname(self._path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(
                {key: asdict(entry) for key, entry in self._entries.items()},
                f, indent=2, ensure_ascii=False,
            )

    def get(self, source_id: str) -> Optional[IndexEntry]:
        """Restituisce l'entry associata al source_id indicato.

        Args:
            source_id: Identificativo univoco del documento sorgente.

        Returns:
            IndexEntry corrispondente o None se non presente.
        """
        return self._entries.get(source_id)

    def upsert(self, source_id: str, entry: IndexEntry) -> None:
        """Inserisce o aggiorna l'entry per il source_id indicato.

        Args:
            source_id: Identificativo univoco del documento sorgente.
            entry: Nuova voce del registro.
        """
        self._entries[source_id] = entry

    def remove(self, source_id: str) -> Optional[IndexEntry]:
        """Rimuove e restituisce l'entry associata al source_id indicato.

        Args:
            source_id: Identificativo univoco del documento sorgente.

        Returns:
            IndexEntry rimossa o None se non presente.
        """
        return self._entries.pop(source_id, None)

    def all_source_ids(self) -> set:
        """Restituisce l'insieme di tutti i source_id tracciati.

        Returns:
            Set di stringhe con tutti gli identificativi registrati.
        """
        return set(self._entries.keys())

    def clear_all(self) -> None:
        """Svuota completamente il registro in memoria."""
        self._entries.clear()
        logger.info("Registro svuotato completamente")

    @staticmethod
    def compute_hash(content: str) -> str:
        """Calcola l'hash SHA-256 di una stringa di contenuto.

        Args:
            content: Testo di cui calcolare l'hash.

        Returns:
            Hash esadecimale SHA-256.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def compute_file_hash(filepath: str) -> str:
        """Calcola l'hash SHA-256 di un file letto in blocchi.

        Args:
            filepath: Percorso del file su disco.

        Returns:
            Hash esadecimale SHA-256.
        """
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                h.update(block)
        return h.hexdigest()