from dataclasses import dataclass
from typing import List

@dataclass
class Document:
    """Rappresentazione unificata di un documento nel sistema."""
    page_content: str
    metadata: dict


@dataclass
class RetrievalResult:
    """Risultato di una query al retrieval engine, con tracciabilità delle fonti."""
    documents: List[Document]
    query_used: str  # La query effettivamente usata (potrebbe essere riscritta)
