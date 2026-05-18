from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

class CleaningRule(ABC):
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def requires_content(self) -> bool:
        return True

    @abstractmethod
    def should_delete(self, filepath: Path, content: Optional[str] = None) -> bool:
        pass

class PdfFilterRule(ABC):
    
    @property
    @abstractmethod
    def name(self) -> str: pass

    @abstractmethod
    def should_discard(self, url: str) -> bool:
        pass