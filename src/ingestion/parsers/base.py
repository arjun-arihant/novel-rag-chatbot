# Base Parser Interface

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chapter:
    """Represents a chapter extracted from a novel."""
    number: int
    title: str
    content: str
    content_hash: str = field(default="")
    
    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute content hash for change detection."""
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]


class BaseParser(ABC):
    """
    Abstract base class for file format parsers.
    
    Implementations must:
    1. Parse the file into a list of Chapter objects
    2. Detect chapter boundaries appropriately for the format
    3. Handle encoding issues gracefully
    """
    
    @abstractmethod
    def parse(self, file_path: Path) -> list[Chapter]:
        """
        Parse a file into chapters.
        
        Args:
            file_path: Path to the novel file
            
        Returns:
            List of Chapter objects
        """
        pass
    
    @abstractmethod
    def supports(self, file_path: Path) -> bool:
        """
        Check if this parser supports the given file.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if this parser can handle the file
        """
        pass
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Normalize whitespace
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            line = line.strip()
            if line:
                cleaned.append(line)
        return "\n".join(cleaned)
