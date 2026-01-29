# Plain Text Parser

import re
import logging
from pathlib import Path

from .base import BaseParser, Chapter

logger = logging.getLogger(__name__)


class TxtParser(BaseParser):
    """
    Parser for plain text (.txt) novel files.
    
    Detects chapters using common patterns:
    - "Chapter X" / "Chapter X: Title"
    - "CHAPTER X" / "CHAPTER X - Title"
    - Roman numerals: "Chapter I", "Chapter II"
    """
    
    # Patterns for chapter detection (ordered by specificity)
    CHAPTER_PATTERNS = [
        # "Chapter 1: The Beginning" or "Chapter 1 - The Beginning"
        r'^Chapter\s+(\d+)\s*[:\-–—]\s*(.+)$',
        # "Chapter One: Title"
        r'^Chapter\s+(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|'
        r'Eleven|Twelve|Thirteen|Fourteen|Fifteen|Sixteen|Seventeen|'
        r'Eighteen|Nineteen|Twenty|[A-Z][a-z]+)\s*[:\-–—]\s*(.+)$',
        # "CHAPTER 1" (uppercase)
        r'^CHAPTER\s+(\d+)\s*[:\-–—]?\s*(.*)$',
        # "Chapter 1" alone
        r'^Chapter\s+(\d+)\s*$',
        # Roman numerals: "Chapter I", "Chapter II"
        r'^Chapter\s+([IVXLC]+)\s*[:\-–—]?\s*(.*)$',
        # Part and Chapter: "Part 1, Chapter 2"
        r'^Part\s+\d+\s*,?\s*Chapter\s+(\d+)\s*[:\-–—]?\s*(.*)$',
    ]
    
    WORD_TO_NUM = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
        'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
        'nineteen': 19, 'twenty': 20
    }
    
    ROMAN_TO_NUM = {
        'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
        'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10,
        'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14, 'XV': 15,
        'XVI': 16, 'XVII': 17, 'XVIII': 18, 'XIX': 19, 'XX': 20,
        'XXI': 21, 'XXII': 22, 'XXIII': 23, 'XXIV': 24, 'XXV': 25
    }
    
    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".txt"
    
    def parse(self, file_path: Path) -> list[Chapter]:
        """Parse a plain text file into chapters."""
        # Try different encodings
        text = None
        for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                text = file_path.read_text(encoding=encoding)
                logger.info(f"Loaded {file_path.name} with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if text is None:
            raise ValueError(f"Could not decode file: {file_path}")
        
        return self._extract_chapters(text)
    
    def _extract_chapters(self, text: str) -> list[Chapter]:
        """Extract chapters from text content."""
        lines = text.split("\n")
        chapters = []
        current_chapter = None
        current_content = []
        
        for line in lines:
            chapter_match = self._match_chapter(line.strip())
            
            if chapter_match:
                # Save previous chapter
                if current_chapter is not None:
                    content = "\n".join(current_content).strip()
                    if content:
                        chapters.append(Chapter(
                            number=current_chapter[0],
                            title=current_chapter[1],
                            content=content
                        ))
                
                current_chapter = chapter_match
                current_content = []
            else:
                current_content.append(line)
        
        # Save last chapter
        if current_chapter is not None:
            content = "\n".join(current_content).strip()
            if content:
                chapters.append(Chapter(
                    number=current_chapter[0],
                    title=current_chapter[1],
                    content=content
                ))
        
        # If no chapters found, treat entire text as one chapter
        if not chapters:
            logger.warning("No chapter markers found, treating as single chapter")
            chapters.append(Chapter(
                number=1,
                title="Full Text",
                content=self._clean_text(text)
            ))
        
        logger.info(f"Extracted {len(chapters)} chapters")
        return chapters
    
    def _match_chapter(self, line: str) -> tuple[int, str] | None:
        """
        Try to match a line as a chapter heading.
        
        Returns:
            Tuple of (chapter_number, chapter_title) or None
        """
        for pattern in self.CHAPTER_PATTERNS:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                groups = match.groups()
                num_str = groups[0]
                title = groups[1] if len(groups) > 1 else ""
                
                # Convert to number
                if num_str.isdigit():
                    num = int(num_str)
                elif num_str.upper() in self.ROMAN_TO_NUM:
                    num = self.ROMAN_TO_NUM[num_str.upper()]
                elif num_str.lower() in self.WORD_TO_NUM:
                    num = self.WORD_TO_NUM[num_str.lower()]
                else:
                    continue
                
                return (num, title.strip() if title else f"Chapter {num}")
        
        return None
