# Novel Loader - Load and parse novel text files

import re
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Chapter:
    """Represents a chapter from the novel."""
    number: int
    title: str
    content: str
    start_pos: int  # Position in original text
    end_pos: int


class NovelLoader:
    """Load and parse novel text files."""
    
    # Multiple chapter patterns for flexibility
    CHAPTER_PATTERNS = [
        # "Chapter 1: Title" or "Chapter 1 - Title"
        re.compile(r'^Chapter\s+(\d+)\s*[:\-–—]\s*(.+?)$', re.MULTILINE | re.IGNORECASE),
        # "CHAPTER 1" (uppercase, no title)
        re.compile(r'^CHAPTER\s+(\d+)\s*$', re.MULTILINE),
        # "Chapter One: Title" (word numbers)
        re.compile(r'^Chapter\s+(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|Eleven|Twelve)\s*[:\-–—]\s*(.+?)$', 
                   re.MULTILINE | re.IGNORECASE),
    ]
    
    WORD_TO_NUM = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12
    }
    
    def __init__(self, path: Path):
        self.path = Path(path)
        self.raw_text: str = ""
        self.chapters: List[Chapter] = []
        
    def load(self) -> str:
        """Load novel text from file."""
        if not self.path.exists():
            raise FileNotFoundError(f"Novel not found: {self.path}")
            
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                self.raw_text = self.path.read_text(encoding=encoding)
                logger.info(f"Loaded novel: {len(self.raw_text)} chars ({encoding})")
                return self.raw_text
            except UnicodeDecodeError:
                continue
                
        raise ValueError(f"Could not decode {self.path} with any known encoding")
    
    def parse_chapters(self) -> List[Chapter]:
        """Parse chapters from loaded text."""
        if not self.raw_text:
            self.load()
            
        # Try each pattern until one works
        for pattern in self.CHAPTER_PATTERNS:
            matches = list(pattern.finditer(self.raw_text))
            if matches:
                self.chapters = self._extract_chapters(matches, pattern)
                logger.info(f"Found {len(self.chapters)} chapters")
                return self.chapters
                
        # No chapters found - treat entire text as one chapter
        logger.warning("No chapter markers found, treating as single chapter")
        self.chapters = [
            Chapter(
                number=1,
                title="Full Text",
                content=self.raw_text.strip(),
                start_pos=0,
                end_pos=len(self.raw_text)
            )
        ]
        return self.chapters
    
    def _extract_chapters(self, matches: List[re.Match], pattern: re.Pattern) -> List[Chapter]:
        """Extract chapter content from regex matches."""
        chapters = []
        
        for i, match in enumerate(matches):
            # Get chapter number
            num_str = match.group(1)
            if num_str.lower() in self.WORD_TO_NUM:
                num = self.WORD_TO_NUM[num_str.lower()]
            else:
                num = int(num_str)
            
            # Get title (may not exist in some patterns)
            try:
                title = match.group(2).strip() if match.lastindex >= 2 else f"Chapter {num}"
            except:
                title = f"Chapter {num}"
            
            # Get content (from end of this match to start of next, or end of text)
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(self.raw_text)
            content = self.raw_text[start:end].strip()
            
            chapters.append(Chapter(
                number=num,
                title=title,
                content=content,
                start_pos=match.start(),
                end_pos=end
            ))
            
        return chapters
    
    def get_chapter(self, number: int) -> Optional[Chapter]:
        """Get a specific chapter by number."""
        for chapter in self.chapters:
            if chapter.number == number:
                return chapter
        return None
    
    def get_full_text(self) -> str:
        """Get the full novel text."""
        return self.raw_text
    
    def get_stats(self) -> Dict:
        """Get statistics about the loaded novel."""
        return {
            "total_chars": len(self.raw_text),
            "total_words": len(self.raw_text.split()),
            "num_chapters": len(self.chapters),
            "avg_chapter_length": (
                sum(len(c.content) for c in self.chapters) // len(self.chapters)
                if self.chapters else 0
            )
        }
