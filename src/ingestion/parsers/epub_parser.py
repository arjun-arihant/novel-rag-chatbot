# EPUB Parser

import re
import logging
from pathlib import Path

from .base import BaseParser, Chapter

logger = logging.getLogger(__name__)


class EpubParser(BaseParser):
    """
    Parser for EPUB (.epub) ebook files.
    
    Uses ebooklib for proper EPUB parsing and BeautifulSoup
    for HTML content extraction.
    """
    
    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".epub"
    
    def parse(self, file_path: Path) -> list[Chapter]:
        """Parse an EPUB file into chapters."""
        try:
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup
        except ImportError as e:
            raise ImportError(
                "EPUB parsing requires ebooklib and beautifulsoup4. "
                "Install with: pip install ebooklib beautifulsoup4"
            ) from e
        
        book = epub.read_epub(str(file_path))
        chapters = []
        chapter_num = 0
        
        # Get spine order for proper chapter sequence
        spine_items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
        
        for item in spine_items:
            content = item.get_content()
            soup = BeautifulSoup(content, "html.parser")
            
            # Extract text
            text = soup.get_text(separator="\n")
            text = self._clean_text(text)
            
            if not text or len(text) < 100:
                # Skip very short items (likely navigation, etc.)
                continue
            
            chapter_num += 1
            
            # Try to extract title from HTML
            title = self._extract_title(soup, chapter_num)
            
            chapters.append(Chapter(
                number=chapter_num,
                title=title,
                content=text
            ))
        
        # If no chapters, try alternative extraction
        if not chapters:
            chapters = self._fallback_extraction(book)
        
        logger.info(f"Extracted {len(chapters)} chapters from EPUB")
        return chapters
    
    def _extract_title(self, soup, default_num: int) -> str:
        """Extract chapter title from HTML."""
        # Try common heading elements
        for tag in ["h1", "h2", "h3", "title"]:
            heading = soup.find(tag)
            if heading:
                title = heading.get_text().strip()
                # Check if it looks like a chapter title
                if title and len(title) < 200:
                    return title
        
        return f"Chapter {default_num}"
    
    def _fallback_extraction(self, book) -> list[Chapter]:
        """Fallback extraction if spine-based fails."""
        try:
            from ebooklib import epub
            from bs4 import BeautifulSoup
        except ImportError:
            return []
        
        chapters = []
        chapter_num = 0
        
        # Try getting all text items
        for item in book.get_items():
            if item.get_type() == 9:  # ITEM_DOCUMENT
                try:
                    soup = BeautifulSoup(item.get_content(), "html.parser")
                    text = soup.get_text(separator="\n")
                    text = self._clean_text(text)
                    
                    if text and len(text) > 200:
                        chapter_num += 1
                        chapters.append(Chapter(
                            number=chapter_num,
                            title=f"Section {chapter_num}",
                            content=text
                        ))
                except Exception:
                    continue
        
        return chapters
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text from EPUB."""
        # Remove excessive whitespace
        lines = []
        for line in text.split("\n"):
            line = line.strip()
            if line:
                lines.append(line)
        
        text = "\n".join(lines)
        
        # Remove multiple consecutive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
