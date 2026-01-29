# PDF Parser

import re
import logging
from pathlib import Path

from .base import BaseParser, Chapter

logger = logging.getLogger(__name__)


class PdfParser(BaseParser):
    """
    Parser for PDF (.pdf) files.
    
    Uses PyMuPDF (fitz) for text extraction. Attempts to detect
    chapter boundaries using TOC or heading patterns.
    """
    
    # Patterns to detect chapter headings in text
    CHAPTER_PATTERNS = [
        r'^Chapter\s+(\d+)\s*[:\-–—]?\s*(.*)$',
        r'^CHAPTER\s+(\d+)\s*[:\-–—]?\s*(.*)$',
        r'^(\d+)\.\s+(.+)$',  # "1. Introduction"
        r'^Part\s+(\d+)\s*[:\-–—]\s*(.+)$',
    ]
    
    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".pdf"
    
    def parse(self, file_path: Path) -> list[Chapter]:
        """Parse a PDF file into chapters."""
        try:
            import fitz  # PyMuPDF
        except ImportError as e:
            raise ImportError(
                "PDF parsing requires PyMuPDF. "
                "Install with: pip install PyMuPDF"
            ) from e
        
        doc = fitz.open(file_path)
        
        # Try TOC-based extraction first
        toc = doc.get_toc()
        if toc:
            chapters = self._extract_from_toc(doc, toc)
            if chapters:
                logger.info(f"Extracted {len(chapters)} chapters from PDF TOC")
                return chapters
        
        # Fallback to pattern-based detection
        chapters = self._extract_from_patterns(doc)
        
        # If still no chapters, treat as single document
        if not chapters:
            full_text = self._extract_all_text(doc)
            chapters = [Chapter(
                number=1,
                title="Full Document",
                content=full_text
            )]
        
        doc.close()
        logger.info(f"Extracted {len(chapters)} chapters from PDF")
        return chapters
    
    def _extract_from_toc(self, doc, toc) -> list[Chapter]:
        """Extract chapters using PDF table of contents."""
        chapters = []
        
        for i, (level, title, page_num) in enumerate(toc):
            if level > 2:  # Skip deep nesting
                continue
            
            # Determine page range
            start_page = page_num - 1  # 0-indexed
            if i + 1 < len(toc):
                end_page = toc[i + 1][2] - 1
            else:
                end_page = len(doc)
            
            # Extract text from page range
            content = []
            for page_idx in range(max(0, start_page), min(end_page, len(doc))):
                page = doc[page_idx]
                content.append(page.get_text())
            
            text = "\n".join(content)
            text = self._clean_text(text)
            
            if text and len(text) > 100:
                chapters.append(Chapter(
                    number=len(chapters) + 1,
                    title=title,
                    content=text
                ))
        
        return chapters
    
    def _extract_from_patterns(self, doc) -> list[Chapter]:
        """Extract chapters by detecting heading patterns in text."""
        all_text = self._extract_all_text(doc)
        lines = all_text.split("\n")
        
        chapters = []
        current_chapter = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            match = self._match_chapter_heading(line)
            
            if match:
                # Save previous chapter
                if current_chapter and current_content:
                    content = "\n".join(current_content).strip()
                    if len(content) > 100:
                        chapters.append(Chapter(
                            number=current_chapter[0],
                            title=current_chapter[1],
                            content=content
                        ))
                
                current_chapter = match
                current_content = []
            else:
                current_content.append(line)
        
        # Save last chapter
        if current_chapter and current_content:
            content = "\n".join(current_content).strip()
            if len(content) > 100:
                chapters.append(Chapter(
                    number=current_chapter[0],
                    title=current_chapter[1],
                    content=content
                ))
        
        return chapters
    
    def _match_chapter_heading(self, line: str) -> tuple[int, str] | None:
        """Try to match a line as a chapter heading."""
        for pattern in self.CHAPTER_PATTERNS:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                num_str = match.group(1)
                title = match.group(2) if len(match.groups()) > 1 else ""
                
                try:
                    num = int(num_str)
                    return (num, title.strip() if title else f"Chapter {num}")
                except ValueError:
                    continue
        
        return None
    
    def _extract_all_text(self, doc) -> str:
        """Extract all text from PDF."""
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        return "\n".join(text_parts)
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted PDF text."""
        # Remove page numbers and headers/footers (common patterns)
        lines = []
        for line in text.split("\n"):
            line = line.strip()
            
            # Skip likely page numbers
            if re.match(r'^\d+$', line):
                continue
            
            # Skip very short lines that might be headers/footers
            if len(line) < 3:
                continue
            
            lines.append(line)
        
        text = "\n".join(lines)
        
        # Remove multiple consecutive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
