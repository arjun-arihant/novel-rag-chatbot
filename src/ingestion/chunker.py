# Token-based Chunker with Sentence Alignment and Paragraph Fallback

import re
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A text chunk with metadata."""
    content: str
    chapter_number: int
    chapter_title: str
    chunk_index: int
    total_chunks: int
    token_count: int
    start_char: int
    end_char: int
    parent_context: str = ""  # First sentence of chapter for context
    
    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "chapter_number": self.chapter_number,
            "chapter_title": self.chapter_title,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "token_count": self.token_count,
        }


class TokenChunker:
    """
    Token-based chunker with sentence alignment and paragraph fallback.
    
    Target: 400-800 tokens per chunk
    Fallback: If >10% of sentences appear malformed, use paragraph-level chunking
    """
    
    # Sentence-ending patterns
    SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+(?=[A-Z"])')
    
    # Paragraph pattern (double newline)
    PARAGRAPH_SPLIT = re.compile(r'\n\s*\n')
    
    # Malformed sentence indicators
    MALFORMED_INDICATORS = [
        re.compile(r'^\s*[a-z]'),  # Starts with lowercase
        re.compile(r'^.{0,10}$'),   # Too short (<10 chars)
        re.compile(r'[.!?]\s*$', re.NEGATE if hasattr(re, 'NEGATE') else 0),  # Doesn't end properly (fallback check)
    ]
    
    def __init__(
        self,
        target_tokens: int = 500,
        min_tokens: int = 400,
        max_tokens: int = 800,
        overlap_tokens: int = 100,
        malformed_threshold: float = 0.1
    ):
        self.target_tokens = target_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.malformed_threshold = malformed_threshold
        
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count. 
        Rough approximation: ~4 chars per token for English.
        """
        return len(text) // 4
    
    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # First split by obvious sentence boundaries
        sentences = self.SENTENCE_ENDINGS.split(text)
        
        # Clean up
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = self.PARAGRAPH_SPLIT.split(text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def is_malformed_sentence(self, sentence: str) -> bool:
        """Check if a sentence appears malformed."""
        if len(sentence) < 10:
            return True
        if sentence[0].islower() and not sentence.startswith('"'):
            return True
        if not re.search(r'[.!?"\']$', sentence):
            return True
        return False
    
    def calculate_malformed_ratio(self, sentences: List[str]) -> float:
        """Calculate ratio of malformed sentences."""
        if not sentences:
            return 1.0
        malformed = sum(1 for s in sentences if self.is_malformed_sentence(s))
        return malformed / len(sentences)
    
    def chunk_by_sentences(
        self,
        sentences: List[str],
        chapter_number: int,
        chapter_title: str,
        parent_context: str = ""
    ) -> List[Chunk]:
        """Create chunks from sentences, respecting token limits."""
        chunks = []
        current_sentences: List[str] = []
        current_tokens = 0
        char_offset = 0
        
        for sentence in sentences:
            sentence_tokens = self.estimate_tokens(sentence)
            
            # If single sentence exceeds max, split it
            if sentence_tokens > self.max_tokens:
                # First, flush current buffer
                if current_sentences:
                    chunk_text = ' '.join(current_sentences)
                    chunks.append(self._create_chunk(
                        chunk_text, chapter_number, chapter_title,
                        len(chunks), char_offset, parent_context
                    ))
                    char_offset += len(chunk_text) + 1
                    current_sentences = []
                    current_tokens = 0
                
                # Split long sentence by words
                words = sentence.split()
                word_chunk = []
                word_tokens = 0
                for word in words:
                    wt = self.estimate_tokens(word + ' ')
                    if word_tokens + wt > self.target_tokens and word_chunk:
                        chunk_text = ' '.join(word_chunk)
                        chunks.append(self._create_chunk(
                            chunk_text, chapter_number, chapter_title,
                            len(chunks), char_offset, parent_context
                        ))
                        char_offset += len(chunk_text) + 1
                        word_chunk = []
                        word_tokens = 0
                    word_chunk.append(word)
                    word_tokens += wt
                    
                if word_chunk:
                    current_sentences = [' '.join(word_chunk)]
                    current_tokens = word_tokens
                continue
            
            # Would adding this sentence exceed target?
            if current_tokens + sentence_tokens > self.target_tokens and current_sentences:
                # Flush current buffer
                chunk_text = ' '.join(current_sentences)
                chunks.append(self._create_chunk(
                    chunk_text, chapter_number, chapter_title,
                    len(chunks), char_offset, parent_context
                ))
                char_offset += len(chunk_text) + 1
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_sentences)
                current_sentences = overlap_sentences
                current_tokens = sum(self.estimate_tokens(s) for s in overlap_sentences)
            
            current_sentences.append(sentence)
            current_tokens += sentence_tokens
        
        # Final chunk
        if current_sentences and current_tokens >= self.min_tokens // 2:
            chunk_text = ' '.join(current_sentences)
            chunks.append(self._create_chunk(
                chunk_text, chapter_number, chapter_title,
                len(chunks), char_offset, parent_context
            ))
        
        # Update total_chunks
        for i, chunk in enumerate(chunks):
            chunk.total_chunks = len(chunks)
            
        return chunks
    
    def chunk_by_paragraphs(
        self,
        paragraphs: List[str],
        chapter_number: int,
        chapter_title: str,
        parent_context: str = ""
    ) -> List[Chunk]:
        """Create chunks from paragraphs (fallback mode)."""
        chunks = []
        current_paragraphs: List[str] = []
        current_tokens = 0
        char_offset = 0
        
        for para in paragraphs:
            para_tokens = self.estimate_tokens(para)
            
            # Would adding this paragraph exceed target?
            if current_tokens + para_tokens > self.target_tokens and current_paragraphs:
                chunk_text = '\n\n'.join(current_paragraphs)
                chunks.append(self._create_chunk(
                    chunk_text, chapter_number, chapter_title,
                    len(chunks), char_offset, parent_context
                ))
                char_offset += len(chunk_text) + 2
                current_paragraphs = []
                current_tokens = 0
            
            current_paragraphs.append(para)
            current_tokens += para_tokens
        
        # Final chunk
        if current_paragraphs:
            chunk_text = '\n\n'.join(current_paragraphs)
            chunks.append(self._create_chunk(
                chunk_text, chapter_number, chapter_title,
                len(chunks), char_offset, parent_context
            ))
        
        # Update total_chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
            
        return chunks
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get last N sentences for overlap based on token count."""
        overlap = []
        tokens = 0
        
        for sentence in reversed(sentences):
            st = self.estimate_tokens(sentence)
            if tokens + st > self.overlap_tokens:
                break
            overlap.insert(0, sentence)
            tokens += st
            
        return overlap
    
    def _create_chunk(
        self,
        text: str,
        chapter_number: int,
        chapter_title: str,
        index: int,
        start_char: int,
        parent_context: str
    ) -> Chunk:
        """Create a Chunk object."""
        return Chunk(
            content=text,
            chapter_number=chapter_number,
            chapter_title=chapter_title,
            chunk_index=index,
            total_chunks=0,  # Will be updated later
            token_count=self.estimate_tokens(text),
            start_char=start_char,
            end_char=start_char + len(text),
            parent_context=parent_context
        )
    
    def chunk_chapter(
        self,
        content: str,
        chapter_number: int,
        chapter_title: str
    ) -> List[Chunk]:
        """
        Chunk a chapter with automatic fallback.
        
        1. Try sentence-level chunking
        2. If >10% malformed, fall back to paragraph-level
        """
        # Get parent context (first sentence)
        sentences = self.split_sentences(content)
        parent_context = sentences[0][:200] if sentences else ""
        
        # Check malformed ratio
        malformed_ratio = self.calculate_malformed_ratio(sentences)
        
        if malformed_ratio > self.malformed_threshold:
            logger.warning(
                f"Chapter {chapter_number}: {malformed_ratio:.1%} malformed sentences, "
                f"falling back to paragraph chunking"
            )
            paragraphs = self.split_paragraphs(content)
            return self.chunk_by_paragraphs(
                paragraphs, chapter_number, chapter_title, parent_context
            )
        
        return self.chunk_by_sentences(
            sentences, chapter_number, chapter_title, parent_context
        )
