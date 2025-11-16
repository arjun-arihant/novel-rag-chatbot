"""
Smart chunker with sentence awareness and semantic boundaries
"""
import re
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class SmartChunker:
    """Intelligent text chunker that respects sentence and paragraph boundaries."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200,
                 respect_sentences: bool = True, min_chunk_size: int = 100):
        """
        Initialize smart chunker.

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            respect_sentences: Try to split at sentence boundaries
            min_chunk_size: Minimum acceptable chunk size
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.respect_sentences = respect_sentences
        self.min_chunk_size = min_chunk_size

        # Initialize base splitter
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        if not self.respect_sentences:
            return self.base_splitter.split_text(text)

        return self._smart_split(text)

    def _smart_split(self, text: str) -> List[str]:
        """
        Smart split that respects sentence boundaries.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        # Split into sentences
        sentences = self._split_into_sentences(text)

        chunks = []
        current_chunk = []
        current_length = 0

        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)

            # If single sentence is too long, split it
            if sentence_length > self.chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0

                # Split long sentence
                sub_chunks = self.base_splitter.split_text(sentence)
                chunks.extend(sub_chunks)
                continue

            # Check if adding this sentence exceeds chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))

                # Add overlap from previous chunk
                overlap_text = " ".join(current_chunk)
                if len(overlap_text) > self.chunk_overlap:
                    # Start new chunk with overlap
                    overlap_sentences = self._get_overlap_sentences(current_chunk, self.chunk_overlap)
                    current_chunk = overlap_sentences
                    current_length = sum(len(s) for s in current_chunk)
                else:
                    current_chunk = []
                    current_length = 0

            current_chunk.append(sentence)
            current_length += sentence_length

        # Add remaining chunk
        if current_chunk and current_length >= self.min_chunk_size:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Use regex to split on sentence boundaries while preserving the delimiter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _get_overlap_sentences(self, sentences: List[str], target_length: int) -> List[str]:
        """
        Get sentences for overlap from end of chunk.

        Args:
            sentences: List of sentences
            target_length: Target overlap length

        Returns:
            List of sentences for overlap
        """
        overlap = []
        current_length = 0

        for sentence in reversed(sentences):
            sentence_len = len(sentence)
            if current_length + sentence_len <= target_length:
                overlap.insert(0, sentence)
                current_length += sentence_len
            else:
                break

        return overlap

    def create_documents(self, texts: List[str], metadatas: List[dict] = None) -> List[Document]:
        """
        Create documents from texts with metadata.

        Args:
            texts: List of texts to chunk
            metadatas: Optional metadata for each text

        Returns:
            List of Document objects
        """
        if metadatas is None:
            metadatas = [{}] * len(texts)

        documents = []
        for text, metadata in zip(texts, metadatas):
            chunks = self.split_text(text)
            for i, chunk in enumerate(chunks):
                # Add chunk index to metadata
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_index'] = i
                chunk_metadata['total_chunks'] = len(chunks)
                chunk_metadata['chunk_size'] = len(chunk)

                documents.append(Document(page_content=chunk, metadata=chunk_metadata))

        return documents

    def chunk_with_context(self, text: str, context_window: int = 100) -> List[dict]:
        """
        Create chunks with additional context for better retrieval.

        Args:
            text: Text to chunk
            context_window: Characters of context to add before/after

        Returns:
            List of dicts with chunk, context_before, context_after
        """
        chunks = self.split_text(text)
        chunks_with_context = []

        full_text = text
        for chunk in chunks:
            # Find chunk position in original text
            start = full_text.find(chunk)
            if start == -1:
                continue

            end = start + len(chunk)

            # Extract context
            context_before = full_text[max(0, start - context_window):start]
            context_after = full_text[end:min(len(full_text), end + context_window)]

            chunks_with_context.append({
                'chunk': chunk,
                'context_before': context_before,
                'context_after': context_after,
                'position': start
            })

        return chunks_with_context
