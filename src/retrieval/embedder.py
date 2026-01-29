# Embedder - Wrapper for embedding generation (with query/doc prefixes)

import logging
from typing import List, Optional

from ..ollama_client import OllamaClient, get_client
from ..config import get_config

logger = logging.getLogger(__name__)


class Embedder:
    """
    Generate embeddings using Ollama.
    
    Uses query/document prefixes for better retrieval.
    Many embedding models are trained with different prefixes for
    queries vs documents to improve retrieval quality.
    """
    
    # Prefixes for different embedding models
    # qwen3-embedding and many other models benefit from these
    QUERY_PREFIX = "query: "
    DOC_PREFIX = "passage: "
    
    def __init__(self, client: Optional[OllamaClient] = None, use_prefixes: bool = True):
        self.config = get_config()
        self.client = client or get_client(self.config.embedding.base_url)
        self.model = self.config.embedding.model
        self.use_prefixes = use_prefixes
        self._dimension: Optional[int] = None
        
    def embed(self, text: str, is_query: bool = True) -> List[float]:
        """
        Get embedding for a single text.
        
        Args:
            text: The text to embed
            is_query: True for queries, False for documents
        """
        if self.use_prefixes:
            prefix = self.QUERY_PREFIX if is_query else self.DOC_PREFIX
            text = prefix + text
        
        return self.client.embed(self.model, text)
    
    def embed_query(self, text: str) -> List[float]:
        """Get embedding for a query."""
        return self.embed(text, is_query=True)
    
    def embed_document(self, text: str) -> List[float]:
        """Get embedding for a document."""
        return self.embed(text, is_query=False)
    
    def embed_batch(self, texts: List[str], is_query: bool = False, batch_size: int = 10) -> List[List[float]]:
        """
        Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            is_query: True for queries, False for documents
            batch_size: Batch size for API calls
        """
        if self.use_prefixes:
            prefix = self.QUERY_PREFIX if is_query else self.DOC_PREFIX
            texts = [prefix + t for t in texts]
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.client.embed_batch(self.model, batch)
            all_embeddings.extend(embeddings)
            
        return all_embeddings
    
    def embed_documents(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """Get embeddings for multiple documents."""
        return self.embed_batch(texts, is_query=False, batch_size=batch_size)
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension (lazy-loaded)."""
        if self._dimension is None:
            # Get dimension from a test embedding
            test_embedding = self.embed("test", is_query=True)
            self._dimension = len(test_embedding)
        return self._dimension
