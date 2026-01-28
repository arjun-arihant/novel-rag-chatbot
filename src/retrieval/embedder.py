# Embedder - Wrapper for embedding generation

import logging
from typing import List, Optional

from ..ollama_client import OllamaClient, get_client
from ..config import get_config

logger = logging.getLogger(__name__)


class Embedder:
    """Generate embeddings using Ollama."""
    
    def __init__(self, client: Optional[OllamaClient] = None):
        self.config = get_config()
        self.client = client or get_client(self.config.embedding.base_url)
        self.model = self.config.embedding.model
        self._dimension: Optional[int] = None
        
    def embed(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        return self.client.embed(self.model, text)
    
    def embed_batch(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.client.embed_batch(self.model, batch)
            all_embeddings.extend(embeddings)
            
        return all_embeddings
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension (lazy-loaded)."""
        if self._dimension is None:
            # Get dimension from a test embedding
            test_embedding = self.embed("test")
            self._dimension = len(test_embedding)
        return self._dimension
