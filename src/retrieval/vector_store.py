# Vector Store - ChromaDB wrapper

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings

from .embedder import Embedder
from ..ingestion.chunker import Chunk
from ..config import get_config

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDoc:
    """A retrieved document with score."""
    content: str
    chapter_number: int
    chapter_title: str
    chunk_index: int
    score: float
    metadata: Dict[str, Any]


class VectorStore:
    """ChromaDB vector store for dense retrieval."""
    
    COLLECTION_NAME = "novel_chunks"
    
    def __init__(self, persist_path: Optional[Path] = None, embedder: Optional[Embedder] = None):
        config = get_config()
        self.persist_path = persist_path or config.paths.chroma_db
        self.embedder = embedder or Embedder()
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.persist_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        
    def add_chunks(self, chunks: List[Chunk], show_progress: bool = True):
        """Add chunks to the vector store."""
        if not chunks:
            return
            
        # Prepare data
        ids = [f"chunk_{c.chapter_number}_{c.chunk_index}" for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [
            {
                "chapter_number": c.chapter_number,
                "chapter_title": c.chapter_title,
                "chunk_index": c.chunk_index,
                "total_chunks": c.total_chunks,
                "token_count": c.token_count,
                "parent_context": c.parent_context[:200] if c.parent_context else ""
            }
            for c in chunks
        ]
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.embedder.embed_batch(documents)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(chunks)} chunks to vector store")
        
    def search(
        self,
        query: str,
        top_k: int = 20,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedDoc]:
        """Search for similar chunks."""
        # Get query embedding
        query_embedding = self.embedder.embed(query)
        
        # Search
        where_filter = filter_metadata if filter_metadata else None
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        # Convert to RetrievedDoc
        docs = []
        if results['documents'] and results['documents'][0]:
            for i, (doc, meta, dist) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                # Convert distance to similarity score (cosine distance → similarity)
                score = 1 - dist  # ChromaDB returns distance, not similarity
                
                docs.append(RetrievedDoc(
                    content=doc,
                    chapter_number=meta.get('chapter_number', 0),
                    chapter_title=meta.get('chapter_title', ''),
                    chunk_index=meta.get('chunk_index', 0),
                    score=score,
                    metadata=meta
                ))
                
        return docs
    
    def get_count(self) -> int:
        """Get number of documents in store."""
        return self.collection.count()
    
    def clear(self):
        """Clear all documents."""
        self.client.delete_collection(self.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        
    def get_all_documents(self) -> List[Tuple[str, Dict]]:
        """Get all documents (for BM25 index building)."""
        results = self.collection.get(
            include=["documents", "metadatas"]
        )
        
        if not results['documents']:
            return []
            
        return list(zip(results['documents'], results['metadatas']))
