# Simple RAG Pipeline - No frills, just works

import logging
from pathlib import Path
from typing import Optional, Iterator
from dataclasses import dataclass

from .config import get_config
from .ollama_client import get_client
from .library import NovelLibrary, get_library
from .ingestion.indexer import IncrementalIndexer
from .retrieval.embedder import Embedder
from .retrieval.vector_store import VectorStore
from .generation.generator import GroundedGenerator, GenerationResult
from .ingestion.metadata import EntityExtractor

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Full pipeline result."""
    answer: str
    refused: bool
    refusal_reason: str
    original_query: str
    rewritten_query: str
    chapters_cited: list
    sources: list
    timing: dict


class SimpleRAGPipeline:
    """
    Simplified RAG pipeline - dense retrieval only, no LLM reranking.
    
    Pipeline:
    1. Dense vector search (top_k=8)
    2. Generate answer directly
    
    That's it. No query rewriting, no hybrid, no reranking.
    """
    
    def __init__(self, library_path: str = "library"):
        self.config = get_config()
        self.client = get_client(self.config.embedding.base_url)
        
        # Library and indexer
        self.library = get_library(library_path)
        self.embedder = Embedder(self.client)
        self.indexer = IncrementalIndexer(self.embedder, self.library)
        
        # Generation only (no reranker!)
        self.entity_extractor = EntityExtractor()
        self.generator = GroundedGenerator(self.client, self.entity_extractor)
        
        # Active novel's vector store
        self._active_vector_store: Optional[VectorStore] = None
        self._active_novel_id: Optional[str] = None
    
    def add_novel(
        self, 
        file_path: Path, 
        title: Optional[str] = None,
        author: str = "Unknown",
        progress_callback=None
    ) -> dict:
        """Add and index a novel."""
        novel = self.library.add_novel(file_path, title, author)
        result = self.indexer.index_novel(novel.id, progress_callback)
        
        updated_novel = self.library.get_novel(novel.id)
        return {
            "novel": updated_novel.to_dict() if updated_novel else None,
            "indexing": {
                "status": result.status,
                "total_chapters": result.total_chapters,
                "new_chapters": result.new_chapters,
                "updated_chapters": result.updated_chapters,
                "total_chunks": result.total_chunks,
                "new_chunks": result.new_chunks,
                "error": result.error_message
            }
        }
    
    def reindex_novel(self, novel_id: str, progress_callback=None) -> dict:
        """Re-index a novel."""
        result = self.indexer.index_novel(novel_id, progress_callback)
        updated_novel = self.library.get_novel(novel_id)
        return {
            "novel": updated_novel.to_dict() if updated_novel else None,
            "indexing": {
                "status": result.status,
                "total_chapters": result.total_chapters,
                "new_chapters": result.new_chapters,
                "updated_chapters": result.updated_chapters,
                "total_chunks": result.total_chunks,
                "new_chunks": result.new_chunks,
                "error": result.error_message
            }
        }
    
    def select_novel(self, novel_id: str) -> bool:
        """Select a novel for querying."""
        if not self.library.set_active_novel(novel_id):
            return False
        
        try:
            vector_store, _ = self.indexer.get_novel_stores(novel_id)
            self._active_vector_store = vector_store
            self._active_novel_id = novel_id
            logger.info(f"Loaded vector store for novel {novel_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to load novel stores: {e}")
            return False
    
    def get_active_novel(self) -> Optional[dict]:
        """Get the currently selected novel."""
        novel = self.library.get_active_novel()
        return novel.to_dict() if novel else None
    
    def list_novels(self) -> list[dict]:
        """List all novels in the library."""
        return [n.to_dict() for n in self.library.list_novels()]
    
    def delete_novel(self, novel_id: str) -> bool:
        """Delete a novel from the library."""
        return self.library.delete_novel(novel_id)
    
    def query(self, user_query: str) -> PipelineResult:
        """
        Process a user query - SIMPLE VERSION.
        
        Steps:
        1. Dense retrieval (top 8 chunks)
        2. Generate answer
        """
        import time
        
        if not self._active_vector_store:
            return PipelineResult(
                answer="Please select a novel first.",
                refused=True,
                refusal_reason="no_novel_selected",
                original_query=user_query,
                rewritten_query=user_query,  # No rewriting!
                chapters_cited=[],
                sources=[],
                timing={}
            )
        
        timing = {}
        
        # Step 1: Dense retrieval ONLY (no hybrid, no reranking!)
        start = time.time()
        dense_results = self._active_vector_store.search(user_query, top_k=8)
        timing['retrieval'] = time.time() - start
        
        # Convert to RerankResult format for compatibility
        from .retrieval.reranker import RerankResult
        reranked_results = [
            RerankResult(
                content=r.content,
                chapter_number=r.chapter_number,
                chapter_title=r.chapter_title,
                rerank_score=r.score * 10,  # Scale to 0-10
                reason="dense_retrieval",
                metadata=r.metadata
            )
            for r in dense_results
        ]
        
        # Step 2: Generation (no refusal check - just try!)
        start = time.time()
        gen_result = self.generator.generate(user_query, reranked_results)
        timing['generation'] = time.time() - start
        
        return PipelineResult(
            answer=gen_result.answer,
            refused=gen_result.refused,
            refusal_reason=gen_result.refusal_reason,
            original_query=user_query,
            rewritten_query=user_query,  # No rewriting
            chapters_cited=gen_result.chapters_cited,
            sources=gen_result.sources,
            timing=timing
        )
    
    def query_stream(self, user_query: str) -> Iterator[str]:
        """Stream query response for UI."""
        if not self._active_vector_store:
            yield "Please select a novel first."
            return
        
        # Dense retrieval only
        dense_results = self._active_vector_store.search(user_query, top_k=8)
        
        # Convert format
        from .retrieval.reranker import RerankResult
        reranked_results = [
            RerankResult(
                content=r.content,
                chapter_number=r.chapter_number,
                chapter_title=r.chapter_title,
                rerank_score=r.score * 10,
                reason="dense_retrieval",
                metadata=r.metadata
            )
            for r in dense_results
        ]
        
        # Stream generation
        for token in self.generator.generate_stream(user_query, reranked_results):
            yield token
    
    def is_ready(self) -> bool:
        """Check if pipeline has an active novel."""
        return self._active_vector_store is not None
    
    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        active = self.library.get_active_novel()
        return {
            "novels_count": len(self.library.list_novels()),
            "active_novel": active.to_dict() if active else None,
            "ready": self.is_ready(),
            "mode": "simple"
        }
