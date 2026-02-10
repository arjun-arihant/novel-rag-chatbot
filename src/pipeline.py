# Novel RAG Pipeline - Multi-Novel Support

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
from .retrieval.sparse_index import BM25Index
from .retrieval.hybrid import HybridRetriever
from .retrieval.reranker import LLMReranker
from .generation.query_rewriter import QueryRewriter
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


class RAGPipeline:
    """
    Multi-novel RAG pipeline orchestrator.
    
    Supports:
    - Multiple novels with separate databases
    - Incremental indexing
    - Novel selection for queries
    """
    
    def __init__(self, library_path: str = "library"):
        self.config = get_config()
        self.client = get_client(self.config.embedding.base_url)
        
        # Library and indexer
        self.library = get_library(library_path)
        self.embedder = Embedder(self.client)
        self.indexer = IncrementalIndexer(self.embedder, self.library)
        
        # Generation components (shared)
        self.reranker = LLMReranker(self.client)
        self.query_rewriter = QueryRewriter(self.client)
        self.entity_extractor = EntityExtractor()
        self.generator = GroundedGenerator(self.client, self.entity_extractor)
        
        # Active novel's retrieval components
        self._active_retriever: Optional[HybridRetriever] = None
    
    def add_novel(
        self, 
        file_path: Path, 
        title: Optional[str] = None,
        author: str = "Unknown",
        progress_callback=None
    ) -> dict:
        """
        Add and index a novel.
        
        Args:
            file_path: Path to novel file
            title: Optional title (defaults to filename)
            author: Author name
            progress_callback: Optional callback(current, total, message)
            
        Returns:
            Dict with novel metadata and indexing result
        """
        # Add to library
        novel = self.library.add_novel(file_path, title, author)
        
        # Index
        result = self.indexer.index_novel(novel.id, progress_callback)
        
        # Return combined info
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
        """Re-index a novel (for incremental updates)."""
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
        """
        Select a novel for querying.
        
        Args:
            novel_id: ID of the novel to select
            
        Returns:
            True if selection successful
        """
        if not self.library.set_active_novel(novel_id):
            return False
        
        # Load retrieval components for this novel
        try:
            vector_store, bm25_index = self.indexer.get_novel_stores(novel_id)
            self._active_retriever = HybridRetriever(
                vector_store,
                bm25_index,
                rrf_k=self.config.retrieval.rrf_k
            )
            logger.info(f"Loaded retrieval components for novel {novel_id}")
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
        Process a user query against the active novel.
        
        Args:
            user_query: The user's question
            
        Returns:
            PipelineResult with answer and metadata
        """
        import time
        
        if not self._active_retriever:
            return PipelineResult(
                answer="Please select a novel first.",
                refused=True,
                refusal_reason="no_novel_selected",
                original_query=user_query,
                rewritten_query="",
                chapters_cited=[],
                sources=[],
                timing={}
            )
        
        timing = {}
        
        # Step 1: Query rewriting
        start = time.time()
        rewritten_query = self.query_rewriter.rewrite(user_query)
        timing['rewrite'] = time.time() - start
        
        # Step 2: Hybrid retrieval
        start = time.time()
        hybrid_results = self._active_retriever.retrieve(rewritten_query)
        timing['retrieval'] = time.time() - start
        
        # Step 3: Reranking
        start = time.time()
        reranked_results = self.reranker.rerank(
            rewritten_query,
            hybrid_results,
            top_k=self.config.retrieval.final_top_k,
            candidate_k=self.config.retrieval.rerank_top_k
        )
        timing['rerank'] = time.time() - start
        
        # Step 4: Generation
        start = time.time()
        gen_result = self.generator.generate(rewritten_query, reranked_results)
        timing['generation'] = time.time() - start
        
        return PipelineResult(
            answer=gen_result.answer,
            refused=gen_result.refused,
            refusal_reason=gen_result.refusal_reason,
            original_query=user_query,
            rewritten_query=rewritten_query,
            chapters_cited=gen_result.chapters_cited,
            sources=gen_result.sources,
            timing=timing
        )
    
    def query_stream(self, user_query: str) -> Iterator[str]:
        """Stream query response for UI."""
        if not self._active_retriever:
            yield "Please select a novel first."
            return
        
        # Rewrite
        rewritten_query = self.query_rewriter.rewrite(user_query)
        
        # Retrieve
        hybrid_results = self._active_retriever.retrieve(rewritten_query)
        
        # Rerank
        reranked_results = self.reranker.rerank(
            rewritten_query,
            hybrid_results,
            top_k=self.config.retrieval.final_top_k,
            candidate_k=self.config.retrieval.rerank_top_k
        )
        
        # Stream generation
        for token in self.generator.generate_stream(rewritten_query, reranked_results):
            yield token
    
    def is_ready(self) -> bool:
        """Check if pipeline has an active novel."""
        return self._active_retriever is not None
    
    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        active = self.library.get_active_novel()
        return {
            "novels_count": len(self.library.list_novels()),
            "active_novel": active.to_dict() if active else None,
            "ready": self.is_ready()
        }
