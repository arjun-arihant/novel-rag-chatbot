# RAG Pipeline - Full orchestration

import logging
from pathlib import Path
from typing import Optional, Iterator
from dataclasses import dataclass

from .config import get_config
from .ollama_client import OllamaClient, get_client
from .ingestion.loader import NovelLoader
from .ingestion.chunker import TokenChunker
from .ingestion.metadata import EntityExtractor, ChapterExtractor
from .retrieval.embedder import Embedder
from .retrieval.vector_store import VectorStore
from .retrieval.sparse_index import BM25Index
from .retrieval.hybrid import HybridRetriever
from .retrieval.reranker import LLMReranker
from .generation.query_rewriter import QueryRewriter
from .generation.generator import GroundedGenerator, GenerationResult

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
    Complete RAG pipeline orchestrator.
    
    Pipeline flow:
    1. Query rewriting (temp=0.0)
    2. Hybrid retrieval (dense + sparse + RRF)
    3. LLM reranking (constrained prompt)
    4. Grounded generation (with refusal logic)
    """
    
    def __init__(self):
        self.config = get_config()
        self.client = get_client(self.config.embedding.base_url)
        
        # Ingestion components
        self.chunker = TokenChunker(
            target_tokens=self.config.chunking.target_tokens,
            min_tokens=self.config.chunking.min_tokens,
            max_tokens=self.config.chunking.max_tokens,
            overlap_tokens=self.config.chunking.overlap_tokens,
            malformed_threshold=self.config.chunking.malformed_threshold
        )
        self.entity_extractor = EntityExtractor()
        self.chapter_extractor = ChapterExtractor()
        
        # Retrieval components
        self.embedder = Embedder(self.client)
        self.vector_store = VectorStore(
            persist_path=self.config.paths.chroma_db,
            embedder=self.embedder
        )
        self.bm25_index = BM25Index(persist_path=self.config.paths.bm25_index)
        self.hybrid_retriever = HybridRetriever(
            self.vector_store,
            self.bm25_index,
            rrf_k=self.config.retrieval.rrf_k
        )
        self.reranker = LLMReranker(self.client)
        
        # Generation components
        self.query_rewriter = QueryRewriter(self.client)
        self.generator = GroundedGenerator(self.client, self.entity_extractor)
        
        self._initialized = False
        
    def ingest_novel(self, novel_path: Path, force_reindex: bool = False) -> dict:
        """
        Ingest and index a novel.
        
        Args:
            novel_path: Path to novel text file
            force_reindex: Whether to clear and rebuild index
            
        Returns:
            Statistics about ingestion
        """
        import time
        start_time = time.time()
        
        # Check if already indexed
        if not force_reindex and self.vector_store.get_count() > 0:
            logger.info("Using existing index")
            if not self.bm25_index.load():
                self._rebuild_bm25()
            self._initialized = True
            return {"status": "skipped", "existing_chunks": self.vector_store.get_count()}
        
        if force_reindex:
            logger.info("Clearing existing index...")
            self.vector_store.clear()
        
        # Load novel
        loader = NovelLoader(novel_path)
        loader.load()
        chapters = loader.parse_chapters()
        
        # Chunk all chapters
        all_chunks = []
        for chapter in chapters:
            chunks = self.chunker.chunk_chapter(
                chapter.content,
                chapter.number,
                chapter.title
            )
            all_chunks.extend(chunks)
            
            # Extract entities
            self.entity_extractor.extract_from_text(chapter.content, chapter.number)
            self.chapter_extractor.add_chapter(chapter.number, chapter.title, chapter.content)
            
        logger.info(f"Created {len(all_chunks)} chunks from {len(chapters)} chapters")
        
        # Add to vector store
        self.vector_store.add_chunks(all_chunks)
        
        # Build BM25 index
        self._rebuild_bm25()
        
        self._initialized = True
        
        return {
            "status": "indexed",
            "chapters": len(chapters),
            "chunks": len(all_chunks),
            "entities": len(self.entity_extractor.get_significant_entities()),
            "time_seconds": time.time() - start_time
        }
    
    def _rebuild_bm25(self):
        """Rebuild BM25 index from vector store."""
        docs = self.vector_store.get_all_documents()
        self.bm25_index.build_from_documents(docs)
        
    def query(self, user_query: str) -> PipelineResult:
        """
        Process a user query through the full pipeline.
        
        Args:
            user_query: The user's question
            
        Returns:
            PipelineResult with answer and metadata
        """
        import time
        timing = {}
        
        # Step 1: Query rewriting
        start = time.time()
        rewritten_query = self.query_rewriter.rewrite(user_query)
        timing['rewrite'] = time.time() - start
        
        # Step 2: Hybrid retrieval
        start = time.time()
        hybrid_results = self.hybrid_retriever.retrieve(rewritten_query)
        timing['retrieval'] = time.time() - start
        
        # Step 3: Reranking
        start = time.time()
        reranked_results = self.reranker.rerank(rewritten_query, hybrid_results)
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
        # Rewrite
        rewritten_query = self.query_rewriter.rewrite(user_query)
        
        # Retrieve
        hybrid_results = self.hybrid_retriever.retrieve(rewritten_query)
        
        # Rerank
        reranked_results = self.reranker.rerank(rewritten_query, hybrid_results)
        
        # Stream generation
        for token in self.generator.generate_stream(rewritten_query, reranked_results):
            yield token
    
    def is_ready(self) -> bool:
        """Check if pipeline is ready for queries."""
        return self._initialized and self.vector_store.get_count() > 0
    
    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        return {
            "initialized": self._initialized,
            "chunks_indexed": self.vector_store.get_count(),
            "bm25_docs": self.bm25_index.get_count(),
            "entities": len(self.entity_extractor.get_significant_entities()),
            "chapters": len(self.chapter_extractor.get_all())
        }
