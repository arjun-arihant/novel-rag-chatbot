# Hybrid Retrieval with Reciprocal Rank Fusion

import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass

from .vector_store import VectorStore, RetrievedDoc
from .sparse_index import BM25Index
from ..config import get_config

logger = logging.getLogger(__name__)


@dataclass
class FusedResult:
    """Result after RRF fusion."""
    content: str
    chapter_number: int
    chapter_title: str
    rrf_score: float
    dense_rank: Optional[int]
    sparse_rank: Optional[int]
    metadata: Dict[str, Any]


class HybridRetriever:
    """
    Hybrid retrieval combining dense (vector) and sparse (BM25) search.
    Uses Reciprocal Rank Fusion (RRF) for result combination.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        rrf_k: int = 60
    ):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.config = get_config()
        self.rrf_k = rrf_k  # RRF constant
        
    def retrieve(
        self,
        query: str,
        dense_top_k: Optional[int] = None,
        sparse_top_k: Optional[int] = None,
        fusion_top_k: Optional[int] = None
    ) -> List[FusedResult]:
        """
        Retrieve documents using hybrid search.
        
        1. Dense search (vector similarity)
        2. Sparse search (BM25)
        3. RRF fusion
        """
        config = self.config.retrieval
        dense_top_k = dense_top_k or config.dense_top_k
        sparse_top_k = sparse_top_k or config.sparse_top_k
        fusion_top_k = fusion_top_k or config.fusion_top_k
        
        # Dense retrieval
        dense_results = self.vector_store.search(query, top_k=dense_top_k)
        
        # Sparse retrieval
        sparse_results = self.bm25_index.search(query, top_k=sparse_top_k)
        
        # Build rankings
        dense_ranking = {self._doc_key(r.content): i for i, r in enumerate(dense_results)}
        sparse_ranking = {self._doc_key(d.content): i for i, (d, _) in enumerate(sparse_results)}
        
        # Collect all unique documents
        all_docs: Dict[str, Dict[str, Any]] = {}
        
        for r in dense_results:
            key = self._doc_key(r.content)
            all_docs[key] = {
                'content': r.content,
                'chapter_number': r.chapter_number,
                'chapter_title': r.chapter_title,
                'metadata': r.metadata,
                'dense_rank': dense_ranking.get(key),
                'sparse_rank': None
            }
            
        for d, score in sparse_results:
            key = self._doc_key(d.content)
            if key in all_docs:
                all_docs[key]['sparse_rank'] = sparse_ranking.get(key)
            else:
                all_docs[key] = {
                    'content': d.content,
                    'chapter_number': d.metadata.get('chapter_number', 0),
                    'chapter_title': d.metadata.get('chapter_title', ''),
                    'metadata': d.metadata,
                    'dense_rank': None,
                    'sparse_rank': sparse_ranking.get(key)
                }
        
        # Calculate RRF scores
        results = []
        for key, doc in all_docs.items():
            rrf_score = 0.0
            
            if doc['dense_rank'] is not None:
                rrf_score += 1.0 / (self.rrf_k + doc['dense_rank'] + 1)
                
            if doc['sparse_rank'] is not None:
                rrf_score += 1.0 / (self.rrf_k + doc['sparse_rank'] + 1)
            
            results.append(FusedResult(
                content=doc['content'],
                chapter_number=doc['chapter_number'],
                chapter_title=doc['chapter_title'],
                rrf_score=rrf_score,
                dense_rank=doc['dense_rank'],
                sparse_rank=doc['sparse_rank'],
                metadata=doc['metadata']
            ))
        
        # Sort by RRF score
        results.sort(key=lambda x: -x.rrf_score)
        
        return results[:fusion_top_k]
    
    def _doc_key(self, content: str) -> str:
        """Create a unique key for a document (first 100 chars)."""
        return content[:100]
    
    def dense_only(self, query: str, top_k: int = 20) -> List[RetrievedDoc]:
        """Dense search only."""
        return self.vector_store.search(query, top_k=top_k)
    
    def sparse_only(self, query: str, top_k: int = 20) -> List[FusedResult]:
        """Sparse search only."""
        results = self.bm25_index.search(query, top_k=top_k)
        return [
            FusedResult(
                content=d.content,
                chapter_number=d.metadata.get('chapter_number', 0),
                chapter_title=d.metadata.get('chapter_title', ''),
                rrf_score=score,
                dense_rank=None,
                sparse_rank=i,
                metadata=d.metadata
            )
            for i, (d, score) in enumerate(results)
        ]
