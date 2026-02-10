# Hybrid Retrieval with Reciprocal Rank Fusion

import hashlib
import logging
from typing import List, Dict, Any, Optional
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
    Uses weighted Reciprocal Rank Fusion (RRF) for result combination.
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
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        dense_top_k: Optional[int] = None,
        sparse_top_k: Optional[int] = None,
        fusion_top_k: Optional[int] = None
    ) -> List[FusedResult]:
        """Retrieve documents using hybrid search + weighted RRF fusion."""
        config = self.config.retrieval
        dense_top_k = dense_top_k or config.dense_top_k
        sparse_top_k = sparse_top_k or config.sparse_top_k
        fusion_top_k = fusion_top_k or config.fusion_top_k

        dense_results = self.vector_store.search(query, top_k=dense_top_k)
        sparse_results = self.bm25_index.search(query, top_k=sparse_top_k)

        dense_ranking = {
            self._doc_key(r.content, r.metadata): i
            for i, r in enumerate(dense_results)
        }
        sparse_ranking = {
            self._doc_key(d.content, d.metadata): i
            for i, (d, _) in enumerate(sparse_results)
        }

        all_docs: Dict[str, Dict[str, Any]] = {}

        for r in dense_results:
            key = self._doc_key(r.content, r.metadata)
            all_docs[key] = {
                'content': r.content,
                'chapter_number': r.chapter_number,
                'chapter_title': r.chapter_title,
                'metadata': r.metadata,
                'dense_rank': dense_ranking.get(key),
                'sparse_rank': None
            }

        for d, _ in sparse_results:
            key = self._doc_key(d.content, d.metadata)
            if key in all_docs:
                all_docs[key]['sparse_rank'] = sparse_ranking.get(key)
                continue

            all_docs[key] = {
                'content': d.content,
                'chapter_number': d.metadata.get('chapter_number', 0),
                'chapter_title': d.metadata.get('chapter_title', ''),
                'metadata': d.metadata,
                'dense_rank': None,
                'sparse_rank': sparse_ranking.get(key)
            }

        dense_weight = max(0.0, config.dense_weight)
        sparse_weight = max(0.0, config.sparse_weight)

        results = []
        for doc in all_docs.values():
            rrf_score = 0.0

            if doc['dense_rank'] is not None:
                rrf_score += dense_weight / (self.rrf_k + doc['dense_rank'] + 1)

            if doc['sparse_rank'] is not None:
                rrf_score += sparse_weight / (self.rrf_k + doc['sparse_rank'] + 1)

            results.append(FusedResult(
                content=doc['content'],
                chapter_number=doc['chapter_number'],
                chapter_title=doc['chapter_title'],
                rrf_score=rrf_score,
                dense_rank=doc['dense_rank'],
                sparse_rank=doc['sparse_rank'],
                metadata=doc['metadata']
            ))

        results.sort(key=lambda x: -x.rrf_score)

        logger.info(
            "Hybrid retrieved %d dense + %d sparse -> %d fused",
            len(dense_results),
            len(sparse_results),
            min(len(results), fusion_top_k)
        )
        return results[:fusion_top_k]

    def _doc_key(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create stable key for deduping across dense + sparse channels."""
        metadata = metadata or {}

        chunk_id = metadata.get('chunk_id')
        if chunk_id:
            return f"chunk:{chunk_id}"

        chapter = metadata.get('chapter_number')
        chunk_index = metadata.get('chunk_index')
        if chapter is not None and chunk_index is not None:
            return f"chapter:{chapter}:chunk:{chunk_index}"

        digest = hashlib.sha1(content.strip().lower().encode('utf-8')).hexdigest()
        return f"content:{digest[:16]}"

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
