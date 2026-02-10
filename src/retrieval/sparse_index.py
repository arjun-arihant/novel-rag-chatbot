# BM25 Sparse Index

import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    logger.warning("rank_bm25 not installed, sparse search disabled")


@dataclass
class BM25Doc:
    """Document for BM25 index."""
    content: str
    tokens: List[str]
    metadata: Dict[str, Any]


class BM25Index:
    """BM25 sparse index for keyword search."""

    # Keep lexical signals useful for fiction names and places.
    TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+(?:['-][a-zA-Z0-9]+)?")

    # Common stopwords
    STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
        'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
        'his', 'our', 'their', 'what', 'which', 'who', 'whom', 'where',
        'when', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now'
    }

    def __init__(self, persist_path: Optional[Path] = None):
        self.persist_path = Path(persist_path) if persist_path else None
        self.documents: List[BM25Doc] = []
        self.bm25: Optional['BM25Okapi'] = None
        self._loaded = False

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25 while preserving lexical entities."""
        tokens = [t.lower() for t in self.TOKEN_PATTERN.findall(text)]
        return [t for t in tokens if t not in self.STOPWORDS and len(t) > 1]

    def build_from_documents(self, documents: List[Tuple[str, Dict]]):
        """Build index from list of (content, metadata) tuples."""
        if not HAS_BM25:
            logger.error("Cannot build BM25 index: rank_bm25 not installed")
            return

        self.documents = []
        tokenized_corpus = []

        for content, metadata in documents:
            tokens = self.tokenize(content)
            self.documents.append(BM25Doc(
                content=content,
                tokens=tokens,
                metadata=metadata
            ))
            tokenized_corpus.append(tokens)

        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info(f"Built BM25 index with {len(self.documents)} documents")

            if self.persist_path:
                self.save()

    def search(self, query: str, top_k: int = 20) -> List[Tuple[BM25Doc, float]]:
        """Search for documents matching query."""
        if not HAS_BM25 or self.bm25 is None:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        bm25_scores = self.bm25.get_scores(query_tokens)

        query_lower = query.lower().strip()
        top_indices = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])[:max(top_k * 3, top_k)]

        results = []
        for idx in top_indices:
            base_score = float(bm25_scores[idx])
            doc = self.documents[idx]
            doc_text = doc.content.lower()

            # Lightweight lexical boosts for fiction entities and exact matches.
            phrase_boost = 0.6 if query_lower and query_lower in doc_text else 0.0
            overlap = len(set(query_tokens).intersection(doc.tokens))
            overlap_boost = min(0.4, 0.05 * overlap)

            score = float(base_score + phrase_boost + overlap_boost)
            results.append((doc, score))

        results.sort(key=lambda item: -item[1])
        return results[:top_k]

    def save(self):
        """Save index to disk."""
        if not self.persist_path:
            return

        data = {
            'documents': [
                {
                    'content': d.content,
                    'tokens': d.tokens,
                    'metadata': d.metadata
                }
                for d in self.documents
            ]
        }

        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.persist_path, 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"Saved BM25 index to {self.persist_path}")

    def load(self) -> bool:
        """Load index from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return False

        if not HAS_BM25:
            return False

        try:
            with open(self.persist_path, 'rb') as f:
                data = pickle.load(f)

            self.documents = [
                BM25Doc(
                    content=d['content'],
                    tokens=d['tokens'],
                    metadata=d['metadata']
                )
                for d in data['documents']
            ]

            tokenized_corpus = [d.tokens for d in self.documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
            self._loaded = True

            logger.info(f"Loaded BM25 index: {len(self.documents)} documents")
            return True

        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            return False

    def get_count(self) -> int:
        """Get number of documents."""
        return len(self.documents)
