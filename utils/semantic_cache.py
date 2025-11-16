"""
Semantic cache for storing and reusing similar query results
"""
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np


class SemanticCache:
    """Cache query results based on semantic similarity."""

    def __init__(self, cache_path: str = "semantic_cache.json",
                 similarity_threshold: float = 0.95,
                 max_cache_size: int = 100,
                 ttl_hours: int = 168):
        """
        Initialize semantic cache.

        Args:
            cache_path: Path to cache file
            similarity_threshold: Minimum similarity to use cached result
            max_cache_size: Maximum number of cached entries
            ttl_hours: Time-to-live for cache entries in hours
        """
        self.cache_path = cache_path
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self.ttl_hours = ttl_hours
        self.cache = {}
        self.load_cache()

    def load_cache(self):
        """Load cache from disk."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                # Clean expired entries on load
                self._clean_expired()
            except Exception as e:
                print(f"Warning: Could not load semantic cache: {e}")
                self.cache = {}

    def save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save semantic cache: {e}")

    def get(self, query: str, query_embedding: List[float] = None) -> Optional[Dict]:
        """
        Get cached result for query if similar query exists.

        Args:
            query: Query string
            query_embedding: Optional embedding for the query

        Returns:
            Cached result if found, None otherwise
        """
        # Check for exact match first
        if query in self.cache:
            entry = self.cache[query]
            if not self._is_expired(entry):
                entry["hits"] += 1
                entry["last_accessed"] = datetime.now().isoformat()
                self.save_cache()
                return entry["result"]

        # Check for similar queries if embedding provided
        if query_embedding:
            similar_query = self._find_similar_query(query_embedding)
            if similar_query:
                entry = self.cache[similar_query]
                entry["hits"] += 1
                entry["last_accessed"] = datetime.now().isoformat()
                self.save_cache()
                return entry["result"]

        return None

    def set(self, query: str, result: Dict, query_embedding: List[float] = None):
        """
        Cache a query result.

        Args:
            query: Query string
            result: Result to cache
            query_embedding: Optional embedding for the query
        """
        # Check cache size limit
        if len(self.cache) >= self.max_cache_size:
            self._evict_oldest()

        self.cache[query] = {
            "result": result,
            "embedding": query_embedding,
            "created": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
            "hits": 0
        }

        self.save_cache()

    def _find_similar_query(self, query_embedding: List[float]) -> Optional[str]:
        """
        Find a similar cached query using cosine similarity.

        Args:
            query_embedding: Query embedding

        Returns:
            Similar query string if found, None otherwise
        """
        best_similarity = 0
        best_query = None

        query_vec = np.array(query_embedding)

        for cached_query, entry in self.cache.items():
            if self._is_expired(entry):
                continue

            if "embedding" not in entry or entry["embedding"] is None:
                continue

            cached_vec = np.array(entry["embedding"])

            # Cosine similarity
            similarity = np.dot(query_vec, cached_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(cached_vec)
            )

            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_query = cached_query

        return best_query

    def _is_expired(self, entry: Dict) -> bool:
        """
        Check if cache entry is expired.

        Args:
            entry: Cache entry

        Returns:
            True if expired, False otherwise
        """
        created = datetime.fromisoformat(entry["created"])
        expiry = created + timedelta(hours=self.ttl_hours)
        return datetime.now() > expiry

    def _clean_expired(self):
        """Remove expired entries from cache."""
        expired_keys = [
            key for key, entry in self.cache.items()
            if self._is_expired(entry)
        ]

        for key in expired_keys:
            del self.cache[key]

        if expired_keys:
            self.save_cache()

    def _evict_oldest(self):
        """Evict least recently used entries to maintain size limit."""
        if not self.cache:
            return

        # Sort by last accessed time
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1]["last_accessed"]
        )

        # Remove oldest 10% of entries
        num_to_remove = max(1, len(self.cache) // 10)
        for i in range(num_to_remove):
            if i < len(sorted_entries):
                del self.cache[sorted_entries[i][0]]

        self.save_cache()

    def invalidate(self, query: str):
        """
        Invalidate a specific cache entry.

        Args:
            query: Query to invalidate
        """
        if query in self.cache:
            del self.cache[query]
            self.save_cache()

    def clear(self):
        """Clear all cache entries."""
        self.cache = {}
        self.save_cache()

    def get_stats(self) -> Dict:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats
        """
        if not self.cache:
            return {
                "total_entries": 0,
                "cache_hit_rate": 0,
                "average_hits_per_entry": 0
            }

        total_hits = sum(entry["hits"] for entry in self.cache.values())
        total_entries = len(self.cache)

        return {
            "total_entries": total_entries,
            "total_hits": total_hits,
            "average_hits_per_entry": total_hits / total_entries if total_entries > 0 else 0,
            "cache_size_limit": self.max_cache_size,
            "similarity_threshold": self.similarity_threshold,
            "ttl_hours": self.ttl_hours
        }

    def get_top_queries(self, n: int = 10) -> List[Tuple[str, int]]:
        """
        Get top N most accessed queries.

        Args:
            n: Number of top queries to return

        Returns:
            List of (query, hits) tuples
        """
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1]["hits"],
            reverse=True
        )

        return [(query, entry["hits"]) for query, entry in sorted_entries[:n]]

    def preload_common_queries(self, queries: List[Dict]):
        """
        Preload cache with common queries.

        Args:
            queries: List of dicts with 'query', 'result', and optional 'embedding'
        """
        for item in queries:
            query = item["query"]
            result = item["result"]
            embedding = item.get("embedding")

            self.set(query, result, embedding)

    def optimize(self):
        """Optimize cache by removing low-hit entries and expired entries."""
        self._clean_expired()

        if len(self.cache) <= self.max_cache_size * 0.8:
            return

        # Remove entries with zero hits that are older than 24 hours
        threshold_time = datetime.now() - timedelta(hours=24)

        to_remove = []
        for query, entry in self.cache.items():
            if entry["hits"] == 0:
                created = datetime.fromisoformat(entry["created"])
                if created < threshold_time:
                    to_remove.append(query)

        for query in to_remove:
            del self.cache[query]

        if to_remove:
            self.save_cache()
