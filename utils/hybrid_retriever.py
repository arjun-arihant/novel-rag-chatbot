"""
Hybrid retriever combining BM25 keyword search and semantic vector search
"""
from typing import List, Dict, Any
from langchain_core.documents import Document
try:
    from langchain.retrievers.ensemble import EnsembleRetriever
except ImportError:
    try:
        from langchain_community.retrievers import EnsembleRetriever
    except ImportError:
        # If neither works, we'll handle it gracefully
        EnsembleRetriever = None
from langchain_community.retrievers import BM25Retriever


class HybridRetriever:
    """Combines BM25 and semantic search for robust retrieval."""

    def __init__(self, vectorstore, documents: List[Document] = None,
                 bm25_weight: float = 0.3, semantic_weight: float = 0.7,
                 top_k: int = 5, similarity_threshold: float = 0.7):
        """
        Initialize hybrid retriever.

        Args:
            vectorstore: ChromaDB vectorstore
            documents: List of documents for BM25 index
            bm25_weight: Weight for BM25 results (0-1)
            semantic_weight: Weight for semantic results (0-1)
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score
        """
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight

        # Create semantic retriever
        self.semantic_retriever = vectorstore.as_retriever(
            search_kwargs={"k": top_k * 2}  # Retrieve more for filtering
        )

        # Create BM25 retriever
        if documents:
            self.bm25_retriever = BM25Retriever.from_documents(documents)
            self.bm25_retriever.k = top_k * 2
        else:
            self.bm25_retriever = None

        # Create ensemble retriever if BM25 available
        if self.bm25_retriever and EnsembleRetriever is not None:
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, self.semantic_retriever],
                weights=[bm25_weight, semantic_weight]
            )
        else:
            self.ensemble_retriever = None
            if self.bm25_retriever and EnsembleRetriever is None:
                print("Warning: EnsembleRetriever not available, falling back to semantic search only")

    def retrieve(self, query: str, search_type: str = "hybrid",
                use_mmr: bool = False, mmr_lambda: float = 0.7,
                filter_metadata: Dict[str, Any] = None) -> List[Document]:
        """
        Retrieve documents using specified search type.

        Args:
            query: Search query
            search_type: 'semantic', 'keyword', or 'hybrid'
            use_mmr: Use Maximal Marginal Relevance for diversity
            mmr_lambda: MMR lambda parameter (relevance vs diversity)
            filter_metadata: Filter results by metadata

        Returns:
            List of retrieved documents
        """
        if search_type == "semantic":
            docs = self._semantic_search(query, use_mmr, mmr_lambda, filter_metadata)
        elif search_type == "keyword":
            docs = self._keyword_search(query)
        elif search_type == "hybrid":
            docs = self._hybrid_search(query, use_mmr, mmr_lambda, filter_metadata)
        else:
            docs = self._hybrid_search(query, use_mmr, mmr_lambda, filter_metadata)

        # Filter by similarity threshold if possible
        filtered_docs = self._filter_by_threshold(docs)

        # Apply metadata filtering
        if filter_metadata:
            filtered_docs = self._filter_by_metadata(filtered_docs, filter_metadata)

        # Limit to top_k
        return filtered_docs[:self.top_k]

    def _semantic_search(self, query: str, use_mmr: bool = False,
                        mmr_lambda: float = 0.7, filter_metadata: Dict = None) -> List[Document]:
        """Perform semantic vector search."""
        if use_mmr:
            # Use MMR for diversity
            search_kwargs = {
                "k": self.top_k * 2,
                "lambda_mult": mmr_lambda
            }
            if filter_metadata:
                search_kwargs["filter"] = filter_metadata

            docs = self.vectorstore.max_marginal_relevance_search(
                query,
                **search_kwargs
            )
        else:
            # Regular similarity search
            search_kwargs = {"k": self.top_k * 2}
            if filter_metadata:
                search_kwargs["filter"] = filter_metadata

            docs = self.vectorstore.similarity_search(query, **search_kwargs)

        return docs

    def _keyword_search(self, query: str) -> List[Document]:
        """Perform BM25 keyword search."""
        if not self.bm25_retriever:
            # Fallback to semantic if BM25 not available
            return self._semantic_search(query)

        return self.bm25_retriever.get_relevant_documents(query)

    def _hybrid_search(self, query: str, use_mmr: bool = False,
                      mmr_lambda: float = 0.7, filter_metadata: Dict = None) -> List[Document]:
        """Perform hybrid search combining both methods."""
        if not self.ensemble_retriever:
            # Fallback to semantic if ensemble not available
            return self._semantic_search(query, use_mmr, mmr_lambda, filter_metadata)

        # Get results from ensemble
        if use_mmr:
            # Manual hybrid with MMR on semantic component
            semantic_docs = self._semantic_search(query, True, mmr_lambda, filter_metadata)
            keyword_docs = self._keyword_search(query)

            # Merge and deduplicate
            docs = self._merge_results(semantic_docs, keyword_docs)
        else:
            docs = self.ensemble_retriever.get_relevant_documents(query)

        return docs

    def _merge_results(self, semantic_docs: List[Document],
                      keyword_docs: List[Document]) -> List[Document]:
        """
        Merge results from different retrievers.

        Args:
            semantic_docs: Documents from semantic search
            keyword_docs: Documents from keyword search

        Returns:
            Merged and deduplicated documents
        """
        # Create weighted scores
        doc_scores = {}

        for i, doc in enumerate(semantic_docs):
            doc_id = doc.page_content[:100]  # Use content prefix as ID
            score = (len(semantic_docs) - i) * self.semantic_weight
            doc_scores[doc_id] = {'doc': doc, 'score': score}

        for i, doc in enumerate(keyword_docs):
            doc_id = doc.page_content[:100]
            score = (len(keyword_docs) - i) * self.bm25_weight

            if doc_id in doc_scores:
                doc_scores[doc_id]['score'] += score
            else:
                doc_scores[doc_id] = {'doc': doc, 'score': score}

        # Sort by combined score
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x['score'], reverse=True)

        return [item['doc'] for item in sorted_docs]

    def _filter_by_threshold(self, docs: List[Document]) -> List[Document]:
        """
        Filter documents by similarity threshold.

        Args:
            docs: Documents to filter

        Returns:
            Filtered documents
        """
        # Try to get similarity scores from metadata
        filtered = []
        for doc in docs:
            if hasattr(doc, 'metadata') and 'score' in doc.metadata:
                if doc.metadata['score'] >= self.similarity_threshold:
                    filtered.append(doc)
            else:
                # If no score available, include all
                filtered.append(doc)

        return filtered if filtered else docs  # Return all if none pass threshold

    def _filter_by_metadata(self, docs: List[Document],
                           filter_dict: Dict[str, Any]) -> List[Document]:
        """
        Filter documents by metadata criteria.

        Args:
            docs: Documents to filter
            filter_dict: Metadata filters

        Returns:
            Filtered documents
        """
        filtered = []
        for doc in docs:
            match = True
            for key, value in filter_dict.items():
                if key not in doc.metadata:
                    match = False
                    break

                if isinstance(value, list):
                    if doc.metadata[key] not in value:
                        match = False
                        break
                else:
                    if doc.metadata[key] != value:
                        match = False
                        break

            if match:
                filtered.append(doc)

        return filtered

    def retrieve_by_chapter(self, query: str, chapters: List[str]) -> List[Document]:
        """
        Retrieve documents from specific chapters.

        Args:
            query: Search query
            chapters: List of chapter numbers to search in

        Returns:
            Retrieved documents from specified chapters
        """
        filter_metadata = {"chapter_number": chapters}
        return self.retrieve(query, filter_metadata=filter_metadata)

    def retrieve_with_character(self, query: str, character: str) -> List[Document]:
        """
        Retrieve documents mentioning a specific character.

        Args:
            query: Search query
            character: Character name

        Returns:
            Retrieved documents mentioning the character
        """
        # This requires character metadata to be added during chunking
        docs = self.retrieve(query, search_type="hybrid")

        # Filter for character mentions in content
        filtered = []
        for doc in docs:
            if character in doc.page_content:
                filtered.append(doc)

        return filtered if filtered else docs  # Return all if none mention character

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        return {
            "search_weights": {
                "bm25": self.bm25_weight,
                "semantic": self.semantic_weight
            },
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "bm25_available": self.bm25_retriever is not None
        }