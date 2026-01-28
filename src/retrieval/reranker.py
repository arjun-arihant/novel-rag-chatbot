# LLM Reranker - Constrained scoring with JSON output

import json
import re
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

from ..ollama_client import OllamaClient, get_client
from ..config import get_config
from .hybrid import FusedResult

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Result after LLM reranking."""
    content: str
    chapter_number: int
    chapter_title: str
    rerank_score: float
    reason: str
    metadata: dict


# STRICT reranker prompt - prevents over-scoring
RERANK_PROMPT = """Score ONLY whether this passage DIRECTLY contains information needed to answer the query.

Query: {query}
Passage: {passage}

Scoring rules:
- 0-2: Passage is unrelated or only mentions same topic
- 3-5: Passage contains related but not directly useful info  
- 6-8: Passage directly answers part of the query
- 9-10: Passage fully answers the query

Do NOT score based on general topic similarity.
Return ONLY valid JSON: {{"score": <0-10>, "reason": "<max 10 words>"}}"""


class LLMReranker:
    """
    LLM-based reranker with strict scoring.
    
    - Temperature 0.0 for deterministic output
    - JSON format enforced
    - Aggressive output parsing/validation
    """
    
    def __init__(self, client: Optional[OllamaClient] = None):
        self.config = get_config()
        self.client = client or get_client(self.config.reranker.base_url)
        self.model = self.config.reranker.model
        
    def rerank(
        self,
        query: str,
        documents: List[FusedResult],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Rerank documents using LLM scoring.
        
        Args:
            query: The user query
            documents: List of FusedResult from hybrid retrieval
            top_k: Number of results to return (default from config)
            
        Returns:
            Reranked results sorted by score
        """
        top_k = top_k or self.config.retrieval.final_top_k
        results = []
        
        for doc in documents:
            score, reason = self._score_document(query, doc.content)
            
            results.append(RerankResult(
                content=doc.content,
                chapter_number=doc.chapter_number,
                chapter_title=doc.chapter_title,
                rerank_score=score,
                reason=reason,
                metadata=doc.metadata
            ))
        
        # Sort by score descending
        results.sort(key=lambda x: -x.rerank_score)
        
        return results[:top_k]
    
    def _score_document(self, query: str, passage: str) -> Tuple[float, str]:
        """Score a single document."""
        # Truncate passage if too long
        max_passage_len = 1500
        if len(passage) > max_passage_len:
            passage = passage[:max_passage_len] + "..."
        
        prompt = RERANK_PROMPT.format(query=query, passage=passage)
        
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                temperature=0.0,
                max_tokens=64,
                format="json"
            )
            
            # Aggressive cleanup and extraction
            content = response.content.strip()
            
            # Try to extract JSON
            parsed = self._extract_json(content)
            
            if parsed:
                score = float(parsed.get('score', 0))
                score = max(0.0, min(10.0, score))  # Clamp to 0-10
                reason = str(parsed.get('reason', ''))[:50]  # Truncate reason
                return score, reason
            else:
                logger.warning(f"Failed to parse rerank response: {content[:100]}")
                return 0.0, "parse_error"
                
        except Exception as e:
            logger.error(f"Rerank error: {e}")
            return 0.0, "error"
    
    def _extract_json(self, text: str) -> Optional[dict]:
        """Extract JSON from potentially messy LLM output."""
        # Try direct parse first
        try:
            return json.loads(text)
        except:
            pass
        
        # Try to find JSON object in text
        json_pattern = re.compile(r'\{[^{}]*\}')
        matches = json_pattern.findall(text)
        
        for match in matches:
            try:
                return json.loads(match)
            except:
                continue
        
        # Try to extract score manually
        score_match = re.search(r'"?score"?\s*:\s*(\d+(?:\.\d+)?)', text)
        if score_match:
            try:
                return {'score': float(score_match.group(1)), 'reason': 'extracted'}
            except:
                pass
        
        return None
    
    def get_min_score_threshold(self) -> float:
        """Get minimum rerank score for refusal logic."""
        return self.config.retrieval.min_rerank_score
