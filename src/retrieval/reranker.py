# LLM Reranker - Constrained scoring with JSON output (Qwen3 optimized)

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


# Reranker prompt from prompts.py
from ..generation.prompts import RERANK_PROMPT


class LLMReranker:
    """
    LLM-based reranker with strict scoring.
    
    Optimized for Qwen3:
    - /no_think directive in prompt
    - Thinking tag stripping as fallback
    - Robust JSON extraction
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
        
        # Log top scores for debugging
        if results:
            top_scores = [f"{r.rerank_score:.1f}" for r in results[:3]]
            logger.info(f"Rerank top scores: {', '.join(top_scores)}")
        
        return results[:top_k]
    
    def _score_document(self, query: str, passage: str) -> Tuple[float, str]:
        """Score a single document."""
        # Truncate passage if too long
        max_passage_len = 1200
        if len(passage) > max_passage_len:
            passage = passage[:max_passage_len] + "..."
        
        prompt = RERANK_PROMPT.format(query=query, passage=passage)
        
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                temperature=0.0,
                max_tokens=128,  # Increased for safety
                format="json"
            )
            
            # Get raw content
            content = response.content.strip()
            
            # Strip thinking tags if present (Qwen3 fallback)
            content = self._strip_thinking_tags(content)
            
            # Try to extract JSON
            parsed = self._extract_json(content)
            
            if parsed:
                score = float(parsed.get('score', 0))
                score = max(0.0, min(10.0, score))  # Clamp to 0-10
                reason = str(parsed.get('reason', ''))[:50]
                return score, reason
            else:
                logger.warning(f"Failed to parse rerank response: {content[:100]}")
                # Fallback: try to find any number in the response
                return self._fallback_score(content)
                
        except Exception as e:
            logger.error(f"Rerank error: {e}")
            return 5.0, "error_fallback"  # Neutral score on error
    
    def _strip_thinking_tags(self, text: str) -> str:
        """Remove Qwen3 thinking tags from output."""
        # Pattern to match <think>...</think> blocks
        pattern = r'<think>.*?</think>'
        cleaned = re.sub(pattern, '', text, flags=re.DOTALL)
        return cleaned.strip()
    
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
    
    def _fallback_score(self, text: str) -> Tuple[float, str]:
        """Fallback scoring when JSON fails - look for any number."""
        # Find any number in the text
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
        for num in numbers:
            try:
                score = float(num)
                if 0 <= score <= 10:
                    return score, "fallback_extracted"
            except:
                continue
        
        # Ultimate fallback - neutral score
        return 5.0, "parse_failed"
    
    def get_min_score_threshold(self) -> float:
        """Get minimum rerank score for refusal logic."""
        return self.config.retrieval.min_rerank_score
