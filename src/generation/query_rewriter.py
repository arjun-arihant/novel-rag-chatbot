# Query Rewriter - Ultra-constrained query transformation (Qwen3 optimized)

import re
import logging
from typing import Optional

from ..ollama_client import OllamaClient, get_client
from ..config import get_config
from .prompts import QUERY_REWRITE_PROMPT

logger = logging.getLogger(__name__)


class QueryRewriter:
    """
    Ultra-constrained query rewriter.
    
    Optimized for Qwen3:
    - /no_think directive in prompt
    - Thinking tag stripping as fallback
    - Aggressive output validation
    """
    
    def __init__(self, client: Optional[OllamaClient] = None):
        self.config = get_config()
        self.client = client or get_client(self.config.query_rewriter.base_url)
        self.model = self.config.query_rewriter.model
        
    def rewrite(self, query: str) -> str:
        """
        Rewrite a query for better retrieval.
        
        Returns original query if rewriting fails or produces
        something problematic.
        """
        # Skip rewriting for very short queries
        if len(query.split()) <= 3:
            logger.debug(f"Query too short, skipping rewrite: {query}")
            return query
            
        prompt = QUERY_REWRITE_PROMPT.format(query=query)
        
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                temperature=0.0,
                max_tokens=64,  # Reduced - we only need a short query
                stop=["\n", "Query:", "Rewritten:"]
            )
            
            rewritten = response.content.strip()
            
            # Strip thinking tags if present (Qwen3 fallback)
            rewritten = self._strip_thinking_tags(rewritten)
            
            # Validation
            if not rewritten:
                logger.debug("Empty rewrite, using original")
                return query
                
            # Too short
            if len(rewritten) < 5:
                logger.debug("Rewrite too short, using original")
                return query
                
            # Too long (ran away)
            if len(rewritten) > 150:
                logger.debug("Rewrite too long, using original")
                return query
                
            # Multiple lines (explanation leaked)
            if '\n' in rewritten:
                rewritten = rewritten.split('\n')[0].strip()
                
            # Contains meta-text
            bad_patterns = ['here is', 'the rewritten', 'rewrite:', 'query:', 'i ', '/no_think']
            if any(p in rewritten.lower() for p in bad_patterns):
                logger.debug("Rewrite contains meta-text, using original")
                return query
            
            # Remove quotes if the whole thing is quoted
            if rewritten.startswith('"') and rewritten.endswith('"'):
                rewritten = rewritten[1:-1]
            
            logger.info(f"Rewrote: '{query}' -> '{rewritten}'")
            return rewritten
            
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}")
            return query
    
    def _strip_thinking_tags(self, text: str) -> str:
        """Remove Qwen3 thinking tags from output."""
        pattern = r'<think>.*?</think>'
        cleaned = re.sub(pattern, '', text, flags=re.DOTALL)
        return cleaned.strip()
