# Query Rewriter - Ultra-constrained query transformation

import logging
from typing import Optional

from ..ollama_client import OllamaClient, get_client
from ..config import get_config
from .prompts import QUERY_REWRITE_PROMPT

logger = logging.getLogger(__name__)


class QueryRewriter:
    """
    Ultra-constrained query rewriter.
    
    - Temperature 0.0 for deterministic output
    - Max 128 tokens
    - Aggressive stop tokens
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
            return query
            
        prompt = QUERY_REWRITE_PROMPT.format(query=query)
        
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                temperature=0.0,
                max_tokens=128,
                stop=list(self.config.query_rewriter.stop_tokens)
            )
            
            rewritten = response.content.strip()
            
            # Validation
            if not rewritten:
                return query
                
            # Too short
            if len(rewritten) < 5:
                return query
                
            # Too long (ran away)
            if len(rewritten) > 200:
                return query
                
            # Multiple lines (explanation leaked)
            if '\n' in rewritten:
                rewritten = rewritten.split('\n')[0].strip()
                
            # Contains meta-text
            bad_patterns = ['here is', 'the rewritten', 'rewrite:', 'query:']
            if any(p in rewritten.lower() for p in bad_patterns):
                return query
            
            logger.debug(f"Rewrote: '{query}' -> '{rewritten}'")
            return rewritten
            
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}")
            return query
