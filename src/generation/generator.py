# Grounded Generator - Answer generation with refusal logic

import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass

from ..ollama_client import OllamaClient, get_client
from ..config import get_config
from ..retrieval.reranker import RerankResult
from ..ingestion.metadata import EntityExtractor
from .prompts import ANSWER_PROMPT, REFUSAL_TEMPLATE, format_context

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of answer generation."""
    answer: str
    refused: bool
    refusal_reason: str
    chapters_cited: List[int]
    sources: List[dict]


class GroundedGenerator:
    """
    Grounded answer generator with three-layer refusal logic.
    
    Refusal conditions:
    1. No relevant chunks retrieved
    2. All rerank scores below threshold
    3. Query entities not found in chunks (NEW)
    """
    
    def __init__(
        self,
        client: Optional[OllamaClient] = None,
        entity_extractor: Optional[EntityExtractor] = None
    ):
        self.config = get_config()
        self.client = client or get_client(self.config.generator.base_url)
        self.model = self.config.generator.model
        self.entity_extractor = entity_extractor or EntityExtractor()
        
    def generate(
        self,
        query: str,
        reranked_results: List[RerankResult]
    ) -> GenerationResult:
        """
        Generate a grounded answer.
        
        Args:
            query: The user's question
            reranked_results: Results from LLM reranker
            
        Returns:
            GenerationResult with answer or refusal
        """
        # Check refusal conditions
        should_refuse, reason = self._should_refuse(query, reranked_results)
        
        if should_refuse:
            return GenerationResult(
                answer=REFUSAL_TEMPLATE.format(reason=reason),
                refused=True,
                refusal_reason=reason,
                chapters_cited=[],
                sources=[]
            )
        
        # Prepare context
        context_chunks = [
            {
                'content': r.content,
                'chapter_title': r.chapter_title,
                'chapter_number': r.chapter_number
            }
            for r in reranked_results
        ]
        
        context = format_context(context_chunks)
        
        # Generate answer
        prompt = ANSWER_PROMPT.format(context=context, question=query)
        
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                temperature=self.config.generator.temperature,
                max_tokens=self.config.generator.max_tokens
            )
            
            answer = response.content.strip()
            
            # Extract cited chapters from answer
            chapters_cited = self._extract_citations(answer)
            
            # If no citations found, add them based on sources
            if not chapters_cited:
                chapters_cited = list(set(r.chapter_number for r in reranked_results))
            
            return GenerationResult(
                answer=answer,
                refused=False,
                refusal_reason="",
                chapters_cited=chapters_cited,
                sources=[
                    {
                        'content': r.content[:300] + "..." if len(r.content) > 300 else r.content,
                        'chapter_number': r.chapter_number,
                        'chapter_title': r.chapter_title,
                        'score': r.rerank_score
                    }
                    for r in reranked_results
                ]
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return GenerationResult(
                answer=f"Error generating answer: {str(e)}",
                refused=True,
                refusal_reason="generation_error",
                chapters_cited=[],
                sources=[]
            )
    
    def _should_refuse(
        self,
        query: str,
        results: List[RerankResult]
    ) -> Tuple[bool, str]:
        """
        Three-layer refusal check.
        
        Layer 1: No results
        Layer 2: Low rerank scores
        Layer 3: Query entities missing from chunks
        """
        # Layer 1: No results
        if not results:
            return True, "No relevant passages were found in the novel."
        
        # Layer 2: All scores below threshold
        min_score = self.config.retrieval.min_rerank_score
        top_score = results[0].rerank_score if results else 0
        
        if top_score < min_score:
            return True, f"The retrieved passages don't contain directly relevant information (confidence too low)."
        
        # Layer 3: Entity coverage check
        query_entities = self.entity_extractor.extract_from_query(query)
        
        if query_entities:
            # Get combined text from top 3 results
            combined_text = " ".join(r.content.lower() for r in results[:3])
            
            # Check which entities are missing
            missing_entities = [
                entity for entity in query_entities
                if entity.lower() not in combined_text
            ]
            
            # If ALL query entities are missing, refuse
            if len(missing_entities) == len(query_entities) and len(query_entities) > 0:
                return True, f"Could not find information about {', '.join(missing_entities)} in the retrieved context."
        
        return False, ""
    
    def _extract_citations(self, answer: str) -> List[int]:
        """Extract chapter numbers from [Chapter X] citations."""
        import re
        pattern = r'\[Chapter\s+(\d+)\]'
        matches = re.findall(pattern, answer, re.IGNORECASE)
        return [int(m) for m in matches]
    
    def generate_stream(
        self,
        query: str,
        reranked_results: List[RerankResult]
    ):
        """Stream answer generation (for UI)."""
        # Check refusal first
        should_refuse, reason = self._should_refuse(query, reranked_results)
        
        if should_refuse:
            yield REFUSAL_TEMPLATE.format(reason=reason)
            return
        
        # Prepare context
        context_chunks = [
            {
                'content': r.content,
                'chapter_title': r.chapter_title,
                'chapter_number': r.chapter_number
            }
            for r in reranked_results
        ]
        
        context = format_context(context_chunks)
        prompt = ANSWER_PROMPT.format(context=context, question=query)
        
        # Stream response
        for token in self.client.generate_stream(
            model=self.model,
            prompt=prompt,
            temperature=self.config.generator.temperature,
            max_tokens=self.config.generator.max_tokens
        ):
            yield token
