# Grounded Generator - Answer generation with refusal logic (Qwen3 optimized)

import re
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
    Grounded answer generator with refusal logic.
    
    Optimized for Qwen3:
    - /no_think directive in prompt
    - Thinking tag stripping from output
    - Less aggressive refusal thresholds
    
    Refusal conditions:
    1. No relevant chunks retrieved
    2. All rerank scores below threshold (lowered to 2.0)
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
        """
        # Log incoming results for debugging
        if reranked_results:
            logger.info(f"Generating with {len(reranked_results)} chunks, top score: {reranked_results[0].rerank_score:.1f}")
        else:
            logger.warning("No reranked results provided to generator")
        
        # Check refusal conditions (less strict now)
        should_refuse, reason = self._should_refuse(query, reranked_results)
        
        if should_refuse:
            logger.info(f"Refusing query: {reason}")
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
            
            # Strip thinking tags if present (Qwen3 fallback)
            answer = self._strip_thinking_tags(answer)
            
            # Clean up the answer
            answer = self._clean_answer(answer)
            
            # Extract cited chapters from answer
            chapters_cited = self._extract_citations(answer)
            
            # If no citations found, add them based on sources
            if not chapters_cited:
                chapters_cited = list(set(r.chapter_number for r in reranked_results))
            
            logger.info(f"Generated answer with {len(answer)} chars, citing chapters: {chapters_cited}")
            
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
        Two-layer refusal check (simplified from three-layer).
        
        Layer 1: No results
        Layer 2: Low rerank scores (threshold lowered to 2.0)
        
        Removed Layer 3 (entity coverage) as it was too aggressive.
        """
        # Layer 1: No results
        if not results:
            return True, "No relevant passages were found in the novel."
        
        # Layer 2: All scores very low
        min_score = self.config.retrieval.min_rerank_score
        top_score = results[0].rerank_score if results else 0
        
        # Only refuse if top score is very low (below threshold)
        if top_score < min_score:
            return True, f"The retrieved passages don't seem relevant enough (score: {top_score:.1f})."
        
        return False, ""
    
    def _strip_thinking_tags(self, text: str) -> str:
        """Remove Qwen3 thinking tags from output."""
        pattern = r'<think>.*?</think>'
        cleaned = re.sub(pattern, '', text, flags=re.DOTALL)
        return cleaned.strip()
    
    def _clean_answer(self, answer: str) -> str:
        """Clean up the generated answer."""
        # Remove any leading/trailing whitespace
        answer = answer.strip()
        
        # Remove markdown artifacts that shouldn't be there
        if answer.startswith('ANSWER:'):
            answer = answer[7:].strip()
        
        # Remove any duplicate citations like [Chapter 1][Chapter 1]
        answer = re.sub(r'(\[Chapter \d+\])\s*\1', r'\1', answer)
        
        return answer
    
    def _extract_citations(self, answer: str) -> List[int]:
        """Extract chapter numbers from [Chapter X] citations."""
        pattern = r'\[Chapter\s+(\d+)\]'
        matches = re.findall(pattern, answer, re.IGNORECASE)
        return list(set(int(m) for m in matches))
    
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
        
        # Track if we're in a thinking block
        buffer = ""
        in_thinking = False
        
        # Stream response
        for token in self.client.generate_stream(
            model=self.model,
            prompt=prompt,
            temperature=self.config.generator.temperature,
            max_tokens=self.config.generator.max_tokens
        ):
            buffer += token
            
            # Check for thinking tags
            if '<think>' in buffer and not in_thinking:
                in_thinking = True
                # Output anything before the think tag
                before = buffer.split('<think>')[0]
                if before:
                    yield before
                buffer = buffer.split('<think>', 1)[1] if '<think>' in buffer else ""
            
            if '</think>' in buffer and in_thinking:
                in_thinking = False
                # Discard thinking content
                buffer = buffer.split('</think>', 1)[1] if '</think>' in buffer else ""
            
            # Only yield if not in thinking block
            if not in_thinking and token and '<think>' not in token and '</think>' not in token:
                yield token
