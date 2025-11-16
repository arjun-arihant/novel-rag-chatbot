"""
Query enhancer for rewriting and expanding user queries
"""
import re
from typing import List, Dict, Optional
from langchain_community.llms import Ollama


class QueryEnhancer:
    """Enhance user queries for better retrieval."""

    def __init__(self, llm_model: str = "mistral:7b", entity_tracker=None):
        """
        Initialize query enhancer.

        Args:
            llm_model: LLM model for query rewriting
            entity_tracker: EntityTracker instance for context
        """
        self.llm = Ollama(model=llm_model, temperature=0.3)
        self.entity_tracker = entity_tracker
        self.conversation_context = []

    def enhance_query(self, query: str, rewrite: bool = True,
                     expand: bool = True, resolve_pronouns: bool = True) -> Dict[str, any]:
        """
        Enhance a query with multiple techniques.

        Args:
            query: Original query
            rewrite: Whether to rewrite the query
            expand: Whether to expand the query
            resolve_pronouns: Whether to resolve pronouns

        Returns:
            Dict with enhanced_query and metadata
        """
        enhanced = query
        metadata = {
            "original_query": query,
            "modifications": []
        }

        # Resolve pronouns using conversation context
        if resolve_pronouns and self.conversation_context:
            enhanced = self._resolve_pronouns(enhanced, metadata)

        # Rewrite query for clarity
        if rewrite:
            enhanced = self._rewrite_query(enhanced, metadata)

        # Expand with synonyms/related terms
        if expand:
            expanded_terms = self._expand_query(enhanced)
            if expanded_terms:
                metadata["expanded_terms"] = expanded_terms

        metadata["enhanced_query"] = enhanced

        return {
            "query": enhanced,
            "metadata": metadata
        }

    def _resolve_pronouns(self, query: str, metadata: Dict) -> str:
        """
        Resolve pronouns like 'he', 'she', 'they' using context.

        Args:
            query: Query text
            metadata: Metadata dict to update

        Returns:
            Query with resolved pronouns
        """
        pronouns = {
            'he': 'male',
            'she': 'female',
            'him': 'male',
            'her': 'female',
            'his': 'male',
            'hers': 'female',
            'they': 'plural',
            'them': 'plural',
            'their': 'plural'
        }

        query_lower = query.lower()
        found_pronouns = []

        for pronoun in pronouns:
            if re.search(r'\b' + pronoun + r'\b', query_lower):
                found_pronouns.append(pronoun)

        if not found_pronouns or not self.conversation_context:
            return query

        # Get last mentioned entities from context
        last_entities = self._get_recent_entities()

        if last_entities:
            # Replace first pronoun with most recent entity
            enhanced = query
            for pronoun in found_pronouns[:1]:  # Only replace first pronoun
                pattern = r'\b' + pronoun + r'\b'
                enhanced = re.sub(pattern, last_entities[0], enhanced, count=1, flags=re.IGNORECASE)
                metadata["modifications"].append(f"Resolved '{pronoun}' to '{last_entities[0]}'")

            return enhanced

        return query

    def _get_recent_entities(self, n: int = 3) -> List[str]:
        """Get recently mentioned entities from conversation context."""
        entities = []

        # Look through recent context in reverse
        for entry in reversed(self.conversation_context[-5:]):
            if 'entities' in entry:
                entities.extend(entry['entities'])

        # Return unique entities in order of recency
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)
                if len(unique_entities) >= n:
                    break

        return unique_entities

    def _rewrite_query(self, query: str, metadata: Dict) -> str:
        """
        Rewrite query for better retrieval.

        Args:
            query: Original query
            metadata: Metadata dict to update

        Returns:
            Rewritten query
        """
        # Skip rewriting for very short queries
        if len(query.split()) <= 3:
            return query

        prompt = f"""Rewrite this question to be more specific and retrieval-friendly while keeping the same meaning.
Make it clearer and more detailed, but keep it concise.

Original question: {query}

Rewritten question:"""

        try:
            rewritten = self.llm.invoke(prompt).strip()

            # Validate rewrite isn't too different or broken
            if rewritten and len(rewritten) > 5 and '?' in rewritten:
                metadata["modifications"].append(f"Rewritten from: '{query}'")
                return rewritten
        except Exception as e:
            print(f"Warning: Query rewriting failed: {e}")

        return query

    def _expand_query(self, query: str) -> List[str]:
        """
        Expand query with related terms and synonyms.

        Args:
            query: Query to expand

        Returns:
            List of expansion terms
        """
        # Extract key terms
        words = query.lower().split()

        # Common synonyms for novel-related terms
        expansion_map = {
            'happen': ['occur', 'take place', 'transpire'],
            'say': ['state', 'mention', 'declare', 'express'],
            'think': ['believe', 'consider', 'ponder', 'reflect'],
            'go': ['travel', 'journey', 'move', 'head'],
            'see': ['observe', 'witness', 'notice', 'perceive'],
            'character': ['person', 'protagonist', 'figure', 'individual'],
            'fight': ['battle', 'combat', 'conflict', 'confrontation'],
            'die': ['perish', 'death', 'deceased', 'demise'],
            'love': ['affection', 'romance', 'feelings', 'attachment'],
            'power': ['ability', 'strength', 'capability', 'skill']
        }

        expansions = []
        for word in words:
            word_clean = word.strip('?.,!').lower()
            if word_clean in expansion_map:
                expansions.extend(expansion_map[word_clean])

        return list(set(expansions))[:5]  # Limit to 5 unique expansions

    def add_to_context(self, query: str, answer: str, entities: List[str]):
        """
        Add query-answer pair to conversation context.

        Args:
            query: User query
            answer: Bot answer
            entities: Entities mentioned
        """
        self.conversation_context.append({
            'query': query,
            'answer': answer,
            'entities': entities
        })

        # Keep only last 10 interactions
        if len(self.conversation_context) > 10:
            self.conversation_context = self.conversation_context[-10:]

    def clear_context(self):
        """Clear conversation context."""
        self.conversation_context = []

    def get_context_summary(self) -> str:
        """
        Get a summary of the conversation context.

        Returns:
            Summary string
        """
        if not self.conversation_context:
            return "No previous conversation."

        recent = self.conversation_context[-3:]
        summary = "Recent conversation:\n"
        for i, entry in enumerate(recent, 1):
            summary += f"{i}. Q: {entry['query'][:100]}...\n"

        return summary

    def generate_related_questions(self, query: str, answer: str) -> List[str]:
        """
        Generate related follow-up questions.

        Args:
            query: Original query
            answer: Answer provided

        Returns:
            List of related questions
        """
        prompt = f"""Based on this question and answer, suggest 3 related follow-up questions:

Question: {query}
Answer: {answer[:300]}...

Related questions:
1."""

        try:
            response = self.llm.invoke(prompt).strip()
            # Parse numbered list
            questions = re.findall(r'\d+\.\s*(.+?)(?=\d+\.|$)', "1. " + response, re.DOTALL)
            return [q.strip() for q in questions if len(q.strip()) > 10][:3]
        except Exception as e:
            print(f"Warning: Related question generation failed: {e}")
            return []
