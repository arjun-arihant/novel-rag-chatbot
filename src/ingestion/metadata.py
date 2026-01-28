# Metadata Extraction - Chapter and Entity extraction

import re
import logging
from typing import List, Set, Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ChapterMetadata:
    """Metadata for a chapter."""
    number: int
    title: str
    word_count: int
    char_count: int
    entities: Set[str] = field(default_factory=set)


class ChapterExtractor:
    """Extract and manage chapter metadata."""
    
    def __init__(self):
        self.chapters: Dict[int, ChapterMetadata] = {}
        
    def add_chapter(self, number: int, title: str, content: str):
        """Add a chapter's metadata."""
        self.chapters[number] = ChapterMetadata(
            number=number,
            title=title,
            word_count=len(content.split()),
            char_count=len(content)
        )
        
    def get_chapter(self, number: int) -> Optional[ChapterMetadata]:
        """Get chapter metadata."""
        return self.chapters.get(number)
    
    def get_all(self) -> List[ChapterMetadata]:
        """Get all chapters sorted by number."""
        return sorted(self.chapters.values(), key=lambda c: c.number)


class EntityExtractor:
    """
    Extract named entities from text.
    Simple pattern-based extraction (no ML dependencies).
    """
    
    # Common words that look like names but aren't
    STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'is', 'it', 'this', 'that', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'ought', 'used', 'i', 'you', 'he', 'she', 'we', 'they', 'me', 'him',
        'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their',
        'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how',
        'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
        'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
        'very', 'just', 'also', 'now', 'here', 'there', 'then', 'once',
        'chapter', 'part', 'book', 'section', 'volume', 'page',
        'said', 'asked', 'replied', 'answered', 'thought', 'knew', 'felt',
        'looked', 'saw', 'heard', 'went', 'came', 'made', 'took', 'got',
        'however', 'therefore', 'thus', 'hence', 'meanwhile', 'furthermore',
        'although', 'though', 'while', 'whereas', 'because', 'since', 'unless',
        'until', 'after', 'before', 'during', 'through', 'between', 'under',
        'above', 'below', 'from', 'into', 'out', 'off', 'over', 'down', 'up',
    }
    
    # Title words that precede names
    TITLE_WORDS = {'mr', 'mrs', 'ms', 'miss', 'dr', 'professor', 'sir', 'lady', 'lord', 'king', 'queen', 'prince', 'princess'}
    
    # Pattern for potential names (capitalized words)
    NAME_PATTERN = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b')
    
    def __init__(self, min_occurrences: int = 2):
        self.min_occurrences = min_occurrences
        self.entity_counts: Dict[str, int] = defaultdict(int)
        self.entity_chapters: Dict[str, Set[int]] = defaultdict(set)
        
    def extract_from_text(self, text: str, chapter_number: int) -> Set[str]:
        """Extract entities from text."""
        entities = set()
        
        # Find all capitalized words/phrases
        matches = self.NAME_PATTERN.findall(text)
        
        for match in matches:
            # Clean and validate
            entity = match.strip()
            
            # Skip if it's a stopword
            if entity.lower() in self.STOPWORDS:
                continue
                
            # Skip if too short
            if len(entity) < 2:
                continue
                
            # Skip if it's just a title
            if entity.lower() in self.TITLE_WORDS:
                continue
            
            entities.add(entity)
            self.entity_counts[entity] += 1
            self.entity_chapters[entity].add(chapter_number)
            
        return entities
    
    def get_significant_entities(self) -> List[str]:
        """Get entities that appear frequently enough."""
        return [
            entity for entity, count in self.entity_counts.items()
            if count >= self.min_occurrences
        ]
    
    def get_entity_chapters(self, entity: str) -> Set[int]:
        """Get chapters where an entity appears."""
        return self.entity_chapters.get(entity, set())
    
    def is_critical_entity(self, entity: str) -> bool:
        """
        Check if an entity is "critical" (appears frequently).
        Used for refusal logic - if query mentions a critical entity,
        retrieved chunks should mention it too.
        """
        return self.entity_counts.get(entity, 0) >= self.min_occurrences
    
    def extract_from_query(self, query: str) -> List[str]:
        """Extract potential entity names from a query."""
        matches = self.NAME_PATTERN.findall(query)
        entities = []
        
        for match in matches:
            entity = match.strip()
            if entity.lower() not in self.STOPWORDS and len(entity) >= 2:
                entities.append(entity)
                
        return entities
    
    def get_stats(self) -> Dict:
        """Get extraction statistics."""
        return {
            "total_entities": len(self.entity_counts),
            "significant_entities": len(self.get_significant_entities()),
            "top_entities": sorted(
                self.entity_counts.items(),
                key=lambda x: -x[1]
            )[:20]
        }
