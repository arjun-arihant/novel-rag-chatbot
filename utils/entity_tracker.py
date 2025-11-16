"""
Entity tracker for extracting and tracking characters, locations, and relationships
"""
import json
import os
import re
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter


class EntityTracker:
    """Track characters, locations, and relationships in the novel."""

    def __init__(self, cache_path: str = "entity_cache.json", min_mentions: int = 2):
        """
        Initialize entity tracker.

        Args:
            cache_path: Path to cache file
            min_mentions: Minimum mentions to track an entity
        """
        self.cache_path = cache_path
        self.min_mentions = min_mentions
        self.entities = {
            "characters": {},
            "locations": {},
            "relationships": [],
            "mentions": defaultdict(lambda: defaultdict(int))
        }
        self.load_cache()

    def load_cache(self):
        """Load entity cache from disk."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.entities = data
                    # Convert mentions back to defaultdict
                    if "mentions" in data:
                        self.entities["mentions"] = defaultdict(
                            lambda: defaultdict(int),
                            {k: defaultdict(int, v) for k, v in data["mentions"].items()}
                        )
            except Exception as e:
                print(f"Warning: Could not load entity cache: {e}")

    def save_cache(self):
        """Save entity cache to disk."""
        try:
            # Convert defaultdict to regular dict for JSON serialization
            data = dict(self.entities)
            data["mentions"] = {k: dict(v) for k, v in self.entities["mentions"].items()}

            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save entity cache: {e}")

    def extract_entities_from_chapter(self, chapter_text: str, chapter_number: str):
        """
        Extract entities from a chapter.

        Args:
            chapter_text: The chapter text
            chapter_number: Chapter identifier
        """
        # Extract proper nouns (capitalized words)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', chapter_text)

        for noun in proper_nouns:
            # Skip common words
            if noun.lower() in {'chapter', 'the', 'a', 'an', 'this', 'that'}:
                continue

            self.entities["mentions"][noun][chapter_number] += 1

        # Detect potential character names (words appearing with action verbs nearby)
        character_patterns = [
            r'\b([A-Z][a-z]+)\s+(?:said|asked|replied|thought|wondered|felt|saw|heard|ran|walked|looked)',
            r'(?:said|asked|replied)\s+([A-Z][a-z]+)',
        ]

        for pattern in character_patterns:
            matches = re.findall(pattern, chapter_text)
            for match in matches:
                if match and len(match) > 1:
                    self.entities["mentions"][match][chapter_number] += 2  # Higher weight for action verbs

    def finalize_entities(self):
        """Process mentions and create final entity lists."""
        # Characters: entities with sufficient mentions
        for entity, chapters in self.entities["mentions"].items():
            total_mentions = sum(chapters.values())
            if total_mentions >= self.min_mentions:
                chapter_list = list(chapters.keys())
                self.entities["characters"][entity] = {
                    "total_mentions": total_mentions,
                    "appears_in_chapters": chapter_list,
                    "first_appearance": min(chapter_list, key=int) if chapter_list else None
                }

    def get_character_info(self, character_name: str) -> Dict:
        """Get information about a character."""
        return self.entities["characters"].get(character_name, {})

    def get_characters_in_chapter(self, chapter_number: str) -> List[str]:
        """Get all characters mentioned in a chapter."""
        characters = []
        for char, info in self.entities["characters"].items():
            if chapter_number in info.get("appears_in_chapters", []):
                characters.append(char)
        return characters

    def get_all_characters(self) -> List[str]:
        """Get all tracked characters."""
        return sorted(self.entities["characters"].keys())

    def search_character(self, query: str) -> List[str]:
        """
        Search for characters matching query.

        Args:
            query: Search term

        Returns:
            List of matching character names
        """
        query_lower = query.lower()
        matches = []
        for char in self.entities["characters"].keys():
            if query_lower in char.lower():
                matches.append(char)
        return matches

    def extract_relationships(self, chapter_text: str, chapter_number: str):
        """
        Extract relationships between characters.

        Args:
            chapter_text: The chapter text
            chapter_number: Chapter identifier
        """
        characters = self.get_all_characters()

        # Look for characters appearing in the same sentence
        sentences = re.split(r'[.!?]+', chapter_text)

        for sentence in sentences:
            chars_in_sentence = [char for char in characters if char in sentence]

            if len(chars_in_sentence) >= 2:
                # Check for relationship keywords
                relationship_keywords = ['friend', 'enemy', 'lover', 'parent', 'child',
                                        'master', 'disciple', 'ally', 'rival', 'companion']

                for keyword in relationship_keywords:
                    if keyword in sentence.lower():
                        for i, char1 in enumerate(chars_in_sentence):
                            for char2 in chars_in_sentence[i+1:]:
                                relationship = {
                                    "character1": char1,
                                    "character2": char2,
                                    "type": keyword,
                                    "chapter": chapter_number,
                                    "context": sentence.strip()
                                }
                                if relationship not in self.entities["relationships"]:
                                    self.entities["relationships"].append(relationship)

    def get_character_relationships(self, character_name: str) -> List[Dict]:
        """Get all relationships for a character."""
        relationships = []
        for rel in self.entities["relationships"]:
            if rel["character1"] == character_name or rel["character2"] == character_name:
                relationships.append(rel)
        return relationships

    def add_character_to_metadata(self, text: str, metadata: Dict) -> Dict:
        """
        Add character mentions to chunk metadata.

        Args:
            text: Chunk text
            metadata: Existing metadata

        Returns:
            Enhanced metadata
        """
        characters_in_chunk = []
        for char in self.get_all_characters():
            if char in text:
                characters_in_chunk.append(char)

        if characters_in_chunk:
            metadata["characters"] = characters_in_chunk

        return metadata
