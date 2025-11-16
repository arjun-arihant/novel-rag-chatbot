"""
Tests for entity tracking functionality
"""
import unittest
import sys
import os
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.entity_tracker import EntityTracker


class TestEntityTracker(unittest.TestCase):
    """Test entity tracking."""

    def setUp(self):
        """Set up test fixtures."""
        # Use temporary file for cache
        self.temp_cache = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        self.temp_cache.close()
        self.tracker = EntityTracker(cache_path=self.temp_cache.name, min_mentions=1)

    def tearDown(self):
        """Clean up."""
        if os.path.exists(self.temp_cache.name):
            os.unlink(self.temp_cache.name)

    def test_extract_proper_nouns(self):
        """Test extraction of proper nouns."""
        text = "Harry Potter went to Hogwarts. Hermione Granger was his friend."
        self.tracker.extract_entities_from_chapter(text, "1")

        self.assertIn("Harry", self.tracker.entities["mentions"])
        self.assertIn("Hermione", self.tracker.entities["mentions"])

    def test_finalize_entities(self):
        """Test entity finalization."""
        text = "Alice spoke to Bob. Bob replied to Alice. Alice and Bob were friends."
        self.tracker.extract_entities_from_chapter(text, "1")
        self.tracker.finalize_entities()

        characters = self.tracker.get_all_characters()
        self.assertIn("Alice", characters)
        self.assertIn("Bob", characters)

    def test_character_search(self):
        """Test character search."""
        self.tracker.entities["characters"] = {
            "Alice": {},
            "Bob": {},
            "Charlie": {}
        }

        results = self.tracker.search_character("ali")
        self.assertIn("Alice", results)
        self.assertEqual(len(results), 1)


if __name__ == '__main__':
    unittest.main()
