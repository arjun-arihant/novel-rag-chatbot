"""
Tests for smart chunking functionality
"""
import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.smart_chunker import SmartChunker


class TestSmartChunker(unittest.TestCase):
    """Test smart chunking."""

    def setUp(self):
        """Set up test fixtures."""
        self.chunker = SmartChunker(chunk_size=100, chunk_overlap=20, respect_sentences=True)

    def test_basic_chunking(self):
        """Test basic text chunking."""
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = self.chunker.split_text(text)

        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)

    def test_respects_sentences(self):
        """Test that chunking respects sentence boundaries."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = self.chunker.split_text(text)

        for chunk in chunks:
            # Chunks should not end mid-sentence
            self.assertTrue(chunk.strip().endswith('.') or len(chunk) < 20)

    def test_document_creation(self):
        """Test document creation with metadata."""
        texts = ["Sample text one.", "Sample text two."]
        metadatas = [{"chapter": "1"}, {"chapter": "2"}]

        docs = self.chunker.create_documents(texts, metadatas)

        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].metadata["chapter"], "1")

if __name__ == '__main__':
    unittest.main()
