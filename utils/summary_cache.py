"""
Chapter summary cache for improved broad question answering
"""
import json
import os
from typing import Dict, List, Optional
from langchain_community.llms import Ollama


class SummaryCache:
    """Cache and manage chapter summaries."""

    def __init__(self, cache_path: str = "summary_cache.json",
                 llm_model: str = "mistral:7b", summary_length: str = "medium"):
        """
        Initialize summary cache.

        Args:
            cache_path: Path to cache file
            llm_model: LLM model for generating summaries
            summary_length: Length of summaries (short, medium, long)
        """
        self.cache_path = cache_path
        self.llm = Ollama(model=llm_model, temperature=0.3)
        self.summary_length = summary_length
        self.summaries = {}
        self.load_cache()

    def load_cache(self):
        """Load summary cache from disk."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    self.summaries = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load summary cache: {e}")
                self.summaries = {}

    def save_cache(self):
        """Save summary cache to disk."""
        try:
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.summaries, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save summary cache: {e}")

    def get_summary(self, chapter_number: str, chapter_title: str,
                   chapter_text: str) -> str:
        """
        Get or generate summary for a chapter.

        Args:
            chapter_number: Chapter number
            chapter_title: Chapter title
            chapter_text: Chapter text content

        Returns:
            Chapter summary
        """
        cache_key = f"{chapter_number}_{chapter_title}"

        # Return cached summary if available
        if cache_key in self.summaries:
            return self.summaries[cache_key]["summary"]

        # Generate new summary
        summary = self._generate_summary(chapter_title, chapter_text)

        # Cache it
        self.summaries[cache_key] = {
            "chapter_number": chapter_number,
            "chapter_title": chapter_title,
            "summary": summary,
            "length": len(summary)
        }

        self.save_cache()

        return summary

    def _generate_summary(self, chapter_title: str, chapter_text: str) -> str:
        """
        Generate a summary for a chapter.

        Args:
            chapter_title: Chapter title
            chapter_text: Chapter text

        Returns:
            Generated summary
        """
        # Determine target length
        length_guide = {
            "short": "2-3 sentences",
            "medium": "4-6 sentences",
            "long": "8-10 sentences"
        }

        target_length = length_guide.get(self.summary_length, "4-6 sentences")

        # Truncate very long chapters for summarization
        text_preview = chapter_text[:4000] if len(chapter_text) > 4000 else chapter_text

        prompt = f"""Summarize this chapter from a novel. Focus on key events, character developments, and plot progression.

Chapter: {chapter_title}

Content:
{text_preview}

Provide a {target_length} summary:"""

        try:
            summary = self.llm.invoke(prompt).strip()
            return summary
        except Exception as e:
            print(f"Warning: Summary generation failed for {chapter_title}: {e}")
            # Return simple fallback summary
            return f"Content from {chapter_title}"

    def get_all_summaries(self) -> Dict[str, str]:
        """
        Get all cached summaries.

        Returns:
            Dict mapping chapter keys to summaries
        """
        return {key: value["summary"] for key, value in self.summaries.items()}

    def get_summaries_for_chapters(self, chapter_numbers: List[str]) -> str:
        """
        Get combined summaries for multiple chapters.

        Args:
            chapter_numbers: List of chapter numbers

        Returns:
            Combined summary text
        """
        combined = []

        for key, value in self.summaries.items():
            if value["chapter_number"] in chapter_numbers:
                combined.append(f"{value['chapter_title']}: {value['summary']}")

        return "\n\n".join(combined) if combined else ""

    def generate_arc_summary(self, start_chapter: int, end_chapter: int) -> str:
        """
        Generate a summary for a range of chapters (story arc).

        Args:
            start_chapter: Starting chapter number
            end_chapter: Ending chapter number

        Returns:
            Arc summary
        """
        # Collect summaries for the range
        arc_summaries = []

        for key, value in self.summaries.items():
            try:
                chapter_num = int(value["chapter_number"])
                if start_chapter <= chapter_num <= end_chapter:
                    arc_summaries.append(value["summary"])
            except (ValueError, KeyError):
                continue

        if not arc_summaries:
            return ""

        # Combine summaries
        combined_text = "\n\n".join(arc_summaries)

        prompt = f"""Based on these chapter summaries, provide an overall summary of this story arc:

{combined_text}

Overall arc summary (3-5 sentences):"""

        try:
            arc_summary = self.llm.invoke(prompt).strip()
            return arc_summary
        except Exception as e:
            print(f"Warning: Arc summary generation failed: {e}")
            return combined_text

    def invalidate_chapter(self, chapter_number: str, chapter_title: str):
        """
        Invalidate cached summary for a chapter.

        Args:
            chapter_number: Chapter number
            chapter_title: Chapter title
        """
        cache_key = f"{chapter_number}_{chapter_title}"
        if cache_key in self.summaries:
            del self.summaries[cache_key]
            self.save_cache()

    def clear_cache(self):
        """Clear all cached summaries."""
        self.summaries = {}
        self.save_cache()

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "total_summaries": len(self.summaries),
            "summary_length_setting": self.summary_length,
            "average_summary_length": sum(s["length"] for s in self.summaries.values()) / len(self.summaries) if self.summaries else 0
        }

    def create_hierarchical_summary(self) -> str:
        """
        Create a hierarchical summary of the entire novel.

        Returns:
            Hierarchical summary
        """
        if not self.summaries:
            return "No summaries available."

        # Sort by chapter number
        sorted_summaries = sorted(
            self.summaries.items(),
            key=lambda x: int(x[1]["chapter_number"]) if x[1]["chapter_number"].isdigit() else 0
        )

        # Group into arcs (every 5 chapters)
        arc_size = 5
        arcs = []

        for i in range(0, len(sorted_summaries), arc_size):
            arc_chapters = sorted_summaries[i:i + arc_size]
            arc_text = "\n".join([f"  - {v['chapter_title']}: {v['summary'][:100]}..."
                                 for _, v in arc_chapters])
            arcs.append(f"Chapters {i+1}-{min(i+arc_size, len(sorted_summaries))}:\n{arc_text}")

        return "\n\n".join(arcs)
