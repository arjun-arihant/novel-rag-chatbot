"""
Analytics and monitoring for the RAG chatbot
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict, Counter


class Analytics:
    """Track and analyze chatbot usage and performance."""

    def __init__(self, analytics_path: str = "analytics.json"):
        """
        Initialize analytics tracker.

        Args:
            analytics_path: Path to analytics file
        """
        self.analytics_path = analytics_path
        self.data = {
            "queries": [],
            "performance": [],
            "errors": [],
            "retrieval_quality": [],
            "session_start": datetime.now().isoformat()
        }
        self.load()

    def load(self):
        """Load analytics from disk."""
        if os.path.exists(self.analytics_path):
            try:
                with open(self.analytics_path, 'r', encoding='utf-8') as f:
                    saved_data = json.load(f)
                    # Merge with current session
                    for key in ["queries", "performance", "errors", "retrieval_quality"]:
                        if key in saved_data:
                            self.data[key].extend(saved_data.get(key, []))
            except Exception as e:
                print(f"Warning: Could not load analytics: {e}")

    def save(self):
        """Save analytics to disk."""
        try:
            with open(self.analytics_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save analytics: {e}")

    def log_query(self, query: str, answer: str, chapters_used: List[str],
                 response_time: float, confidence: Optional[float] = None):
        """
        Log a query and response.

        Args:
            query: User query
            answer: Bot answer
            chapters_used: Chapters referenced
            response_time: Time taken to respond
            confidence: Optional confidence score
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer_length": len(answer),
            "chapters_used": chapters_used,
            "num_chapters": len(chapters_used),
            "response_time": response_time,
            "confidence": confidence
        }

        self.data["queries"].append(entry)
        self.save()

    def log_performance(self, operation: str, duration: float, success: bool = True):
        """
        Log performance metrics.

        Args:
            operation: Operation name
            duration: Time taken in seconds
            success: Whether operation succeeded
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "duration": duration,
            "success": success
        }

        self.data["performance"].append(entry)
        self.save()

    def log_error(self, error_type: str, error_message: str, context: Optional[Dict] = None):
        """
        Log an error.

        Args:
            error_type: Type of error
            error_message: Error message
            context: Optional context dict
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {}
        }

        self.data["errors"].append(entry)
        self.save()

    def log_retrieval_quality(self, query: str, num_retrieved: int,
                              avg_similarity: float, chapters_diversity: int):
        """
        Log retrieval quality metrics.

        Args:
            query: Query string
            num_retrieved: Number of documents retrieved
            avg_similarity: Average similarity score
            chapters_diversity: Number of unique chapters
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "num_retrieved": num_retrieved,
            "avg_similarity": avg_similarity,
            "chapters_diversity": chapters_diversity
        }

        self.data["retrieval_quality"].append(entry)
        self.save()

    def get_query_stats(self) -> Dict:
        """Get statistics about queries."""
        if not self.data["queries"]:
            return {"total_queries": 0}

        queries = self.data["queries"]
        response_times = [q["response_time"] for q in queries]
        chapters_used = [q["num_chapters"] for q in queries]

        return {
            "total_queries": len(queries),
            "avg_response_time": sum(response_times) / len(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "avg_chapters_per_query": sum(chapters_used) / len(chapters_used),
            "total_unique_queries": len(set(q["query"] for q in queries))
        }

    def get_popular_chapters(self, top_n: int = 10) -> List[tuple]:
        """
        Get most frequently referenced chapters.

        Args:
            top_n: Number of top chapters to return

        Returns:
            List of (chapter, count) tuples
        """
        chapter_counter = Counter()

        for query in self.data["queries"]:
            for chapter in query["chapters_used"]:
                chapter_counter[chapter] += 1

        return chapter_counter.most_common(top_n)

    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        if not self.data["performance"]:
            return {"total_operations": 0}

        perf = self.data["performance"]

        # Group by operation type
        by_operation = defaultdict(list)
        for entry in perf:
            by_operation[entry["operation"]].append(entry["duration"])

        stats = {"total_operations": len(perf)}

        for operation, durations in by_operation.items():
            stats[operation] = {
                "count": len(durations),
                "avg_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations)
            }

        return stats

    def get_error_summary(self) -> Dict:
        """Get error summary."""
        if not self.data["errors"]:
            return {"total_errors": 0}

        errors = self.data["errors"]
        error_types = Counter(e["error_type"] for e in errors)

        return {
            "total_errors": len(errors),
            "by_type": dict(error_types),
            "recent_errors": errors[-5:]  # Last 5 errors
        }

    def get_retrieval_quality_stats(self) -> Dict:
        """Get retrieval quality statistics."""
        if not self.data["retrieval_quality"]:
            return {"total_retrievals": 0}

        retrievals = self.data["retrieval_quality"]

        avg_similarity = [r["avg_similarity"] for r in retrievals]
        diversity = [r["chapters_diversity"] for r in retrievals]

        return {
            "total_retrievals": len(retrievals),
            "avg_similarity_score": sum(avg_similarity) / len(avg_similarity),
            "avg_chapter_diversity": sum(diversity) / len(diversity),
            "min_similarity": min(avg_similarity),
            "max_similarity": max(avg_similarity)
        }

    def get_hourly_distribution(self) -> Dict[int, int]:
        """Get query distribution by hour of day."""
        hourly = defaultdict(int)

        for query in self.data["queries"]:
            timestamp = datetime.fromisoformat(query["timestamp"])
            hourly[timestamp.hour] += 1

        return dict(hourly)

    def get_query_length_distribution(self) -> Dict:
        """Get distribution of query lengths."""
        if not self.data["queries"]:
            return {}

        lengths = [len(q["query"].split()) for q in self.data["queries"]]

        return {
            "avg_words": sum(lengths) / len(lengths),
            "min_words": min(lengths),
            "max_words": max(lengths),
            "median_words": sorted(lengths)[len(lengths) // 2]
        }

    def get_comprehensive_report(self) -> str:
        """
        Generate a comprehensive analytics report.

        Returns:
            Formatted report string
        """
        report = ["=" * 60]
        report.append("CHATBOT ANALYTICS REPORT")
        report.append("=" * 60)
        report.append("")

        # Query stats
        report.append("QUERY STATISTICS")
        report.append("-" * 60)
        query_stats = self.get_query_stats()
        for key, value in query_stats.items():
            if isinstance(value, float):
                report.append(f"  {key}: {value:.2f}")
            else:
                report.append(f"  {key}: {value}")
        report.append("")

        # Popular chapters
        report.append("TOP 10 REFERENCED CHAPTERS")
        report.append("-" * 60)
        for chapter, count in self.get_popular_chapters():
            report.append(f"  Chapter {chapter}: {count} references")
        report.append("")

        # Performance
        report.append("PERFORMANCE METRICS")
        report.append("-" * 60)
        perf_stats = self.get_performance_stats()
        for operation, stats in perf_stats.items():
            if operation != "total_operations":
                report.append(f"  {operation}:")
                if isinstance(stats, dict):
                    for key, value in stats.items():
                        if isinstance(value, float):
                            report.append(f"    {key}: {value:.3f}s")
                        else:
                            report.append(f"    {key}: {value}")
        report.append("")

        # Errors
        report.append("ERROR SUMMARY")
        report.append("-" * 60)
        error_summary = self.get_error_summary()
        report.append(f"  Total errors: {error_summary.get('total_errors', 0)}")
        if "by_type" in error_summary:
            for error_type, count in error_summary["by_type"].items():
                report.append(f"  {error_type}: {count}")
        report.append("")

        # Retrieval quality
        report.append("RETRIEVAL QUALITY")
        report.append("-" * 60)
        retrieval_stats = self.get_retrieval_quality_stats()
        for key, value in retrieval_stats.items():
            if isinstance(value, float):
                report.append(f"  {key}: {value:.3f}")
            else:
                report.append(f"  {key}: {value}")
        report.append("")

        report.append("=" * 60)

        return "\n".join(report)

    def export_to_csv(self, output_path: str):
        """
        Export query data to CSV.

        Args:
            output_path: Path to output CSV file
        """
        import csv

        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp", "Query", "Answer Length", "Chapters Used",
                    "Response Time", "Confidence"
                ])

                for query in self.data["queries"]:
                    writer.writerow([
                        query["timestamp"],
                        query["query"],
                        query["answer_length"],
                        ", ".join(query["chapters_used"]),
                        query["response_time"],
                        query.get("confidence", "N/A")
                    ])

            print(f"Analytics exported to {output_path}")
        except Exception as e:
            print(f"Error exporting analytics: {e}")

    def clear_old_data(self, days: int = 30):
        """
        Clear analytics data older than specified days.

        Args:
            days: Number of days to keep
        """
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(days=days)

        for key in ["queries", "performance", "errors", "retrieval_quality"]:
            self.data[key] = [
                entry for entry in self.data[key]
                if datetime.fromisoformat(entry["timestamp"]) > cutoff
            ]

        self.save()
