# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An advanced, privacy-first, offline RAG (Retrieval-Augmented Generation) chatbot for novels. This enhanced version features hybrid search, conversation memory, entity tracking, semantic caching, query enhancement, smart chunking, chapter summarization, and comprehensive analytics. All processing happens locally using LangChain + Ollama with no external API dependencies.

## Core Architecture

### Modular Design

The codebase is organized into a modular structure:

```
novel-rag-chatbot/
├── chatbot.py              # Main application with NovelRAGChatbot class
├── config.yaml             # Centralized configuration
├── utils/                  # Utility modules
│   ├── config_loader.py    # Configuration management
│   ├── entity_tracker.py   # Character/entity extraction
│   ├── smart_chunker.py    # Intelligent text chunking
│   ├── query_enhancer.py   # Query rewriting and enhancement
│   ├── hybrid_retriever.py # BM25 + semantic search
│   ├── summary_cache.py    # Chapter summarization
│   ├── semantic_cache.py   # Query result caching
│   ├── analytics.py        # Usage tracking and monitoring
│   └── validators.py       # System validation
├── tests/                  # Unit tests
└── novel.txt              # Input novel
```

### Data Flow Architecture

1. **Initialization Phase** (chatbot.py:54-107)
   - Load config.yaml with all settings
   - Run system validations (Ollama, models, novel format)
   - Initialize all utility components
   - Load novel and extract chapters

2. **Chapter Processing** (chatbot.py:242-305)
   - Extract chapters using regex pattern
   - Check chapter_index.json for already embedded chapters
   - For each new chapter:
     - Extract entities (characters, locations) → entity_cache.json
     - Generate summary → summary_cache.json
     - Smart chunk with sentence awareness
     - Add character metadata to chunks
   - Batch embed chunks → chroma_db/
   - Update chapter_index.json

3. **Query Processing** (chatbot.py:409-541)
   - Check semantic cache for similar past queries
   - Enhance query (rewrite, expand, resolve pronouns)
   - Apply search mode configuration (broad/precise/character_focused/timeline)
   - Hybrid retrieval (BM25 + semantic, configurable weights)
   - LLM generation with enhanced prompt
   - Extract chapter citations from source documents
   - Cache result for future similar queries
   - Log analytics (query time, retrieval quality)
   - Update conversation context
   - Append to chatlog.txt

4. **UI Layer** (chatbot.py:577-658)
   - Gradio web interface with search modes
   - CLI mode with commands (exit, clear, stats)
   - Real-time analytics display

### Key Components Deep Dive

#### NovelRAGChatbot Class (chatbot.py)

Main orchestrator that ties all components together:

- `__init__()`: Initializes system with config, validations, components
- `_load_and_process_novel()`: Handles novel loading and chapter processing
- `_process_chapters()`: Entity extraction, summarization, chunking pipeline
- `_embed_chunks()`: Batch embedding with progress tracking and backup
- `_setup_retrieval()`: Creates hybrid retriever with BM25 + semantic
- `_setup_qa_chain()`: Configures LangChain QA with enhanced prompt
- `ask_question()`: Main query processing with all enhancements
- `launch_ui()`: Gradio interface with search modes and analytics

#### EntityTracker (utils/entity_tracker.py)

Extracts and tracks characters, locations, and relationships:

- Uses regex to identify proper nouns (capitalized words)
- Tracks mentions per chapter
- Identifies characters by action verb proximity ("Alice said", "Bob walked")
- Builds relationship graph from co-occurrence patterns
- Enriches chunk metadata with character mentions
- Persistent cache: entity_cache.json

Key methods:
- `extract_entities_from_chapter()`: First pass extraction
- `finalize_entities()`: Filters by min_mention_count threshold
- `extract_relationships()`: Builds character relationship graph
- `add_character_to_metadata()`: Enriches chunks for better retrieval

#### SmartChunker (utils/smart_chunker.py)

Sentence-aware text chunking:

- Respects sentence boundaries (never splits mid-sentence)
- Configurable chunk size, overlap, min size
- Falls back to character splitting for long sentences
- Adds chunk metadata (index, total_chunks, size)
- Sliding window with sentence-level overlap
- Optional context window for enhanced chunks

Key difference from base RecursiveCharacterTextSplitter:
- `_smart_split()`: Sentence boundary detection
- `_get_overlap_sentences()`: Intelligent overlap selection
- `chunk_with_context()`: Adds surrounding context for better retrieval

#### QueryEnhancer (utils/query_enhancer.py)

Improves query quality before retrieval:

- **Pronoun resolution**: "What did he do?" → "What did Li Fan do?" using conversation context
- **Query rewriting**: Vague queries made specific via LLM
- **Query expansion**: Adds synonyms for better retrieval coverage
- **Conversation context**: Tracks last 10 query-answer pairs
- **Related question generation**: Suggests follow-up questions

Key methods:
- `enhance_query()`: Main enhancement pipeline
- `_resolve_pronouns()`: Uses `_get_recent_entities()` from context
- `_rewrite_query()`: LLM-based rewriting for clarity
- `_expand_query()`: Synonym mapping for common terms
- `add_to_context()`: Updates conversation memory

#### HybridRetriever (utils/hybrid_retriever.py)

Combines BM25 keyword and semantic vector search:

- **Semantic search**: ChromaDB vector similarity
- **BM25 search**: Keyword matching with TF-IDF variant
- **Ensemble**: Weighted combination (configurable in config.yaml)
- **MMR support**: Maximal Marginal Relevance for result diversity
- **Metadata filtering**: Filter by chapter, character mentions
- **Threshold filtering**: Only return results above similarity cutoff

Search types (retrieval_config['search_type']):
- `semantic`: Pure vector similarity
- `keyword`: BM25 only
- `hybrid`: Weighted combination (default, better results)

Key methods:
- `retrieve()`: Main retrieval with configurable search_type
- `_merge_results()`: Weighted score combination
- `retrieve_by_chapter()`: Filter to specific chapters
- `retrieve_with_character()`: Filter to character mentions

#### SummaryCache (utils/summary_cache.py)

Generates and caches chapter summaries:

- LLM-generated summaries (short/medium/long)
- Persistent cache: summary_cache.json
- Used for broad questions about story arcs
- `get_summary()`: Returns cached or generates new
- `generate_arc_summary()`: Multi-chapter storyline summaries
- `create_hierarchical_summary()`: Entire novel overview

Use case: "What happens in the novel?" retrieves summaries instead of raw chunks, then LLM synthesizes answer.

#### SemanticCache (utils/semantic_cache.py)

Caches query results based on semantic similarity:

- Exact match cache for repeated queries
- Semantic similarity matching (cosine similarity on embeddings)
- Configurable similarity threshold (default 0.95)
- TTL-based expiration (default 7 days)
- LRU eviction when cache full
- Tracks hit counts and access times

Significantly improves response time for similar/repeated questions.

#### Analytics (utils/analytics.py)

Comprehensive usage tracking:

- Query logs (response time, chapters used, confidence)
- Performance metrics (embedding time, retrieval time, LLM time)
- Error tracking (type, message, context)
- Retrieval quality (similarity scores, diversity metrics)
- Exportable reports (CLI, CSV, comprehensive text)

Access via: `python chatbot.py --analytics` or CLI command `stats`

#### Validators (utils/validators.py)

System health checks:

- Ollama running check (subprocess call to `ollama list`)
- Model availability check (nomic-embed-text, mistral:7b)
- Novel format validation (chapter regex, numbering)
- Directory structure check
- File permissions check
- Config validation (required keys, value ranges)

Runs automatically on startup if `validation.check_ollama_running: true` in config.

## Configuration System (config.yaml)

All settings centralized in YAML file. Key sections:

**paths**: File locations (novel, databases, caches, logs)

**models**: LLM and embedding model names

**chunking**: Size, overlap, sentence awareness

**retrieval**: Top-k, similarity threshold, search type, MMR settings

**search_modes**: Presets (broad/precise/character_focused/timeline) with custom top_k and thresholds

**memory**: Conversation memory settings

**query_enhancement**: Toggle rewriting, expansion, pronoun resolution

**summarization**: Enable/disable, cache settings, summary length

**entity_extraction**: Characters, locations, relationships, min mentions

**semantic_cache**: Similarity threshold, max size, TTL

**performance**: Batch sizes, analytics toggle

**ui**: Theme, display options, streaming

**validation**: Startup checks

## Development Commands

### Setup

```bash
# Create and activate virtual environment
python -m venv project_env
project_env\Scripts\activate  # Windows
# source project_env/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download Ollama models
ollama pull nomic-embed-text
ollama pull mistral:7b
```

### Running

```bash
# Standard web UI mode
python chatbot.py

# CLI mode
python chatbot.py --no-ui

# With custom novel
python chatbot.py --novel mybook.txt

# With different model
python chatbot.py --model llama2

# With search mode preset
python chatbot.py --search-mode precise

# Show analytics
python chatbot.py --analytics

# Custom config file
python chatbot.py --config custom_config.yaml
```

### Testing

```bash
# Run all tests
python -m unittest discover tests

# Run specific test
python -m unittest tests.test_chunking

# Run with verbose output
python -m unittest discover tests -v
```

### Database Management

```bash
# Reset database (re-embed everything)
rm -rf chroma_db chapter_index.json entity_cache.json summary_cache.json

# Clear caches only (keep embeddings)
rm entity_cache.json summary_cache.json semantic_cache.json analytics.json

# Restore from backup
rm -rf chroma_db && cp -r chroma_db_backup chroma_db
```

## Important Implementation Details

### Incremental Updates

The system tracks embedded chapters in `chapter_index.json`. When you add new chapters to `novel.txt`:

1. New chapters detected by comparing against index
2. Only new chapters are embedded (saves time)
3. Entity tracker updates with new character mentions
4. Summaries generated only for new chapters
5. Backup created before any database changes

### Conversation Context

QueryEnhancer maintains conversation context for pronoun resolution:

- Last 10 query-answer pairs stored in memory
- Entities extracted from answers
- When query contains pronoun (he/she/they), replaced with recent entity
- Example: "What did he do?" → "What did Li Fan do?" if Li Fan recently mentioned

### Hybrid Search Rationale

Pure semantic search can miss exact keyword matches. Pure BM25 can miss semantically similar content. Hybrid approach:

- BM25 weight: 0.3 (good for character names, specific terms)
- Semantic weight: 0.7 (good for conceptual queries)
- Combined scores ranked together
- Configurable per search mode in config.yaml

### Search Modes

Four presets for different query types:

1. **Broad** (top_k=10, threshold=0.6): General story questions
2. **Precise** (top_k=3, threshold=0.8): Specific fact checking
3. **Character Focused** (top_k=8, threshold=0.7): Character analysis
4. **Timeline** (top_k=7, threshold=0.65): Event sequences

Users select mode in Gradio dropdown or via CLI `--search-mode`.

### Smart Chunking Benefits

Standard character-based chunking can:
- Split mid-sentence reducing context quality
- Create awkward overlaps
- Produce inconsistent chunk sizes

SmartChunker:
- Never splits mid-sentence
- Overlap contains complete sentences
- Maintains semantic coherence
- Better retrieval quality (fewer fragmented results)

### Entity Metadata Enhancement

Each chunk's metadata includes:
- `chapter_title`: "Chapter 5: The Battle"
- `chapter_number`: "5"
- `characters`: ["Li Fan", "Zhang Haobo"]  # Added by EntityTracker
- `chunk_index`: 0
- `total_chunks`: 15
- `chunk_size`: 987

Enables character-focused retrieval: "Show all chunks mentioning Alice and Bob"

### Analytics Schema

analytics.json structure:

```json
{
  "queries": [
    {
      "timestamp": "2025-01-16T10:30:00",
      "query": "Who is Li Fan?",
      "answer_length": 342,
      "chapters_used": ["1", "3", "5"],
      "num_chapters": 3,
      "response_time": 2.45,
      "confidence": 0.87
    }
  ],
  "performance": [
    {"operation": "embedding", "duration": 15.3, "success": true}
  ],
  "errors": [
    {"error_type": "retrieval", "error_message": "...", "context": {}}
  ],
  "retrieval_quality": [
    {"num_retrieved": 5, "avg_similarity": 0.82, "chapters_diversity": 3}
  ]
}
```

### Semantic Cache Mechanics

When user asks: "Who is the protagonist?"

1. Check for exact match in cache
2. If no exact match, compute embedding of query
3. Compare with all cached query embeddings (cosine similarity)
4. If similarity > 0.95, return cached result
5. Otherwise, process query normally and cache result

Subsequent similar query "Who is the main character?" (similarity ~0.96) returns cached result instantly.

### Error Handling

All major operations wrapped in try-except:

- Novel loading failure → exit with error
- Embedding failure → logged to analytics, batch retried
- Query processing failure → error returned to user, logged
- Cache read/write failure → warning logged, continues without cache
- Validation failure → user prompted to continue or exit

## Modifying the System

### Adding a New Utility Module

1. Create `utils/my_module.py`
2. Add class/functions
3. Import in `utils/__init__.py`
4. Use in `chatbot.py` via `from utils import MyModule`

### Changing Search Behavior

Edit `config.yaml`:

```yaml
retrieval:
  search_type: "hybrid"  # or "semantic" or "keyword"
  bm25_weight: 0.4       # Increase for more keyword matching
  semantic_weight: 0.6   # Decrease accordingly
  use_mmr: true          # Toggle diversity
```

### Adding a New Search Mode

Edit `config.yaml`:

```yaml
search_modes:
  my_custom_mode:
    top_k: 6
    similarity_threshold: 0.75
    description: "My custom retrieval mode"
```

Then use: `python chatbot.py --search-mode my_custom_mode`

### Customizing the Prompt

Edit `chatbot.py:377-388`:

```python
prompt_template = """Your custom prompt here.

Context: {context}
Question: {question}
Answer:"""
```

### Adding Analytics Metrics

Edit `utils/analytics.py`:

1. Add new field to data structure
2. Create `log_my_metric()` method
3. Add to `get_comprehensive_report()`
4. Call from `chatbot.py` where metric is measured

### Disabling Features

Set in `config.yaml`:

```yaml
entity_extraction:
  enabled: false  # Disable entity tracking

summarization:
  enabled: false  # Disable chapter summaries

semantic_cache:
  enabled: false  # Disable query caching
```

## Common Issues and Solutions

### Import Errors

If you get `ModuleNotFoundError: No module named 'utils'`:

- Ensure `utils/__init__.py` exists and is properly structured
- Run from project root directory
- Check Python path: `sys.path.insert(0, os.path.dirname(__file__))`

### Ollama Connection Errors

```
Error: Ollama is not responding
```

Solution:
- Ensure Ollama is running: `ollama serve`
- Check models installed: `ollama list`
- Pull models: `ollama pull nomic-embed-text && ollama pull mistral:7b`

### ChromaDB Persistence Issues

```
Error: Could not load collection
```

Solution:
- Delete `chroma_db/` and re-embed
- Check disk space and permissions
- Ensure consistent embedding model between runs

### Memory Issues with Large Novels

If embedding runs out of memory:

- Reduce `performance.embedding_batch_size` in config (from 5 to 2)
- Reduce `chunking.size` (from 1000 to 800)
- Process chapters in smaller batches

### Query Enhancement Too Slow

If query rewriting adds too much latency:

- Set `query_enhancement.rewrite_enabled: false`
- Use faster model: `models.query_rewriter: "mistral:7b"` → `"llama2"`
- Reduce `query_enhancement.max_rewrites` from 3 to 1

## Testing

Unit tests in `tests/` directory:

- `test_chunking.py`: SmartChunker functionality
- `test_entity_tracker.py`: Entity extraction and tracking
- `test_config_loader.py`: Configuration loading and parsing

Add new tests following the pattern:

```python
import unittest
from utils.my_module import MyClass

class TestMyClass(unittest.TestCase):
    def setUp(self):
        self.instance = MyClass()

    def test_my_feature(self):
        result = self.instance.my_method()
        self.assertEqual(result, expected_value)
```

## Privacy and Offline Operation

System design ensures complete privacy:

- No external API calls (all via local Ollama)
- All data stored locally
- No telemetry or tracking
- Suitable for sensitive/proprietary content
- Can run fully offline (after initial model download)

## Performance Optimization

For optimal performance:

1. **Use GPU** if available (Ollama will auto-detect)
2. **Adjust batch sizes** in config for your hardware
3. **Enable semantic cache** for repeated query patterns
4. **Tune chunk size** - larger chunks = fewer embeddings but less precise retrieval
5. **Use precise mode** for specific queries to reduce LLM context size
6. **Pre-generate summaries** on initial load for faster broad queries
