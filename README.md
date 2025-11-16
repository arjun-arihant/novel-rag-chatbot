# 📚 Enhanced Local RAG Chatbot for Novels

A **private, offline AI chatbot** that can answer questions about any novel using **advanced Retrieval-Augmented Generation (RAG)** techniques. Built with [LangChain](https://github.com/langchain-ai/langchain) and [Ollama](https://ollama.com/), this tool intelligently indexes, embeds, and chats over large novels with sophisticated features like hybrid search, entity tracking, conversation memory, and much more — all while respecting your privacy and working completely offline.

---

## ✨ Features

### Core RAG Capabilities

- 🔍 **Chapter-aware metadata tagging**
  Automatically extracts and stores chapter titles/numbers for context-aware answers with precise citations.

- 🧠 **Dual-model architecture**
  Uses `nomic-embed-text` for fast embedding + `mistral:7b` for accurate answering.

- 💾 **Persistent, update-safe vector database**
  Avoids re-embedding already indexed content and backs up your database before changes.

- ➕ **Automatic detection of new chapters**
  Seamlessly updates the chatbot with new content when a novel is updated.

### Advanced Features

- 🔎 **Hybrid Search (BM25 + Semantic)**
  Combines keyword matching and semantic similarity for superior retrieval accuracy. Configurable weights for different query types.

- 💬 **Conversation Memory**
  Tracks conversation context for pronoun resolution and follow-up questions. "What did he do?" automatically resolves to the last mentioned character.

- ✏️ **Query Enhancement**
  Automatically rewrites vague queries, expands with synonyms, and resolves pronouns using conversation context.

- 🎭 **Entity Tracking**
  Extracts characters, locations, and relationships. Tracks which chapters each character appears in and enables character-focused queries.

- 📝 **Smart Chunking**
  Sentence-aware text splitting that never breaks mid-sentence, maintaining semantic coherence for better retrieval.

- 📊 **Chapter Summarization**
  Auto-generates and caches summaries for each chapter. Enables quick answers to broad questions about story arcs.

- 💾 **Semantic Caching**
  Caches similar queries to dramatically speed up repeated or similar questions (e.g., "Who is the protagonist?" ≈ "Who is the main character?").

- 🎯 **Multiple Search Modes**
  - **Broad**: Cast wide net for general questions (top_k=10)
  - **Precise**: Exact matches for specific facts (top_k=3, high threshold)
  - **Character Focused**: Optimized for character analysis (top_k=8)
  - **Timeline**: Track events chronologically (top_k=7)

- 📈 **Analytics & Monitoring**
  Comprehensive tracking of query performance, retrieval quality, error rates, and popular chapters. Export to CSV or view reports.

- 🛡️ **Validation & Error Handling**
  Automatic checks for Ollama status, model availability, novel format, and graceful error recovery with detailed logging.

- 📊 **Progress tracking + logging**
  Tracks embedding progress and saves a full chat log with timestamps and chapter references.

- 🌐 **Enhanced Web UI (Gradio)**
  Intuitive interface with search mode selection, real-time analytics, and response time metrics.

- 🖥️ **CLI Mode**
  Run in terminal with commands like `stats` for analytics, `clear` for conversation reset.

---

## 📂 Project Structure

```
novel-rag-chatbot/
├── chatbot.py                # Main application
├── config.yaml               # Configuration file
├── novel.txt                 # Your input novel (plain text)
├── utils/                    # Utility modules
│   ├── __init__.py
│   ├── config_loader.py      # Configuration management
│   ├── entity_tracker.py     # Character/entity extraction
│   ├── smart_chunker.py      # Intelligent text chunking
│   ├── query_enhancer.py     # Query rewriting & enhancement
│   ├── hybrid_retriever.py   # BM25 + semantic search
│   ├── summary_cache.py      # Chapter summarization
│   ├── semantic_cache.py     # Query result caching
│   ├── analytics.py          # Usage tracking
│   └── validators.py         # System validation
├── tests/                    # Unit tests
│   ├── test_chunking.py
│   ├── test_entity_tracker.py
│   └── test_config_loader.py
├── chroma_db/                # Vector database (auto-created)
├── chroma_db_backup/         # Auto-created backup
├── chapter_index.json        # Tracks embedded chapters
├── entity_cache.json         # Character/entity data
├── summary_cache.json        # Chapter summaries
├── semantic_cache.json       # Cached query results
├── analytics.json            # Usage analytics
├── chatlog.txt               # Chat history with sources
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── CLAUDE.md                 # Development guide
```

---

## 🚀 Getting Started

### Prerequisites

1. **Python 3.9+** installed
2. **[Ollama](https://ollama.com/)** installed and running

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv project_env

# Activate virtual environment
# Windows:
project_env\Scripts\activate
# Linux/Mac:
source project_env/bin/activate

# Install Python packages
pip install -r requirements.txt
```

### 2. Download LLMs via Ollama

```bash
ollama pull nomic-embed-text
ollama pull mistral:7b
```

### 3. Add Your Novel

Place your novel text file as `novel.txt`. Chapters must follow this format:

```
Chapter 1: The Beginning
Once upon a time...

Chapter 2: The Journey Continues
The hero ventured forth...
```

Each chapter **must follow this naming pattern** for automatic detection.

### 4. Run the Chatbot

```bash
# Web UI mode (default)
python chatbot.py

# CLI mode (terminal only)
python chatbot.py --no-ui

# With custom novel
python chatbot.py --novel mybook.txt

# With different LLM model
python chatbot.py --model llama2

# With specific search mode
python chatbot.py --search-mode precise

# Show analytics report
python chatbot.py --analytics

# Custom config file
python chatbot.py --config my_config.yaml
```

The web UI will launch at `http://localhost:7860`

---

## 📈 Example Output

```
Your question: What did the hero do in the forest?

--- Answer ---
The hero entered the forest in Chapter 3 to retrieve the lost crystal. He encountered
several wild beasts and had to fight them off using his newfound magical abilities.
Later in Chapter 5, he returned to the forest to hide from his pursuers, where he
discovered an ancient shrine.

📘 Based on chapters: 3, 5, 7
⏱️ Response time: 2.34s
📊 Documents retrieved: 5
```

---

## ⚙️ Configuration

All settings are in `config.yaml`. Key sections:

### Models
```yaml
models:
  embedding: "nomic-embed-text"
  llm: "mistral:7b"
  query_rewriter: "mistral:7b"
```

### Chunking
```yaml
chunking:
  size: 1000
  overlap: 200
  respect_sentences: true
  min_chunk_size: 100
```

### Retrieval
```yaml
retrieval:
  top_k: 5
  similarity_threshold: 0.7
  search_type: "hybrid"  # semantic, keyword, or hybrid
  bm25_weight: 0.3
  semantic_weight: 0.7
  use_mmr: true  # Maximal Marginal Relevance for diversity
```

### Search Modes
```yaml
search_modes:
  broad:
    top_k: 10
    similarity_threshold: 0.6
  precise:
    top_k: 3
    similarity_threshold: 0.8
  character_focused:
    top_k: 8
    similarity_threshold: 0.7
  timeline:
    top_k: 7
    similarity_threshold: 0.65
```

### Features Toggle
```yaml
entity_extraction:
  enabled: true  # Character tracking
summarization:
  enabled: true  # Chapter summaries
semantic_cache:
  enabled: true  # Query caching
query_enhancement:
  rewrite_enabled: true  # Query rewriting
  pronoun_resolution: true  # Pronoun → entity name
```

See `config.yaml` for all available options.

---

## 🎯 Usage Tips

### CLI Mode Commands

When running with `--no-ui`:

- Type your question normally
- `exit` - Quit the application
- `clear` - Clear conversation context
- `stats` - Show analytics report

### Search Modes

Choose the right mode for your question:

- **Broad**: "What happens in the story?" "Tell me about the plot"
- **Precise**: "What color was the dragon?" "Who said 'I am your father'?"
- **Character Focused**: "Describe Alice's character development" "What is the relationship between X and Y?"
- **Timeline**: "What happened first?" "Trace the sequence of events"

### Conversation Context

The chatbot remembers recent conversation:

```
Q: Who is Li Fan?
A: Li Fan is the protagonist, a scholar who...

Q: What did he do in Chapter 3?
A: Li Fan (automatically resolved) attempted to capture immortals...
```

### Entity Queries

With entity tracking enabled:

```
Q: Which chapters does Alice appear in?
A: Alice appears in chapters 1, 5, 7, 12, and 15.

Q: Show me all characters
A: Main characters found: Alice, Bob, Charlie, Dave...
```

---

## 📊 Analytics

View comprehensive analytics:

```bash
# Show report and exit
python chatbot.py --analytics

# In CLI mode, type:
stats
```

Analytics include:
- Total queries and average response time
- Most referenced chapters
- Query length distribution
- Performance metrics (embedding, retrieval, LLM)
- Error summary
- Retrieval quality scores
- Hourly usage patterns

Export to CSV:
```python
from utils import Analytics
analytics = Analytics()
analytics.export_to_csv("analytics_report.csv")
```

---

## 🧪 Testing

Run unit tests:

```bash
# All tests
python -m unittest discover tests

# Specific test
python -m unittest tests.test_chunking

# Verbose output
python -m unittest discover tests -v
```

---

## 🔧 Advanced Usage

### Reset Database

To re-embed the entire novel:

```bash
# Windows
rmdir /s /q chroma_db
del chapter_index.json entity_cache.json summary_cache.json

# Linux/Mac
rm -rf chroma_db chapter_index.json entity_cache.json summary_cache.json

# Then run chatbot.py again
python chatbot.py
```

### Clear Caches Only

Keep embeddings, clear caches:

```bash
# Windows
del entity_cache.json summary_cache.json semantic_cache.json analytics.json

# Linux/Mac
rm entity_cache.json summary_cache.json semantic_cache.json analytics.json
```

### Backup and Restore

```bash
# Backup is automatic before updates, but you can manually backup:
cp -r chroma_db my_backup

# Restore from backup:
rm -rf chroma_db && cp -r chroma_db_backup chroma_db
```

### Adding New Chapters

Simply append new chapters to `novel.txt` following the format, then run:

```bash
python chatbot.py
```

Only new chapters will be embedded (automatic incremental update).

---

## ✅ Quality-of-Life Features

- ✔ Timestamped logging to `chatlog.txt` with chapter citations
- ✔ Deduplication to skip already embedded content
- ✔ Auto-backup of DB before updates
- ✔ Chapter-title-based chunk tracking
- ✔ Conversation memory for follow-up questions
- ✔ Entity tracking for character-aware queries
- ✔ Semantic caching for faster repeated queries
- ✔ Comprehensive analytics and monitoring
- ✔ System validation on startup
- ✔ Graceful error handling and recovery
- ✔ Progress bars for long operations
- ✔ CLI arguments for flexibility
- ✔ Modular, extensible architecture

---

## 🔒 Privacy First

This tool **runs fully offline** and does not use any external APIs or internet services:

- All models run locally via Ollama
- No data sent to external servers
- No telemetry or tracking
- All caches stored locally
- Your data stays on your machine

Perfect for sensitive, proprietary, or private content.

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- Additional search strategies
- More entity extraction patterns
- Alternative embedding models
- Performance optimizations
- UI enhancements
- More comprehensive tests

---

## 📝 Troubleshooting

### Ollama not responding

```
Error: Ollama is not responding
```

**Solution**: Ensure Ollama is running:
```bash
ollama serve
```

### Missing models

```
Error: Missing models: nomic-embed-text
```

**Solution**: Download required models:
```bash
ollama pull nomic-embed-text
ollama pull mistral:7b
```

### Import errors

```
ModuleNotFoundError: No module named 'utils'
```

**Solution**: Run from project root and ensure virtual environment is activated:
```bash
cd novel-rag-chatbot
source project_env/bin/activate  # or project_env\Scripts\activate on Windows
python chatbot.py
```

### Memory issues

If embedding large novels causes memory errors:

1. Reduce batch size in `config.yaml`:
   ```yaml
   performance:
     embedding_batch_size: 2  # default is 5
   ```

2. Reduce chunk size:
   ```yaml
   chunking:
     size: 800  # default is 1000
   ```

### Slow query rewriting

If query enhancement adds too much latency:

```yaml
query_enhancement:
  rewrite_enabled: false  # Disable query rewriting
```

---

## 📄 License

MIT License

---

## 🙌 Credits

- Built with [LangChain](https://github.com/langchain-ai/langchain)
- Embeddings by [nomic-embed-text](https://ollama.com/library/nomic-embed-text)
- LLM powered by [mistral:7b](https://ollama.com/library/mistral)
- Vector database: [ChromaDB](https://www.trychroma.com/)
- UI framework: [Gradio](https://gradio.app/)

---

## 📚 Documentation

- **README.md** (this file): User guide and features
- **CLAUDE.md**: Developer guide and architecture details
- **config.yaml**: Configuration reference with comments

---

**Built with ❤️ for book lovers and AI enthusiasts**
