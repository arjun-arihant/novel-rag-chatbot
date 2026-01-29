# AGENTS.md - AI Agent Context

> This file helps AI coding assistants understand and work with the Novel RAG Chatbot codebase.

## 🎯 Project Overview

**Novel RAG Chatbot** is a Retrieval-Augmented Generation system for chatting with novels. It runs locally using Ollama and supports multiple books with incremental indexing.

### Core Concepts

| Concept | Description |
|---------|-------------|
| **RAG Pipeline** | Query → Rewrite → Retrieve → Rerank → Generate |
| **Multi-Novel** | Each novel has its own vector DB and BM25 index |
| **Incremental Indexing** | Hash-based change detection; only new chapters indexed |
| **Hybrid Retrieval** | Semantic (ChromaDB) + Keyword (BM25) with RRF fusion |
| **Refusal Logic** | Three-layer check before answering |

---

## 📁 Key Files

### Entry Points

| File | Purpose |
|------|---------|
| `src/main.py` | CLI entry point - `web` or `cli` mode |
| `src/ui/app.py` | FastAPI application with all endpoints |
| `src/pipeline.py` | RAG orchestrator - ties everything together |

### Configuration

| File | Purpose |
|------|---------|
| `config.yaml` | All tunable parameters |
| `src/config.py` | Config dataclasses |

### Library & Indexing

| File | Purpose |
|------|---------|
| `src/library.py` | Novel metadata, multi-book management |
| `src/ingestion/indexer.py` | Incremental indexing with hash detection |
| `src/ingestion/parsers/` | TXT, PDF, EPUB parsers |
| `src/ingestion/chunker.py` | Token-based chunking |

### Retrieval

| File | Purpose |
|------|---------|
| `src/retrieval/embedder.py` | Ollama embedding wrapper |
| `src/retrieval/vector_store.py` | ChromaDB operations |
| `src/retrieval/sparse_index.py` | BM25 keyword index |
| `src/retrieval/hybrid.py` | RRF score fusion |
| `src/retrieval/reranker.py` | LLM-based reranking |

### Generation

| File | Purpose |
|------|---------|
| `src/generation/query_rewriter.py` | Query enhancement (temp=0) |
| `src/generation/generator.py` | Answer generation with refusal logic |
| `src/generation/prompts.py` | All prompt templates |

---

## 🔧 Common Tasks

### Add a New File Parser

1. Create `src/ingestion/parsers/xyz_parser.py`
2. Implement `BaseParser` interface
3. Register in `parsers/__init__.py`

```python
from .base import BaseParser, Chapter

class XyzParser(BaseParser):
    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".xyz"
    
    def parse(self, file_path: Path) -> list[Chapter]:
        # Extract chapters with number, title, content
        return [Chapter(number=1, title="...", content="...")]
```

### Modify Chunking Strategy

Edit `src/ingestion/chunker.py`:
- `target_tokens`, `min_tokens`, `max_tokens` in config
- Sentence alignment logic in `_split_into_sentences`

### Adjust Refusal Sensitivity

Edit `src/generation/generator.py`:
- `min_rerank_score` threshold in config
- Entity coverage logic in `_check_entity_coverage`

### Add API Endpoint

Edit `src/ui/app.py`:
```python
@app.get("/api/your-endpoint")
async def your_endpoint():
    return {"data": "..."}
```

---

## 🚨 Important Constraints

### Never Do

- **Use LangChain** — Direct Ollama HTTP calls only
- **Hardcode novel paths** — Use library system
- **Skip refusal checks** — Hallucination prevention is critical
- **Modify prompts casually** — Temperature and token limits are specific

### Always Do

- **Test with multiple file formats** — TXT, PDF, EPUB
- **Verify incremental indexing** — Re-upload should only index new chapters
- **Check refusal behavior** — Out-of-scope queries must be refused
- **Preserve chapter citations** — Every answer needs sources

---

## 🧪 Testing

```bash
# Run the app
python -m src.main --mode web

# Test endpoints
curl http://localhost:8000/api/health
curl http://localhost:8000/api/novels

# Upload a novel
curl -X POST http://localhost:8000/api/novels \
  -F "file=@novel.txt" \
  -F "title=My Novel"
```

---

## 📊 Data Flow

```
User Query
    ↓
QueryRewriter (temp=0, max=128 tokens)
    ↓
HybridRetriever
    ├── VectorStore.search() → top_k dense results
    └── BM25Index.search() → top_k sparse results
    ↓
RRF Fusion → combined ranked list
    ↓
LLMReranker (temp=0, JSON output)
    ↓
GroundedGenerator
    ├── Check: No results? → Refuse
    ├── Check: Low scores? → Refuse
    ├── Check: Missing entities? → Refuse
    └── Generate grounded answer
    ↓
Answer + Citations
```

---

## 🔑 Environment

- **Python**: 3.11+
- **Ollama Models**: `llama3.1:8b`, `qwen3-embedding:0.6b`
- **Key Deps**: FastAPI, ChromaDB, rank-bm25, httpx

---

> 💡 **Tip**: When making changes, always consider the incremental indexing system. Novel metadata and chapter hashes are stored in `library/`.
