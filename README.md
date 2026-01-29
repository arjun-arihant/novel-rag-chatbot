# 📚 Novel RAG Chatbot

A local-first, privacy-focused chatbot that lets you have intelligent conversations with your novels. Upload your books, ask questions, and get grounded answers with source citations.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Ollama](https://img.shields.io/badge/Ollama-Powered-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## ✨ Features

- **Multi-Novel Library** — Manage multiple books with separate vector databases
- **Multi-Format Support** — Upload `.txt`, `.pdf`, and `.epub` files
- **Incremental Indexing** — Re-upload with new chapters; only new content is processed
- **Grounded Answers** — Every response cites specific chapters from your novel
- **Hybrid Search** — Combines semantic + keyword search for accurate retrieval
- **LLM Reranking** — Intelligent reranking with refusal logic for hallucination prevention
- **Beautiful UI** — Dark theme with drag-drop upload, processing progress, and library management
- **100% Local** — All processing happens on your machine with Ollama

## 🚀 Quick Start

### Prerequisites

1. **Python 3.11+**
2. **Ollama** — [Install Ollama](https://ollama.com/download)
3. **Required models:**
   ```bash
   ollama pull llama3.1:8b
   ollama pull qwen3-embedding:0.6b
   ```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/novel-rag-chatbot.git
cd novel-rag-chatbot

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

### Run

```bash
# Start the web interface
python -m src.main --mode web

# Open http://localhost:8000 in your browser
```

## 📖 Usage

### Web Interface

1. **Open the Library** — Click the Library button in the header
2. **Upload a Book** — Drag & drop or click to browse (supports .txt, .pdf, .epub)
3. **Wait for Processing** — Watch the progress as chapters are parsed and indexed
4. **Select Your Book** — Click on a book card to make it active
5. **Start Chatting** — Ask questions about your novel!

### CLI Mode

```bash
python -m src.main --mode cli

# Commands:
#   add <path> [title] [author] - Add a novel
#   list                        - List all novels
#   select <id>                 - Select a novel for querying
#   <your question>             - Ask a question
#   quit                        - Exit
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
embedding:
  model: "qwen3-embedding:0.6b"
  base_url: "http://localhost:11434"

llm:
  query_rewriter:
    model: "llama3.1:8b"
    temperature: 0.0
    max_tokens: 128
  
  reranker:
    model: "llama3.1:8b"
    temperature: 0.0
  
  generator:
    model: "llama3.1:8b"
    temperature: 0.25
    max_tokens: 512

chunking:
  target_tokens: 600
  min_tokens: 400
  max_tokens: 800
  overlap_tokens: 100

retrieval:
  top_k: 20
  rerank_top_k: 5
  min_rerank_score: 3.0
```

## 🏗️ Architecture

```
src/
├── main.py              # Entry point (web/cli modes)
├── pipeline.py          # RAG orchestrator
├── library.py           # Multi-novel library manager
├── config.py            # Configuration system
├── ollama_client.py     # Ollama API client
├── ingestion/
│   ├── parsers/         # File format parsers (txt, pdf, epub)
│   ├── indexer.py       # Incremental indexing
│   ├── chunker.py       # Token-based chunking
│   └── metadata.py      # Entity extraction
├── retrieval/
│   ├── embedder.py      # Embedding generation
│   ├── vector_store.py  # ChromaDB wrapper
│   ├── sparse_index.py  # BM25 index
│   ├── hybrid.py        # Reciprocal Rank Fusion
│   └── reranker.py      # LLM reranking
├── generation/
│   ├── prompts.py       # Prompt templates
│   ├── query_rewriter.py # Query enhancement
│   └── generator.py     # Answer generation with refusal logic
└── ui/
    ├── app.py           # FastAPI endpoints
    ├── templates/       # HTML templates
    └── static/          # CSS/JS assets
```

## 📡 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/stats` | GET | Pipeline statistics |
| `/api/novels` | GET | List all novels |
| `/api/novels` | POST | Upload a new novel |
| `/api/novels/{id}` | GET | Get novel details |
| `/api/novels/{id}` | DELETE | Delete a novel |
| `/api/novels/{id}/select` | POST | Select novel for querying |
| `/api/novels/{id}/reindex` | POST | Re-index a novel |
| `/api/novels/active` | GET | Get active novel |
| `/api/query` | POST | Query the active novel |

## 🔧 Troubleshooting

### "Connection refused" error
Ensure Ollama is running:
```bash
ollama serve
```

### Slow embedding generation
- Use a smaller embedding model: `nomic-embed-text`
- Reduce chunk size in config

### PDF/EPUB parsing issues
- Ensure PyMuPDF and ebooklib are installed
- Try converting to `.txt` for problematic files

## 📝 License

MIT License - feel free to use, modify, and distribute.

---

Built with ❤️ for book lovers who want to keep their reading private.
