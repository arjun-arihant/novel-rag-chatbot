# Novel RAG Chatbot

A modern, local RAG system for chatting with novels using Ollama.

## Features
- **Accurate Retrieval**: Hybrid search (Dense + BM25) with Reciprocal Rank Fusion.
- **Smart Chunking**: 400-800 token chunks that respect sentence boundaries.
- **Grounded Answers**: Strict refusal logic and citation-backed responses.
- **Modern UI**: Dark-themed web interface with source visualization.
- **Private**: Runs 100% locally with Ollama.

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Pull Models**
   ```bash
   ollama pull qwen3-embedding:0.6b
   ollama pull llama3.1:8b
   ```

3. **Run**
   ```bash
   # Web Interface
   python -m src.main --mode web --novel novel.txt
   
   # Command Line
   python -m src.main --mode cli --novel novel.txt
   ```

## Configuration
Edit `config.yaml` to change models, chunking parameters, or UI settings.
