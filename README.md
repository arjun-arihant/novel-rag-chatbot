# 📚 Local RAG Chatbot for Novels

A **private, offline AI chatbot** that can answer questions about any novel using **Retrieval-Augmented Generation (RAG)**. Built with [LangChain](https://github.com/langchain-ai/langchain) and [Ollama](https://ollama.com/), this tool intelligently indexes, embeds, and chats over large novels — while respecting chapter metadata and working completely offline.

---

## ✨ Features

- 🔍 **Chapter-aware metadata tagging**  
  Automatically extracts and stores chapter titles/numbers for context-aware answers.

- 🧠 **Dual-model architecture**  
  Uses `nomic-embed-text` for fast embedding + `mistral:7b` for accurate answering.

- 💾 **Persistent, update-safe vector database**  
  Avoids re-embedding already indexed content and backs up your database before changes.

- ➕ **Automatic detection of new chapters**  
  Seamlessly updates the chatbot with new content when a novel is updated.

- 📊 **Progress tracking + logging**  
  Tracks embedding progress and saves a full chat log with timestamps and chapter references.

- 🌐 **Optional Web UI (Gradio)**  
  Easily accessible chatbot via a local browser tab.

---

## 📂 Project Structure

```
├── novel.txt              # Your input novel (in plain text format)
├── main.py                # Main script to run the chatbot
├── chroma_db/             # Persistent vector database (auto-created)
├── chroma_db_backup/      # Auto-created backup of the DB
├── chapter_index.json     # Tracks embedded chapters
├── chatlog.txt            # Auto-generated chat history with chapter sources
├── README.md              # You're here :)
```

---

## 🚀 Getting Started

### 1. Install Dependencies

Make sure you have Python 3.9+ and [Ollama](https://ollama.com/) installed and running.

```bash
pip install -r requirements.txt
```

### 2. Download LLMs via Ollama

Run this before launching:

```bash
ollama pull nomic-embed-text
ollama pull mistral:7b
```

### 3. Add Your Novel

Place your novel text file as `novel.txt`. Chapters must follow this format:

```
Chapter 3: A Grand Adventure
Once upon a time...
```

Each chapter **must follow this naming style.**

### 4. Run the Chatbot

```bash
python chatbot.py
```

You'll be prompted to ask questions. Type `exit` to quit.

---

## 📈 Example Output

```
Your question: What did the hero do in the forest?

--- Answer ---
He entered the forest to retrieve the lost crystal and fought off several wild beasts.

📘 Based on chapters: 3, 5, 7
```

---

## ✅ Quality-of-Life Features

- ✔ Timestamped logging to `chatlog.txt`
- ✔ Deduplication to skip already embedded content
- ✔ Auto-backup of DB before updates
- ✔ Chapter-title-based chunk tracking
- ✔ Easy-to-extend for more documents

---

## 🔒 Privacy First

This tool **runs fully offline** and does not use any external APIs or internet services. Your data stays on your machine.

---

## 📄 License

MIT License

---

## 🙌 Credits

- Built with [LangChain](https://github.com/langchain-ai/langchain)
- Embeddings by [nomic-embed-text](https://ollama.com/library/nomic-embed-text)
- LLM powered by [mistral:7b](https://ollama.com/library/mistral)
