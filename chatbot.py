import os
import json
import shutil
import re
from datetime import datetime
from tqdm import tqdm

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# --- Configuration ---
NOVEL_FILE_PATH = "novel.txt"
PERSIST_DIRECTORY = "chroma_db"
CHAPTER_INDEX_FILE = "chapter_index.json"
DB_BACKUP_DIR = "chroma_db_backup"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral:7b"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Helper Functions ---
def load_chapter_index():
    if os.path.exists(CHAPTER_INDEX_FILE):
        with open(CHAPTER_INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_chapter_index(index):
    with open(CHAPTER_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

def backup_database():
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.copytree(PERSIST_DIRECTORY, DB_BACKUP_DIR, dirs_exist_ok=True)
        print(f"🔁 Backup created at: {DB_BACKUP_DIR}")

def extract_chapters(text):
    chapter_regex = re.compile(r"Chapter\s+(\d+):\s+(.*?)\n")
    matches = list(chapter_regex.finditer(text))
    chapters = []
    for i in range(len(matches)):
        start = matches[i].end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        number = matches[i].group(1)
        title = matches[i].group(2)
        content = text[start:end].strip()
        chapters.append({
            "chapter_number": number,
            "chapter_title": f"Chapter {number}: {title}",
            "content": content
        })
    return chapters

# --- Load Novel ---
print(f"📖 Loading novel from {NOVEL_FILE_PATH}...")
try:
    loader = TextLoader(NOVEL_FILE_PATH, encoding="utf-8")
    documents = loader.load()
    full_text = documents[0].page_content
    print(f"✅ Loaded. Total characters: {len(full_text)}")
except Exception as e:
    print(f"❌ Failed to load novel: {e}")
    exit()

# --- Parse Chapters ---
chapter_index = load_chapter_index()
chapters = extract_chapters(full_text)
print(f"🔍 Found {len(chapters)} chapters in novel.")

# --- Prepare Chunks to Embed ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks_to_add = []

for chapter in chapters:
    title = chapter["chapter_title"]
    number = chapter["chapter_number"]
    if title in chapter_index:
        print(f"⏩ Skipping already embedded: {title}")
        continue
    chapter_docs = text_splitter.create_documents([chapter["content"]])
    for doc in chapter_docs:
        doc.metadata = {"chapter_title": title, "chapter_number": number}
        chunks_to_add.append(doc)
    chapter_index[title] = "embedded"

print(f"📦 {len(chunks_to_add)} new chunks prepared from new chapters.")

# --- Embedding & Vector DB ---
if not os.path.exists(PERSIST_DIRECTORY):
    os.makedirs(PERSIST_DIRECTORY)

backup_database()

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

if chunks_to_add:
    print("🧠 Embedding and adding chunks (batch mode)...")
    for i in tqdm(range(0, len(chunks_to_add), 5), desc="🔄 Embedding", unit="batch"):
        batch = chunks_to_add[i:i+5]
        vectordb.add_documents(batch)
    vectordb.persist()
    print("✅ Vector DB updated and saved.")
    save_chapter_index(chapter_index)
else:
    print("👍 No new chapters to add.")

# --- Set up LLM and RAG ---
llm = Ollama(model=LLM_MODEL)
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

prompt_template = """Use the following pieces of context to answer the user's question.
If you don't know the answer, say so.
Do not make up anything.

Context:
{context}

Question: {question}
Answer:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# --- Interactive Chat Loop ---
print("\n🤖 Chatbot is ready! Ask questions. Type 'exit' to quit.")
import gradio as gr

def ask_question(query):
    if not query.strip():
        return "Please enter a question."

    try:
        result = qa_chain.invoke({"query": query})
        answer = result["result"]

        # Extract chapter numbers
        chapters_used = set()
        for doc in result["source_documents"]:
            chapter_number = doc.metadata.get("chapter_number")
            if chapter_number:
                chapters_used.add(int(chapter_number))

        chapter_list = ", ".join(map(str, sorted(chapters_used)))
        display = f"{answer}\n\n📘 Based on chapters: {chapter_list}" if chapter_list else answer

        # Log
        with open("chatlog.txt", "a", encoding="utf-8") as log:
            log.write(f"\n[{datetime.now().isoformat()}]\nQ: {query}\nA: {answer}\nChapters: {chapter_list if chapters_used else 'N/A'}\n")

        return display
    except Exception as e:
        return f"❌ Error: {e}"

gr.Interface(
    fn=ask_question,
    inputs=gr.Textbox(lines=2, placeholder="Ask a question about the novel..."),
    outputs=gr.Textbox(lines=10),
    title="📚 RAG Novel Chatbot",
    description="Ask questions about the novel. Powered by local LLM and embeddings.",
    theme="soft"
).launch()
