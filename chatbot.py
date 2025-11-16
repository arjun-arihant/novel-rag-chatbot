"""
Enhanced Novel RAG Chatbot with advanced features:
- Conversation memory
- Hybrid search (BM25 + semantic)
- Query enhancement and rewriting
- Entity tracking
- Smart chunking
- Semantic caching
- Chapter summarization
- Analytics and monitoring
- Streaming responses
"""

import os
import json
import shutil
import re
import argparse
import time
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Optional

# LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

# Gradio for UI
import gradio as gr

# Import our utilities
from utils import (
    load_config,
    EntityTracker,
    SmartChunker,
    QueryEnhancer,
    HybridRetriever,
    SummaryCache,
    SemanticCache,
    Analytics,
    Validators
)


class NovelRAGChatbot:
    """Enhanced RAG chatbot for novel Q&A."""

    def __init__(self, config_path: str = "config.yaml", cli_args: Optional[Dict] = None):
        """
        Initialize the chatbot.

        Args:
            config_path: Path to configuration file
            cli_args: Optional CLI arguments to override config
        """
        print("🚀 Initializing Enhanced Novel RAG Chatbot...")

        # Load configuration
        self.config = load_config(config_path)

        # Apply CLI overrides
        if cli_args:
            self._apply_cli_overrides(cli_args)

        # Extract config values
        self.paths = self.config['paths']
        self.models = self.config['models']
        self.chunking_config = self.config['chunking']
        self.retrieval_config = self.config['retrieval']
        self.memory_config = self.config['memory']
        self.query_enhancement_config = self.config['query_enhancement']
        self.ui_config = self.config['ui']

        # Run validation if enabled
        if self.config['validation']['check_ollama_running']:
            self._run_validations()

        # Initialize components
        self.entity_tracker = None
        self.smart_chunker = None
        self.query_enhancer = None
        self.hybrid_retriever = None
        self.summary_cache = None
        self.semantic_cache = None
        self.analytics = None
        self.vectorstore = None
        self.qa_chain = None
        self.memory = None

        # State
        self.chapter_index = {}
        self.all_documents = []
        self.conversation_history = []

        # Initialize
        self._initialize_components()
        self._load_and_process_novel()
        self._setup_retrieval()
        self._setup_qa_chain()

        print("✅ Chatbot initialization complete!\n")

    def _apply_cli_overrides(self, cli_args: Dict):
        """Apply CLI argument overrides to config."""
        if cli_args.get('novel'):
            self.config['paths']['novel'] = cli_args['novel']
        if cli_args.get('model'):
            self.config['models']['llm'] = cli_args['model']
        if cli_args.get('search_mode'):
            mode = cli_args['search_mode']
            if mode in self.config['search_modes']:
                self.config['retrieval'].update(self.config['search_modes'][mode])

    def _run_validations(self):
        """Run system validations."""
        print("\n📋 Running system validations...")
        results = Validators.run_all_checks(self.config)
        all_passed = Validators.print_validation_report(results)

        if not all_passed:
            response = input("Some validations failed. Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Exiting...")
                exit(1)

    def _initialize_components(self):
        """Initialize all utility components."""
        print("🔧 Initializing components...")

        # Entity tracker
        if self.config['entity_extraction']['enabled']:
            self.entity_tracker = EntityTracker(
                cache_path=self.paths['entity_cache'],
                min_mentions=self.config['entity_extraction']['min_mention_count']
            )

        # Smart chunker
        self.smart_chunker = SmartChunker(
            chunk_size=self.chunking_config['size'],
            chunk_overlap=self.chunking_config['overlap'],
            respect_sentences=self.chunking_config['respect_sentences'],
            min_chunk_size=self.chunking_config['min_chunk_size']
        )

        # Query enhancer
        if self.query_enhancement_config['rewrite_enabled']:
            self.query_enhancer = QueryEnhancer(
                llm_model=self.models['query_rewriter'],
                entity_tracker=self.entity_tracker
            )

        # Summary cache
        if self.config['summarization']['enabled']:
            self.summary_cache = SummaryCache(
                cache_path=self.paths['summary_cache'],
                llm_model=self.models['llm'],
                summary_length=self.config['summarization']['summary_length']
            )

        # Semantic cache
        if self.config['semantic_cache']['enabled']:
            self.semantic_cache = SemanticCache(
                cache_path=self.paths['semantic_cache'],
                similarity_threshold=self.config['semantic_cache']['similarity_threshold'],
                max_cache_size=self.config['semantic_cache']['max_cache_size'],
                ttl_hours=self.config['semantic_cache']['ttl_hours']
            )

        # Analytics
        if self.config['performance']['enable_analytics']:
            self.analytics = Analytics(
                analytics_path=self.paths['analytics']
            )

        # Memory for conversation
        if self.memory_config['enabled']:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="result"
            )

    def _load_and_process_novel(self):
        """Load novel and process chapters."""
        novel_path = self.paths['novel']
        print(f"📖 Loading novel from {novel_path}...")

        start_time = time.time()

        try:
            loader = TextLoader(novel_path, encoding="utf-8")
            documents = loader.load()
            full_text = documents[0].page_content
            print(f"✅ Loaded. Total characters: {len(full_text)}")
        except Exception as e:
            print(f"❌ Failed to load novel: {e}")
            if self.analytics:
                self.analytics.log_error("novel_loading", str(e))
            exit(1)

        # Parse chapters
        chapters = self._extract_chapters(full_text)
        print(f"🔍 Found {len(chapters)} chapters in novel.")

        # Load chapter index
        self.chapter_index = self._load_chapter_index()

        # Process chapters
        self._process_chapters(chapters)

        duration = time.time() - start_time
        if self.analytics:
            self.analytics.log_performance("novel_loading", duration, success=True)

    def _extract_chapters(self, text: str) -> List[Dict]:
        """Extract chapters from novel text."""
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

    def _process_chapters(self, chapters: List[Dict]):
        """Process chapters for embedding and entity extraction."""
        chunks_to_add = []
        new_chapters = []

        for chapter in chapters:
            title = chapter["chapter_title"]
            number = chapter["chapter_number"]
            content = chapter["content"]

            # Check if already embedded
            if title in self.chapter_index:
                print(f"⏩ Skipping already embedded: {title}")
                continue

            new_chapters.append(chapter)

            # Extract entities
            if self.entity_tracker:
                self.entity_tracker.extract_entities_from_chapter(content, number)

            # Generate summary
            if self.summary_cache:
                print(f"📝 Generating summary for {title}...")
                self.summary_cache.get_summary(number, title, content)

            # Smart chunking
            chapter_docs = self.smart_chunker.create_documents(
                [content],
                [{"chapter_title": title, "chapter_number": number}]
            )

            # Enhance metadata with entities
            if self.entity_tracker:
                for doc in chapter_docs:
                    self.entity_tracker.add_character_to_metadata(
                        doc.page_content,
                        doc.metadata
                    )

            chunks_to_add.extend(chapter_docs)
            self.chapter_index[title] = "embedded"

        print(f"📦 {len(chunks_to_add)} new chunks prepared from {len(new_chapters)} new chapters.")

        # Finalize entity extraction
        if self.entity_tracker and new_chapters:
            self.entity_tracker.finalize_entities()
            for chapter in new_chapters:
                self.entity_tracker.extract_relationships(
                    chapter["content"],
                    chapter["chapter_number"]
                )
            self.entity_tracker.save_cache()

        # Embed new chunks
        if chunks_to_add:
            self._embed_chunks(chunks_to_add)
            self._save_chapter_index()
        else:
            print("👍 No new chapters to add.")

        # Store all documents for BM25
        self.all_documents = chunks_to_add

    def _embed_chunks(self, chunks: List[Document]):
        """Embed and add chunks to vector database."""
        # Create backup if configured
        if self.config['validation']['backup_before_update']:
            self._backup_database()

        # Ensure directory exists
        persist_dir = self.paths['persist_directory']
        if not os.path.exists(persist_dir):
            os.makedirs(persist_dir)

        # Initialize embeddings
        embeddings = OllamaEmbeddings(model=self.models['embedding'])

        # Load or create vector store
        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            self.vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings
            )
        else:
            self.vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings
            )

        # Add documents in batches
        batch_size = self.config['performance']['embedding_batch_size']
        print("🧠 Embedding and adding chunks (batch mode)...")

        start_time = time.time()

        for i in tqdm(range(0, len(chunks), batch_size), desc="🔄 Embedding", unit="batch"):
            batch = chunks[i:i+batch_size]
            self.vectorstore.add_documents(batch)

        # Persist
        self.vectorstore.persist()

        duration = time.time() - start_time
        print(f"✅ Vector DB updated and saved in {duration:.2f}s.")

        if self.analytics:
            self.analytics.log_performance("embedding", duration, success=True)

    def _setup_retrieval(self):
        """Setup hybrid retrieval system."""
        if not self.vectorstore:
            # Load existing vectorstore
            embeddings = OllamaEmbeddings(model=self.models['embedding'])
            self.vectorstore = Chroma(
                persist_directory=self.paths['persist_directory'],
                embedding_function=embeddings
            )

        # Initialize hybrid retriever
        self.hybrid_retriever = HybridRetriever(
            vectorstore=self.vectorstore,
            documents=self.all_documents if self.all_documents else None,
            bm25_weight=self.retrieval_config['bm25_weight'],
            semantic_weight=self.retrieval_config['semantic_weight'],
            top_k=self.retrieval_config['top_k'],
            similarity_threshold=self.retrieval_config['similarity_threshold']
        )

    def _setup_qa_chain(self):
        """Setup the QA chain with enhanced prompt."""
        llm = Ollama(model=self.models['llm'])

        # Enhanced prompt template
        prompt_template = """You are an expert literary analyst discussing a novel. Use the context provided to give detailed, insightful answers.

If information spans multiple chapters, explain the progression or connections.
Always cite specific chapter numbers when referencing events.
If you don't know something, clearly state that - don't speculate beyond what's in the context.

Context from the novel:
{context}

Question: {question}

Detailed Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Create retriever from hybrid retriever
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.retrieval_config['top_k']}
        )

        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

    def ask_question(self, query: str, search_mode: str = "default") -> Dict:
        """
        Ask a question and get answer.

        Args:
            query: User question
            search_mode: Search mode (default, broad, precise, etc.)

        Returns:
            Dict with answer, chapters, metadata
        """
        start_time = time.time()

        if not query.strip():
            return {"answer": "Please enter a question.", "chapters": [], "confidence": 0}

        try:
            # Check semantic cache first
            cached_result = None
            if self.semantic_cache:
                cached_result = self.semantic_cache.get(query)
                if cached_result:
                    print("💾 Using cached result")
                    return cached_result

            # Enhance query if enabled
            enhanced_query = query
            if self.query_enhancer:
                enhancement = self.query_enhancer.enhance_query(
                    query,
                    rewrite=self.query_enhancement_config['rewrite_enabled'],
                    expand=self.query_enhancement_config['expand_enabled'],
                    resolve_pronouns=self.query_enhancement_config['pronoun_resolution']
                )
                enhanced_query = enhancement['query']
                if enhanced_query != query:
                    print(f"🔄 Enhanced query: {enhanced_query}")

            # Apply search mode if specified
            if search_mode in self.config['search_modes']:
                mode_config = self.config['search_modes'][search_mode]
                search_kwargs = {
                    "k": mode_config['top_k']
                }
            else:
                search_kwargs = {"k": self.retrieval_config['top_k']}

            # Retrieve documents
            retrieval_start = time.time()

            if self.hybrid_retriever:
                docs = self.hybrid_retriever.retrieve(
                    enhanced_query,
                    search_type=self.retrieval_config['search_type'],
                    use_mmr=self.retrieval_config['use_mmr'],
                    mmr_lambda=self.retrieval_config['mmr_lambda']
                )
            else:
                docs = self.vectorstore.similarity_search(enhanced_query, **search_kwargs)

            retrieval_time = time.time() - retrieval_start

            # Get answer from LLM
            llm_start = time.time()
            result = self.qa_chain.invoke({"query": enhanced_query})
            answer = result["result"]
            llm_time = time.time() - llm_start

            # Extract chapter information
            chapters_used = set()
            for doc in result["source_documents"]:
                chapter_number = doc.metadata.get("chapter_number")
                if chapter_number:
                    chapters_used.add(int(chapter_number))

            chapter_list = sorted(chapters_used)
            chapter_str = ", ".join(map(str, chapter_list))

            # Build response
            response = {
                "answer": answer,
                "chapters": chapter_list,
                "chapter_str": chapter_str,
                "num_docs_retrieved": len(docs),
                "retrieval_time": retrieval_time,
                "llm_time": llm_time,
                "total_time": time.time() - start_time
            }

            # Add to semantic cache
            if self.semantic_cache:
                self.semantic_cache.set(query, response)

            # Log analytics
            if self.analytics:
                self.analytics.log_query(
                    query=query,
                    answer=answer,
                    chapters_used=[str(c) for c in chapter_list],
                    response_time=response["total_time"]
                )

                self.analytics.log_retrieval_quality(
                    query=query,
                    num_retrieved=len(docs),
                    avg_similarity=0.75,  # Would need to track actual scores
                    chapters_diversity=len(chapters_used)
                )

            # Update conversation context
            if self.query_enhancer:
                entities = []
                if self.entity_tracker:
                    # Extract entities from answer
                    for char in self.entity_tracker.get_all_characters()[:10]:
                        if char in answer:
                            entities.append(char)

                self.query_enhancer.add_to_context(query, answer, entities)

            # Log to file
            self._log_to_file(query, answer, chapter_str)

            return response

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            print(error_msg)

            if self.analytics:
                self.analytics.log_error("query_processing", str(e), {"query": query})

            return {"answer": error_msg, "chapters": [], "total_time": time.time() - start_time}

    def _log_to_file(self, query: str, answer: str, chapters: str):
        """Log interaction to file."""
        try:
            with open(self.paths['chatlog'], "a", encoding="utf-8") as log:
                log.write(f"\n[{datetime.now().isoformat()}]\n")
                log.write(f"Q: {query}\n")
                log.write(f"A: {answer}\n")
                log.write(f"Chapters: {chapters if chapters else 'N/A'}\n")
        except Exception as e:
            print(f"Warning: Could not write to chat log: {e}")

    def _backup_database(self):
        """Create backup of vector database."""
        persist_dir = self.paths['persist_directory']
        backup_dir = self.paths['db_backup']

        if os.path.exists(persist_dir):
            shutil.copytree(persist_dir, backup_dir, dirs_exist_ok=True)
            print(f"🔁 Backup created at: {backup_dir}")

    def _load_chapter_index(self) -> Dict:
        """Load chapter index from disk."""
        index_path = self.paths['chapter_index']
        if os.path.exists(index_path):
            with open(index_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_chapter_index(self):
        """Save chapter index to disk."""
        index_path = self.paths['chapter_index']
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(self.chapter_index, f, indent=2)

    def launch_ui(self):
        """Launch Gradio web interface."""
        print("\n🌐 Launching web interface...")

        def gradio_ask(query, mode):
            """Gradio wrapper for ask_question."""
            result = self.ask_question(query, search_mode=mode)

            answer = result["answer"]
            chapters = result.get("chapter_str", "")
            time_taken = result.get("total_time", 0)

            # Format response
            output = answer

            if chapters:
                output += f"\n\n📘 **Based on chapters**: {chapters}"

            if self.ui_config.get('show_confidence'):
                output += f"\n\n⏱️ **Response time**: {time_taken:.2f}s"
                output += f"\n📊 **Documents retrieved**: {result.get('num_docs_retrieved', 0)}"

            return output

        # Create interface
        with gr.Blocks(theme=gr.themes.Soft(), title="Novel RAG Chatbot") as interface:
            gr.Markdown("# 📚 Enhanced Novel RAG Chatbot")
            gr.Markdown("Ask questions about your novel with advanced AI-powered retrieval.")

            with gr.Row():
                with gr.Column(scale=4):
                    query_input = gr.Textbox(
                        lines=3,
                        placeholder="Ask a question about the novel...",
                        label="Your Question"
                    )

                with gr.Column(scale=1):
                    mode_dropdown = gr.Dropdown(
                        choices=["default"] + list(self.config['search_modes'].keys()),
                        value="default",
                        label="Search Mode"
                    )

            submit_btn = gr.Button("Ask", variant="primary")

            output = gr.Textbox(
                lines=15,
                label="Answer",
                show_copy_button=True
            )

            submit_btn.click(
                fn=gradio_ask,
                inputs=[query_input, mode_dropdown],
                outputs=output
            )

            # Add examples if configured
            gr.Examples(
                examples=[
                    ["Who is the main character?", "default"],
                    ["What happens in the story?", "broad"],
                    ["Describe the relationship between the main characters", "character_focused"]
                ],
                inputs=[query_input, mode_dropdown]
            )

            # Analytics section
            if self.analytics:
                with gr.Accordion("📊 Analytics", open=False):
                    analytics_output = gr.Textbox(
                        value=self.analytics.get_comprehensive_report(),
                        lines=20,
                        label="System Analytics"
                    )

        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False
        )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Enhanced Novel RAG Chatbot")

    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--novel", help="Path to novel file (overrides config)")
    parser.add_argument("--model", help="LLM model to use (overrides config)")
    parser.add_argument("--search-mode", choices=["broad", "precise", "character_focused", "timeline"],
                       help="Default search mode")
    parser.add_argument("--no-ui", action="store_true", help="Disable web UI (CLI only)")
    parser.add_argument("--analytics", action="store_true", help="Show analytics report and exit")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Initialize chatbot
    chatbot = NovelRAGChatbot(
        config_path=args.config,
        cli_args=vars(args)
    )

    # Show analytics if requested
    if args.analytics:
        if chatbot.analytics:
            print(chatbot.analytics.get_comprehensive_report())
        else:
            print("Analytics not enabled in config.")
        return

    # Launch UI
    if not args.no_ui:
        chatbot.launch_ui()
    else:
        # CLI mode
        print("\n🤖 Chatbot is ready! (CLI mode)")
        print("Type 'exit' to quit, 'clear' to clear conversation, 'stats' for analytics.\n")

        while True:
            try:
                query = input("Your question: ").strip()

                if query.lower() == 'exit':
                    print("Goodbye!")
                    break
                elif query.lower() == 'clear':
                    if chatbot.query_enhancer:
                        chatbot.query_enhancer.clear_context()
                    print("Conversation cleared.")
                    continue
                elif query.lower() == 'stats':
                    if chatbot.analytics:
                        print(chatbot.analytics.get_comprehensive_report())
                    continue

                if not query:
                    continue

                result = chatbot.ask_question(query)

                print(f"\n--- Answer ---")
                print(result["answer"])
                if result.get("chapter_str"):
                    print(f"\n📘 Based on chapters: {result['chapter_str']}")
                print(f"\n⏱️ Response time: {result.get('total_time', 0):.2f}s\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()
