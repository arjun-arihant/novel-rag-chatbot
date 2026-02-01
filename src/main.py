# Novel RAG Chatbot - Main Entry Point

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('novel_rag.log')
    ]
)

logger = logging.getLogger(__name__)


def run_web(host: str = "127.0.0.1", port: int = 8000):
    """Run the web server."""
    import uvicorn
    from src.ui.app import app
    
    logger.info(f"Starting web server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


def run_cli():
    """Run interactive CLI mode."""
    from src.config import get_config
    
    config = get_config()
    mode = getattr(config, 'pipeline_mode', 'simple')
    
    # Import appropriate pipeline
    if mode == "simple":
        from src.simple_pipeline import SimpleRAGPipeline as RAGPipeline
        print("\n  Mode: SIMPLE (fast, dense retrieval only)")
    else:
        from src.pipeline import RAGPipeline
        print("\n  Mode: ADVANCED (hybrid + reranking)")
    
    print("\n" + "=" * 60)
    print("  Novel RAG Chatbot - CLI Mode")
    print("=" * 60 + "\n")
    
    pipeline = RAGPipeline()
    
    # Check for novels
    novels = pipeline.list_novels()
    if not novels:
        print("No novels in library. Use 'add <path>' to add a novel.\n")
        print("Commands:")
        print("  add <path> [title] [author] - Add a novel")
        print("  list                        - List all novels")
        print("  select <id>                 - Select a novel")
        print("  <question>                  - Ask a question")
        print("  quit                        - Exit")
        print()
    else:
        print(f"Found {len(novels)} novel(s) in library.\n")
        for n in novels:
            status = "★" if n.get('status') == 'ready' else "○"
            print(f"  {status} [{n['id']}] {n['title']} ({n['total_chapters']} chapters)")
        print()
    
    while True:
        try:
            active = pipeline.get_active_novel()
            prompt = f"[{active['title'][:20]}] > " if active else "> "
            user_input = input(prompt).strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if user_input.lower() == 'list':
                novels = pipeline.list_novels()
                if not novels:
                    print("No novels in library.")
                else:
                    for n in novels:
                        status = "★" if n.get('status') == 'ready' else "○"
                        print(f"  {status} [{n['id']}] {n['title']}")
                continue
            
            if user_input.lower().startswith('add '):
                parts = user_input[4:].strip().split(' ', 2)
                file_path = Path(parts[0])
                title = parts[1] if len(parts) > 1 else None
                author = parts[2] if len(parts) > 2 else "Unknown"
                
                if not file_path.exists():
                    print(f"File not found: {file_path}")
                    continue
                
                print(f"Adding {file_path.name}...")
                result = pipeline.add_novel(file_path, title, author)
                if result["indexing"]["status"] == "error":
                    print(f"Error: {result['indexing']['error']}")
                else:
                    print(f"Added: {result['novel']['title']} ({result['indexing']['total_chapters']} chapters)")
                    # Auto-select if first novel
                    if not active:
                        pipeline.select_novel(result['novel']['id'])
                        print(f"Selected: {result['novel']['title']}")
                continue
            
            if user_input.lower().startswith('select '):
                novel_id = user_input[7:].strip()
                if pipeline.select_novel(novel_id):
                    novel = pipeline.get_active_novel()
                    print(f"Selected: {novel['title']}")
                else:
                    print("Failed to select novel. Make sure it's indexed.")
                continue
            
            # Query
            if not pipeline.is_ready():
                print("Please select a novel first. Use 'select <id>'")
                continue
            
            print()
            result = pipeline.query(user_input)
            
            if result.refused:
                print(f"[Refused: {result.refusal_reason}]")
            print(result.answer)
            
            if result.chapters_cited:
                print(f"\n  Sources: Chapters {', '.join(map(str, result.chapters_cited))}")
            
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Novel RAG Chatbot - Chat with your novels'
    )
    parser.add_argument(
        '--mode',
        choices=['web', 'cli'],
        default='web',
        help='Run mode: web (default) or cli'
    )
    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Web server host (default: 127.0.0.1)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Web server port (default: 8000)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'web':
        run_web(args.host, args.port)
    elif args.mode == 'cli':
        run_cli()


if __name__ == '__main__':
    main()
