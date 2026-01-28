# Novel RAG Chatbot - Main Entry Point

import argparse
import logging
import sys
from pathlib import Path

import uvicorn

from src.config import get_config, reset_config
from src.pipeline import RAGPipeline
from src.ui.app import create_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('novel_rag.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Novel RAG Chatbot - Ask questions about your novel'
    )
    
    parser.add_argument(
        '--mode',
        choices=['web', 'cli', 'ingest'],
        default='web',
        help='Run mode: web (default), cli, or ingest-only'
    )
    
    parser.add_argument(
        '--novel',
        type=str,
        default='novel.txt',
        help='Path to novel text file'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Host for web server'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port for web server'
    )
    
    parser.add_argument(
        '--force-reindex',
        action='store_true',
        help='Force rebuild of index'
    )
    
    args = parser.parse_args()
    
    # Initialize config
    reset_config()
    config = get_config(args.config)
    
    if args.mode == 'ingest':
        # Ingest only mode
        run_ingest(args.novel, args.force_reindex)
        
    elif args.mode == 'cli':
        # CLI mode
        run_cli(args.novel, args.force_reindex)
        
    else:
        # Web mode (default)
        run_web(args.host, args.port, args.novel, args.force_reindex)


def run_ingest(novel_path: str, force_reindex: bool):
    """Run ingestion only."""
    logger.info(f"Ingesting novel: {novel_path}")
    
    pipeline = RAGPipeline()
    result = pipeline.ingest_novel(Path(novel_path), force_reindex)
    
    print(f"\nIngestion complete:")
    print(f"  Status: {result.get('status')}")
    print(f"  Chapters: {result.get('chapters', 0)}")
    print(f"  Chunks: {result.get('chunks', 0)}")
    print(f"  Entities: {result.get('entities', 0)}")
    print(f"  Time: {result.get('time_seconds', 0):.2f}s")


def run_cli(novel_path: str, force_reindex: bool):
    """Run CLI interactive mode."""
    logger.info("Starting CLI mode...")
    
    pipeline = RAGPipeline()
    
    # Ingest if needed
    if not pipeline.is_ready() or force_reindex:
        print(f"Loading novel: {novel_path}")
        result = pipeline.ingest_novel(Path(novel_path), force_reindex)
        print(f"Loaded {result.get('chunks', 0)} chunks\n")
    
    print("Novel RAG Chatbot - CLI Mode")
    print("Type 'quit' or 'exit' to stop\n")
    
    while True:
        try:
            query = input("You: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if not query:
                continue
            
            result = pipeline.query(query)
            
            print(f"\nAssistant: {result.answer}")
            
            if result.chapters_cited:
                print(f"\n[Based on: {', '.join(f'Chapter {c}' for c in result.chapters_cited)}]")
            
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"Error: {e}\n")


def run_web(host: str, port: int, novel_path: str, force_reindex: bool):
    """Run web server."""
    logger.info(f"Starting web server at http://{host}:{port}")
    
    # Pre-load if novel exists
    if Path(novel_path).exists():
        logger.info(f"Pre-loading novel: {novel_path}")
        pipeline = RAGPipeline()
        pipeline.ingest_novel(Path(novel_path), force_reindex)
    
    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
