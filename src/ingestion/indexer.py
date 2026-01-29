# Incremental Indexer

import logging
from dataclasses import dataclass
from pathlib import Path

from .parsers import Chapter, get_parser
from .chunker import TokenChunker
from .metadata import EntityExtractor
from ..library import NovelLibrary, ChapterMetadata, get_library
from ..retrieval.embedder import Embedder
from ..retrieval.vector_store import VectorStore
from ..retrieval.sparse_index import BM25Index
from ..config import get_config

logger = logging.getLogger(__name__)


@dataclass
class IndexResult:
    """Result of an indexing operation."""
    status: str  # created, updated, up_to_date, error
    total_chapters: int
    new_chapters: int
    updated_chapters: int
    total_chunks: int
    new_chunks: int
    error_message: str = ""


class IncrementalIndexer:
    """
    Smart indexer that only processes new or changed chapters.
    
    Uses content hashing to detect changes:
    1. Parse file into chapters
    2. Compare chapter hashes with stored metadata
    3. Only embed and index NEW or CHANGED chapters
    4. Update metadata with new state
    """
    
    def __init__(
        self,
        embedder: Embedder,
        library: NovelLibrary = None
    ):
        self.embedder = embedder
        self.library = library or get_library()
        self.config = get_config()
        
        self.chunker = TokenChunker(
            target_tokens=self.config.chunking.target_tokens,
            min_tokens=self.config.chunking.min_tokens,
            max_tokens=self.config.chunking.max_tokens,
            overlap_tokens=self.config.chunking.overlap_tokens
        )
        self.entity_extractor = EntityExtractor()
    
    def index_novel(
        self, 
        novel_id: str,
        progress_callback=None
    ) -> IndexResult:
        """
        Index a novel, processing only new or changed chapters.
        
        Args:
            novel_id: ID of the novel to index
            progress_callback: Optional callback(current, total, message)
            
        Returns:
            IndexResult with statistics
        """
        novel = self.library.get_novel(novel_id)
        if not novel:
            return IndexResult(
                status="error",
                total_chapters=0,
                new_chapters=0,
                updated_chapters=0,
                total_chunks=0,
                new_chunks=0,
                error_message=f"Novel not found: {novel_id}"
            )
        
        self.library.update_novel(novel_id, status="processing")
        
        try:
            # Get file path and parse
            file_path = self.library.get_upload_path(novel_id)
            if not file_path or not file_path.exists():
                raise FileNotFoundError(f"Upload file not found for {novel_id}")
            
            parser = get_parser(file_path)
            chapters = parser.parse(file_path)
            
            if progress_callback:
                progress_callback(0, len(chapters), "Parsed chapters")
            
            # Load existing chapter metadata
            existing_chapters = self.library.get_chapter_metadata(novel_id)
            
            # Determine what needs indexing
            chapters_to_index = []
            for chapter in chapters:
                existing = existing_chapters.get(chapter.number)
                if existing is None or existing.content_hash != chapter.content_hash:
                    chapters_to_index.append(chapter)
            
            if not chapters_to_index:
                logger.info(f"Novel {novel_id} is up to date")
                self.library.update_novel(
                    novel_id,
                    status="ready",
                    total_chapters=len(chapters)
                )
                return IndexResult(
                    status="up_to_date",
                    total_chapters=len(chapters),
                    new_chapters=0,
                    updated_chapters=0,
                    total_chunks=novel.chunks_count,
                    new_chunks=0
                )
            
            # Initialize per-novel stores
            novel_dir = self.library.get_novel_path(novel_id)
            vector_store = VectorStore(
                persist_path=str(novel_dir / "chroma_db"),
                embedder=self.embedder,
                collection_name=f"novel_{novel_id}"
            )
            bm25_index = BM25Index(
                persist_path=str(novel_dir / "bm25_index.pkl")
            )
            
            # Index new/changed chapters
            new_chunks = 0
            updated_chapters_metadata = {}
            
            for i, chapter in enumerate(chapters_to_index):
                if progress_callback:
                    progress_callback(
                        i + 1, 
                        len(chapters_to_index),
                        f"Indexing Chapter {chapter.number}: {chapter.title}"
                    )
                
                # Remove old chunks for this chapter if it exists
                existing = existing_chapters.get(chapter.number)
                if existing and existing.chunk_ids:
                    vector_store.delete_chunks(existing.chunk_ids)
                
                # Chunk the chapter
                chunks = self.chunker.chunk_chapter(
                    chapter.content,
                    chapter.number,
                    chapter.title
                )
                
                # Add to vector store
                chunk_ids = vector_store.add_chunks(chunks)
                new_chunks += len(chunks)
                
                # Extract entities
                self.entity_extractor.extract_from_text(
                    chapter.content, 
                    chapter.number
                )
                
                # Update chapter metadata
                updated_chapters_metadata[chapter.number] = ChapterMetadata(
                    chapter_number=chapter.number,
                    title=chapter.title,
                    content_hash=chapter.content_hash,
                    chunk_ids=chunk_ids
                )
            
            # Merge with existing metadata
            for num, meta in existing_chapters.items():
                if num not in updated_chapters_metadata:
                    updated_chapters_metadata[num] = meta
            
            # Save chapter metadata
            self.library.save_chapter_metadata(novel_id, updated_chapters_metadata)
            
            # Rebuild BM25 index (includes all documents)
            all_docs = vector_store.get_all_documents()
            bm25_index.build_from_documents(all_docs)
            
            # Calculate totals
            total_chunks = sum(
                len(m.chunk_ids) 
                for m in updated_chapters_metadata.values()
            )
            
            # Update novel metadata
            self.library.update_novel(
                novel_id,
                status="ready",
                chapters_indexed=len(updated_chapters_metadata),
                total_chapters=len(chapters),
                chunks_count=total_chunks
            )
            
            logger.info(
                f"Indexed {len(chapters_to_index)} chapters "
                f"({new_chunks} chunks) for novel {novel_id}"
            )
            
            return IndexResult(
                status="updated" if existing_chapters else "created",
                total_chapters=len(chapters),
                new_chapters=len([c for c in chapters_to_index 
                                  if c.number not in existing_chapters]),
                updated_chapters=len([c for c in chapters_to_index 
                                      if c.number in existing_chapters]),
                total_chunks=total_chunks,
                new_chunks=new_chunks
            )
            
        except Exception as e:
            logger.error(f"Indexing failed for {novel_id}: {e}")
            self.library.update_novel(
                novel_id,
                status="error",
                error_message=str(e)
            )
            return IndexResult(
                status="error",
                total_chapters=0,
                new_chapters=0,
                updated_chapters=0,
                total_chunks=0,
                new_chunks=0,
                error_message=str(e)
            )
    
    def get_novel_stores(self, novel_id: str) -> tuple[VectorStore, BM25Index]:
        """Get the vector store and BM25 index for a novel."""
        novel_dir = self.library.get_novel_path(novel_id)
        if not novel_dir:
            raise ValueError(f"Novel not found: {novel_id}")
        
        vector_store = VectorStore(
            persist_path=str(novel_dir / "chroma_db"),
            embedder=self.embedder,
            collection_name=f"novel_{novel_id}"
        )
        
        bm25_index = BM25Index(
            persist_path=str(novel_dir / "bm25_index.pkl")
        )
        bm25_index.load()
        
        return vector_store, bm25_index
