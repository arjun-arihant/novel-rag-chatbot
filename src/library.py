# Novel Library Manager

import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class NovelMetadata:
    """Metadata for a novel in the library."""
    id: str
    title: str
    author: str
    filename: str
    file_hash: str
    format: str  # txt, pdf, epub
    chapters_indexed: int = 0
    total_chapters: int = 0
    chunks_count: int = 0
    status: str = "pending"  # pending, processing, ready, error
    error_message: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "NovelMetadata":
        return cls(**data)


@dataclass
class ChapterMetadata:
    """Metadata for an indexed chapter."""
    chapter_number: int
    title: str
    content_hash: str
    chunk_ids: list = field(default_factory=list)
    indexed_at: str = field(default_factory=lambda: datetime.now().isoformat())


class NovelLibrary:
    """
    Manages a library of novels with per-novel databases.
    
    Storage structure:
    library/
    ├── metadata.json       # All novel metadata
    ├── uploads/            # Original uploaded files
    ├── {novel_id}/
    │   ├── chapters.json   # Chapter-level metadata for incremental indexing
    │   ├── chroma_db/      # Vector store
    │   └── bm25_index.pkl  # Sparse index
    """
    
    def __init__(self, library_path: str = "library"):
        self.library_path = Path(library_path)
        self.library_path.mkdir(exist_ok=True)
        (self.library_path / "uploads").mkdir(exist_ok=True)
        
        self.metadata_file = self.library_path / "metadata.json"
        self._novels: dict[str, NovelMetadata] = {}
        self._load_metadata()
        
        self._active_novel_id: Optional[str] = None
    
    def _load_metadata(self):
        """Load novel metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._novels = {
                        k: NovelMetadata.from_dict(v) 
                        for k, v in data.items()
                    }
                logger.info(f"Loaded {len(self._novels)} novels from library")
            except Exception as e:
                logger.error(f"Failed to load library metadata: {e}")
                self._novels = {}
        else:
            self._novels = {}
    
    def _save_metadata(self):
        """Persist novel metadata to disk."""
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(
                    {k: v.to_dict() for k, v in self._novels.items()},
                    f,
                    indent=2
                )
        except Exception as e:
            logger.error(f"Failed to save library metadata: {e}")
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file for change detection."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]
    
    def _detect_format(self, filename: str) -> str:
        """Detect file format from extension."""
        ext = Path(filename).suffix.lower()
        format_map = {
            ".txt": "txt",
            ".pdf": "pdf",
            ".epub": "epub"
        }
        return format_map.get(ext, "unknown")
    
    def list_novels(self) -> list[NovelMetadata]:
        """Get all novels in the library."""
        return list(self._novels.values())
    
    def get_novel(self, novel_id: str) -> Optional[NovelMetadata]:
        """Get metadata for a specific novel."""
        return self._novels.get(novel_id)
    
    def add_novel(
        self, 
        file_path: Path, 
        title: Optional[str] = None,
        author: str = "Unknown"
    ) -> NovelMetadata:
        """
        Add a new novel to the library.
        
        Args:
            file_path: Path to the novel file
            title: Optional title (defaults to filename)
            author: Author name
            
        Returns:
            NovelMetadata for the new novel
        """
        novel_id = str(uuid4())[:8]
        file_hash = self._compute_file_hash(file_path)
        format_type = self._detect_format(file_path.name)
        
        if format_type == "unknown":
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Check for duplicate by hash
        for existing in self._novels.values():
            if existing.file_hash == file_hash:
                logger.info(f"Novel already exists: {existing.title}")
                return existing
        
        # Copy file to uploads
        upload_path = self.library_path / "uploads" / f"{novel_id}{file_path.suffix}"
        upload_path.write_bytes(file_path.read_bytes())
        
        # Create novel directory
        novel_dir = self.library_path / novel_id
        novel_dir.mkdir(exist_ok=True)
        
        metadata = NovelMetadata(
            id=novel_id,
            title=title or file_path.stem,
            author=author,
            filename=file_path.name,
            file_hash=file_hash,
            format=format_type,
            status="pending"
        )
        
        self._novels[novel_id] = metadata
        self._save_metadata()
        
        logger.info(f"Added novel: {metadata.title} ({novel_id})")
        return metadata
    
    def update_novel(self, novel_id: str, **updates) -> Optional[NovelMetadata]:
        """Update novel metadata."""
        if novel_id not in self._novels:
            return None
        
        novel = self._novels[novel_id]
        for key, value in updates.items():
            if hasattr(novel, key):
                setattr(novel, key, value)
        
        novel.updated_at = datetime.now().isoformat()
        self._save_metadata()
        return novel
    
    def delete_novel(self, novel_id: str) -> bool:
        """Delete a novel and all its data."""
        if novel_id not in self._novels:
            return False
        
        novel = self._novels[novel_id]
        
        # Delete novel directory
        novel_dir = self.library_path / novel_id
        if novel_dir.exists():
            import shutil
            shutil.rmtree(novel_dir)
        
        # Delete uploaded file
        for ext in [".txt", ".pdf", ".epub"]:
            upload_file = self.library_path / "uploads" / f"{novel_id}{ext}"
            if upload_file.exists():
                upload_file.unlink()
        
        del self._novels[novel_id]
        self._save_metadata()
        
        logger.info(f"Deleted novel: {novel.title} ({novel_id})")
        return True
    
    def get_novel_path(self, novel_id: str) -> Optional[Path]:
        """Get the path to a novel's directory."""
        if novel_id not in self._novels:
            return None
        return self.library_path / novel_id
    
    def get_upload_path(self, novel_id: str) -> Optional[Path]:
        """Get the path to an uploaded novel file."""
        novel = self._novels.get(novel_id)
        if not novel:
            return None
        
        ext_map = {"txt": ".txt", "pdf": ".pdf", "epub": ".epub"}
        ext = ext_map.get(novel.format, ".txt")
        return self.library_path / "uploads" / f"{novel_id}{ext}"
    
    def get_chapter_metadata(self, novel_id: str) -> dict[int, ChapterMetadata]:
        """Load chapter-level indexing metadata."""
        novel_dir = self.library_path / novel_id
        chapters_file = novel_dir / "chapters.json"
        
        if not chapters_file.exists():
            return {}
        
        try:
            with open(chapters_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {
                    int(k): ChapterMetadata(**v)
                    for k, v in data.items()
                }
        except Exception as e:
            logger.error(f"Failed to load chapter metadata: {e}")
            return {}
    
    def save_chapter_metadata(
        self, 
        novel_id: str, 
        chapters: dict[int, ChapterMetadata]
    ):
        """Save chapter-level indexing metadata."""
        novel_dir = self.library_path / novel_id
        novel_dir.mkdir(exist_ok=True)
        chapters_file = novel_dir / "chapters.json"
        
        try:
            with open(chapters_file, "w", encoding="utf-8") as f:
                json.dump(
                    {str(k): asdict(v) for k, v in chapters.items()},
                    f,
                    indent=2
                )
        except Exception as e:
            logger.error(f"Failed to save chapter metadata: {e}")
    
    # Active novel management
    def get_active_novel(self) -> Optional[NovelMetadata]:
        """Get currently selected novel."""
        if self._active_novel_id:
            return self._novels.get(self._active_novel_id)
        return None
    
    def set_active_novel(self, novel_id: str) -> bool:
        """Set the active novel for querying."""
        if novel_id in self._novels:
            novel = self._novels[novel_id]
            if novel.status == "ready":
                self._active_novel_id = novel_id
                logger.info(f"Active novel set to: {novel.title}")
                return True
            else:
                logger.warning(f"Novel {novel_id} is not ready (status: {novel.status})")
                return False
        return False
    
    def clear_active_novel(self):
        """Clear the active novel selection."""
        self._active_novel_id = None


# Singleton instance
_library: Optional[NovelLibrary] = None


def get_library(library_path: str = "library") -> NovelLibrary:
    """Get or create the global library instance."""
    global _library
    if _library is None:
        _library = NovelLibrary(library_path)
    return _library


def reset_library():
    """Reset the global library instance."""
    global _library
    _library = None
