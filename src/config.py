# Novel RAG Chatbot - Configuration

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model: str = "qwen3-embedding:0.6b"
    base_url: str = "http://localhost:11434"
    dimension: int = 1024  # Verify with actual model


@dataclass
class QueryRewriterConfig:
    """Query rewriter - ultra-strict, deterministic."""
    model: str = "llama3.1:8b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    max_tokens: int = 128
    stop_tokens: tuple = ("\n", "?", "Query:", "Question:")


@dataclass
class RerankerConfig:
    """Reranker - JSON output, deterministic scoring."""
    model: str = "llama3.1:8b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    max_tokens: int = 64
    format: str = "json"


@dataclass
class GeneratorConfig:
    """Answer generator - grounded, concise."""
    model: str = "llama3.1:8b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.25
    max_tokens: int = 512
    default_sentences: tuple = (3, 6)  # min, max default


@dataclass 
class ChunkingConfig:
    """Chunking parameters - token-based."""
    target_tokens: int = 500
    min_tokens: int = 400
    max_tokens: int = 800
    overlap_tokens: int = 100
    malformed_threshold: float = 0.1  # Fallback to paragraph if >10% malformed


@dataclass
class RetrievalConfig:
    """Retrieval parameters."""
    dense_top_k: int = 32
    sparse_top_k: int = 32
    fusion_top_k: int = 24
    rerank_top_k: int = 10
    final_top_k: int = 5
    rrf_k: int = 60  # RRF constant
    dense_weight: float = 1.0
    sparse_weight: float = 1.15
    min_rerank_score: float = 3.0  # Refuse if top score below this


@dataclass
class PathsConfig:
    """File paths."""
    novel: Path = Path("novel.txt")
    chroma_db: Path = Path("chroma_db_v2")
    bm25_index: Path = Path("bm25_index.pkl")
    chapter_index: Path = Path("chapter_index_v2.json")


@dataclass
class UIConfig:
    """UI configuration."""
    host: str = "127.0.0.1"
    port: int = 8000
    theme: str = "dark"


@dataclass
class Config:
    """Main configuration container."""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    query_rewriter: QueryRewriterConfig = field(default_factory=QueryRewriterConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    pipeline_mode: str = "advanced"
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        config = cls()
        
        if not Path(path).exists():
            return config
            
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        
        # Update from YAML (simple merge)
        if 'embedding' in data:
            for k, v in data['embedding'].items():
                if hasattr(config.embedding, k):
                    setattr(config.embedding, k, v)
                    
        if 'query_rewriter' in data:
            for k, v in data['query_rewriter'].items():
                if hasattr(config.query_rewriter, k):
                    setattr(config.query_rewriter, k, v)
                    
        if 'reranker' in data:
            for k, v in data['reranker'].items():
                if hasattr(config.reranker, k):
                    setattr(config.reranker, k, v)
                    
        if 'generator' in data:
            for k, v in data['generator'].items():
                if hasattr(config.generator, k):
                    setattr(config.generator, k, v)
                    
        if 'chunking' in data:
            for k, v in data['chunking'].items():
                if hasattr(config.chunking, k):
                    setattr(config.chunking, k, v)
                    
        if 'retrieval' in data:
            for k, v in data['retrieval'].items():
                if hasattr(config.retrieval, k):
                    setattr(config.retrieval, k, v)
                    
        if 'paths' in data:
            for k, v in data['paths'].items():
                if hasattr(config.paths, k):
                    setattr(config.paths, k, Path(v))
                    
        if 'ui' in data:
            for k, v in data['ui'].items():
                if hasattr(config.ui, k):
                    setattr(config.ui, k, v)

        if 'pipeline_mode' in data:
            mode = str(data['pipeline_mode']).strip().lower()
            if mode in {'simple', 'advanced'}:
                config.pipeline_mode = mode
        
        return config


# Global config instance
_config: Optional[Config] = None


def get_config(config_path: str = "config.yaml") -> Config:
    """Get or create global config instance."""
    global _config
    if _config is None:
        _config = Config.from_yaml(config_path)
    return _config


def reset_config():
    """Reset global config (for testing)."""
    global _config
    _config = None
