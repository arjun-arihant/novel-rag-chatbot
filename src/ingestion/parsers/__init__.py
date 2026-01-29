# File Format Parsers

from .base import BaseParser, Chapter
from .txt_parser import TxtParser
from .epub_parser import EpubParser
from .pdf_parser import PdfParser

__all__ = ["BaseParser", "Chapter", "TxtParser", "EpubParser", "PdfParser", "get_parser"]


def get_parser(file_path) -> BaseParser:
    """
    Get the appropriate parser for a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Parser instance for the file type
        
    Raises:
        ValueError: If no parser supports the file type
    """
    from pathlib import Path
    path = Path(file_path)
    ext = path.suffix.lower()
    
    parsers = {
        ".txt": TxtParser,
        ".epub": EpubParser,
        ".pdf": PdfParser,
    }
    
    parser_class = parsers.get(ext)
    if parser_class is None:
        raise ValueError(f"Unsupported file format: {ext}")
    
    return parser_class()
