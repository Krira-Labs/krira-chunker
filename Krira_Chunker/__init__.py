"""
Krira_Chunker - Production-grade document chunking library for RAG.

This library provides best-in-class chunking for various file formats:
- PDF, DOCX, CSV, XLSX, JSON/JSONL, XML, URLs

Key features:
- Hybrid boundary-aware chunking (avoids splitting sentences/code blocks)
- Streaming-first ingestion (memory efficient)
- Deterministic chunk IDs and consistent ordering
- SSRF protection for URL fetching
- Optional dependencies via extras

Usage:
    # Simple usage with auto-detection
    from Krira_Chunker import iter_chunks_auto, ChunkConfig

    cfg = ChunkConfig(max_chars=2000, chunk_strategy="hybrid")
    for chunk in iter_chunks_auto("document.pdf", cfg):
        print(chunk["text"])

    # Class-based usage
    from Krira_Chunker.PDFChunker import PDFChunker

    pdf = PDFChunker(cfg)
    for chunk in pdf.chunk_file("report.pdf"):
        print(chunk["text"])

    # Facade pattern
    from Krira_Chunker import KriraChunker

    engine = KriraChunker(cfg)
    for chunk in engine.process("any_file.pdf"):
        print(chunk["text"])
"""

# Core imports (no heavy dependencies)
from .config import ChunkConfig
from .core import FastChunker, HybridBoundaryChunker, LOGGER, clean_text, stable_id
from .router import iter_chunks_auto, stream_chunks_to_sink
from .exceptions import (
    KriraChunkerError,
    ConfigError,
    DependencyNotInstalledError,
    UnsupportedFormatError,
    SecurityViolationError,
    SSRFError,
    FileSizeLimitError,
    ContentTypeDeniedError,
    ZipSlipError,
    ProcessingError,
    OCRRequiredError,
    EmptyDocumentError,
)

# Lazy imports for format-specific functions (backward compatibility)
# These are imported on first use to avoid loading heavy dependencies
def __getattr__(name: str):
    """Lazy attribute loading for format-specific exports."""
    if name == "iter_chunks_from_csv":
        from .CSVChunker import iter_chunks_from_csv
        return iter_chunks_from_csv
    elif name == "iter_chunks_from_json":
        from .JSON_JSONLChunker import iter_chunks_from_json
        return iter_chunks_from_json
    elif name == "iter_chunks_from_pdf":
        from .PDFChunker import iter_chunks_from_pdf
        return iter_chunks_from_pdf
    elif name == "iter_chunks_from_docx":
        from .DOCXChunker import iter_chunks_from_docx
        return iter_chunks_from_docx
    elif name == "iter_chunks_from_xml":
        from .XMLChunker import iter_chunks_from_xml
        return iter_chunks_from_xml
    elif name == "iter_chunks_from_xlsx":
        from .XLSXChunker import iter_chunks_from_xlsx
        return iter_chunks_from_xlsx
    elif name == "iter_chunks_from_url":
        from .URLChunker import iter_chunks_from_url
        return iter_chunks_from_url
    elif name == "iter_chunks_from_text":
        from .TextChunker import iter_chunks_from_text
        return iter_chunks_from_text
    elif name == "iter_chunks_from_markdown":
        from .TextChunker import iter_chunks_from_markdown
        return iter_chunks_from_markdown
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class KriraChunker:
    """
    Facade class for convenient document chunking.
    
    Automatically detects file format and uses appropriate chunker.
    
    Example:
        >>> cfg = ChunkConfig(max_chars=2000)
        >>> engine = KriraChunker(cfg)
        >>> for chunk in engine.process("report.pdf"):
        ...     print(chunk["text"][:100])
    """
    
    def __init__(self, cfg: ChunkConfig = None):
        """
        Initialize KriraChunker.
        
        Args:
            cfg: Chunk configuration. Uses defaults if None.
        """
        self.cfg = cfg or ChunkConfig()
    
    def process(self, source: str):
        """
        Process a source and yield chunks.
        
        Args:
            source: Path to file or URL.
            
        Yields:
            Chunk dictionaries.
        """
        return iter_chunks_auto(source, self.cfg)
    
    def process_to_list(self, source: str) -> list:
        """
        Process a source and return all chunks as a list.
        
        Args:
            source: Path to file or URL.
            
        Returns:
            List of chunk dictionaries.
        """
        return list(self.process(source))
    
    def process_to_sink(
        self,
        source: str,
        sink,
        batch_size: int = None
    ) -> int:
        """
        Process a source and stream to a sink.
        
        Args:
            source: Path to file or URL.
            sink: Callable that receives batches of chunks.
            batch_size: Batch size override.
            
        Returns:
            Total number of chunks processed.
        """
        return stream_chunks_to_sink(source, sink, self.cfg, batch_size)


__all__ = [
    # Configuration
    "ChunkConfig",
    
    # Core classes
    "FastChunker",
    "HybridBoundaryChunker",
    "KriraChunker",
    
    # Router functions
    "iter_chunks_auto",
    "stream_chunks_to_sink",
    
    # Format-specific functions (lazy loaded)
    "iter_chunks_from_csv",
    "iter_chunks_from_json",
    "iter_chunks_from_pdf",
    "iter_chunks_from_docx",
    "iter_chunks_from_xml",
    "iter_chunks_from_xlsx",
    "iter_chunks_from_url",
    "iter_chunks_from_text",
    "iter_chunks_from_markdown",
    
    # Utilities
    "LOGGER",
    "clean_text",
    "stable_id",
    
    # Exceptions
    "KriraChunkerError",
    "ConfigError",
    "DependencyNotInstalledError",
    "UnsupportedFormatError",
    "SecurityViolationError",
    "SSRFError",
    "FileSizeLimitError",
    "ContentTypeDeniedError",
    "ZipSlipError",
    "ProcessingError",
    "OCRRequiredError",
    "EmptyDocumentError",
]

__version__ = "0.2.0"
