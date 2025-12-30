"""
Text and Markdown chunker for Krira_Chunker.
"""

import os
from typing import Generator, Dict, Any, Iterator

from ..config import ChunkConfig
from ..core import FastChunker, HybridBoundaryChunker, clean_text
from ..exceptions import FileSizeLimitError


class TextChunker:
    """
    Class-based text/markdown chunker.
    
    Handles plain text (.txt) and markdown (.md) files.
    Uses HybridBoundaryChunker for markdown to preserve code blocks and structure.
    
    Example:
        >>> cfg = ChunkConfig(max_chars=2000)
        >>> chunker = TextChunker(cfg)
        >>> for chunk in chunker.chunk_file("document.md"):
        ...     print(chunk["text"][:100])
    """
    
    def __init__(self, cfg: ChunkConfig = None):
        """
        Initialize text chunker.
        
        Args:
            cfg: Chunk configuration. Uses defaults if None.
        """
        self.cfg = cfg or ChunkConfig()
        self._chunker = None
        self._hybrid_chunker = None
    
    @property
    def chunker(self) -> FastChunker:
        """Lazy-load FastChunker."""
        if self._chunker is None:
            self._chunker = FastChunker(self.cfg)
        return self._chunker
    
    @property
    def hybrid_chunker(self) -> HybridBoundaryChunker:
        """Lazy-load HybridBoundaryChunker."""
        if self._hybrid_chunker is None:
            self._hybrid_chunker = HybridBoundaryChunker(self.cfg)
        return self._hybrid_chunker
    
    def chunk_file(self, file_path: str) -> Iterator[Dict[str, Any]]:
        """
        Chunk a text or markdown file.
        
        Args:
            file_path: Path to text file.
            
        Yields:
            Chunk dictionaries.
            
        Raises:
            FileSizeLimitError: If file exceeds size limit.
        """
        cfg = self.cfg
        
        # Security: check file size
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            if size > cfg.security_max_file_bytes:
                raise FileSizeLimitError(file_path, size, cfg.security_max_file_bytes)
        
        # Detect file type
        ext = file_path.lower()
        is_markdown = ext.endswith(".md") or ext.endswith(".markdown")
        
        source_type = "markdown" if is_markdown else "text"
        
        base_meta = {
            "source": os.path.basename(file_path),
            "source_path": os.path.abspath(file_path),
            "source_type": source_type,
        }
        
        # Read file content
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            from ..core import LOGGER
            LOGGER.error("Error reading file %s: %s", file_path, e)
            return
        
        text = clean_text(text)
        if not text:
            return
        
        chunk_index = 0
        
        # Use hybrid chunker for markdown OR if hybrid strategy is set
        if is_markdown or cfg.chunk_strategy == "hybrid":
            for ch in self.hybrid_chunker.chunk_text(
                text=text,
                base_meta=base_meta,
                locator=source_type,
                start_chunk_index=chunk_index,
            ):
                yield ch
        else:
            for ch in self.chunker.chunk_text(
                text=text,
                base_meta=base_meta,
                mode="prose",
                locator=source_type,
                joiner=" ",
                start_chunk_index=chunk_index,
            ):
                yield ch


# Backward compatibility functions
def iter_chunks_from_text(
    file_path: str,
    cfg: ChunkConfig = None
) -> Generator[Dict[str, Any], None, None]:
    """
    Iterate over chunks from a text file.
    
    Args:
        file_path: Path to text file.
        cfg: Chunk configuration.
        
    Yields:
        Chunk dictionaries.
    """
    chunker = TextChunker(cfg)
    yield from chunker.chunk_file(file_path)


def iter_chunks_from_markdown(
    file_path: str,
    cfg: ChunkConfig = None
) -> Generator[Dict[str, Any], None, None]:
    """
    Iterate over chunks from a markdown file.
    
    Args:
        file_path: Path to markdown file.
        cfg: Chunk configuration.
        
    Yields:
        Chunk dictionaries.
    """
    chunker = TextChunker(cfg)
    yield from chunker.chunk_file(file_path)
