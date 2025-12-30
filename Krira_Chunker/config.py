"""
Configuration for Krira_Chunker.
"""

from dataclasses import dataclass, field
from typing import Literal, Tuple, Optional
import hashlib
import json

from .exceptions import ConfigError


ChunkStrategy = Literal["fixed", "sentence", "markdown", "hybrid"]


@dataclass(frozen=True)
class ChunkConfig:
    """
    Configuration for the chunking process.
    
    Attributes:
        max_chars: Maximum characters per chunk (when use_tokens=False).
        overlap_chars: Overlap characters between chunks.
        use_tokens: If True, use token-based limits instead of char-based.
        max_tokens: Maximum tokens per chunk (when use_tokens=True).
        overlap_tokens: Overlap tokens between chunks.
        min_chars: Minimum characters for a valid chunk.
        
        chunk_strategy: Chunking strategy to use.
            - "fixed": Simple fixed-size chunks.
            - "sentence": Split at sentence boundaries.
            - "markdown": Respect markdown structure.
            - "hybrid": Best-effort boundary-aware chunking (default).
        
        preserve_code_blocks: Avoid splitting inside code blocks.
        preserve_tables: Avoid splitting inside tables.
        preserve_lists: Avoid splitting list items.
        
        rows_per_chunk: For CSV/XLSX, max rows per chunk (None = auto).
        
        sink_batch_size: Batch size for sink operations.
        csv_batch_rows: Rows per batch for CSV streaming.
        xlsx_batch_rows: Rows per batch for XLSX streaming.
        xlsx_batch_overlap_rows: Overlap rows between XLSX batches.
        
        http_timeout_s: HTTP request timeout.
        http_user_agent: User agent for HTTP requests.
        url_retries: Number of retries for URL requests.
        url_backoff_factor: Backoff factor for retries.
        url_max_bytes: Maximum bytes to download from URL.
        url_allow_private: Allow fetching from private IP ranges.
        url_rate_limit_s: Delay between URL requests.
        url_content_type_allowlist: Allowed content types for URL fetching.
        
        security_max_file_bytes: Maximum file size allowed.
        pdf_min_chars_per_page: Minimum chars per page before OCR warning.
    """
    
    # Chunk sizing
    max_chars: int = 2200
    overlap_chars: int = 250
    
    # Token-based control (OFF by default for speed)
    use_tokens: bool = False
    max_tokens: int = 512
    overlap_tokens: int = 64
    
    # Filters
    min_chars: int = 30
    
    # Chunking strategy
    chunk_strategy: ChunkStrategy = "hybrid"
    
    # Preservation flags
    preserve_code_blocks: bool = True
    preserve_tables: bool = True
    preserve_lists: bool = True
    
    # Tabular data
    rows_per_chunk: Optional[int] = None
    
    # Streaming/batching
    sink_batch_size: int = 256
    csv_batch_rows: int = 50_000
    xlsx_batch_rows: int = 25_000
    xlsx_batch_overlap_rows: int = 200
    
    # HTTP settings
    http_timeout_s: int = 15
    http_user_agent: str = "Mozilla/5.0 (compatible; KriraChunker/1.0)"
    url_retries: int = 3
    url_backoff_factor: float = 0.5
    url_max_bytes: int = 8 * 1024 * 1024  # 8MB
    url_allow_private: bool = False  # SSRF protection default
    url_rate_limit_s: float = 0.0
    url_content_type_allowlist: Tuple[str, ...] = (
        "text/html",
        "application/xhtml+xml",
        "text/plain",
        "application/xml",
        "text/xml",
    )
    
    # Security
    security_max_file_bytes: int = 50_000_000  # 50MB
    
    # PDF settings
    pdf_min_chars_per_page: int = 25
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.overlap_chars >= self.max_chars:
            raise ConfigError(
                f"overlap_chars ({self.overlap_chars}) must be less than "
                f"max_chars ({self.max_chars})"
            )
        if self.use_tokens and self.overlap_tokens >= self.max_tokens:
            raise ConfigError(
                f"overlap_tokens ({self.overlap_tokens}) must be less than "
                f"max_tokens ({self.max_tokens})"
            )
        if self.max_chars <= 0:
            raise ConfigError(f"max_chars must be positive, got {self.max_chars}")
        if self.overlap_chars < 0:
            raise ConfigError(f"overlap_chars must be non-negative, got {self.overlap_chars}")
        if self.min_chars < 0:
            raise ConfigError(f"min_chars must be non-negative, got {self.min_chars}")
    
    def config_hash(self) -> str:
        """
        Generate a stable hash of chunking-relevant configuration.
        
        Returns:
            First 12 characters of MD5 hash.
        """
        config_dict = {
            "max_chars": self.max_chars,
            "overlap_chars": self.overlap_chars,
            "use_tokens": self.use_tokens,
            "max_tokens": self.max_tokens,
            "overlap_tokens": self.overlap_tokens,
            "min_chars": self.min_chars,
            "chunk_strategy": self.chunk_strategy,
            "preserve_code_blocks": self.preserve_code_blocks,
            "preserve_tables": self.preserve_tables,
            "preserve_lists": self.preserve_lists,
        }
        raw = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()[:12]
    
    def get_max_size(self) -> int:
        """Get max size in the appropriate unit (chars or tokens)."""
        return self.max_tokens if self.use_tokens else self.max_chars
    
    def get_overlap_size(self) -> int:
        """Get overlap size in the appropriate unit (chars or tokens)."""
        return self.overlap_tokens if self.use_tokens else self.overlap_chars
