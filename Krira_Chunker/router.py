"""
Router for automatic format detection and dispatching.
"""

import os
from typing import Generator, Dict, Any, Optional, Callable, List, Union

from .config import ChunkConfig
from .exceptions import UnsupportedFormatError


def iter_chunks_auto(
    input_path: str,
    cfg: Optional[ChunkConfig] = None
) -> Generator[Dict[str, Any], None, None]:
    """
    Automatically detect file type and iterate over chunks.
    
    Supports local files and URLs. Format is detected from:
    - URL scheme (http/https)
    - File extension
    
    Args:
        input_path: Path to file or URL.
        cfg: Chunk configuration. Uses defaults if None.
        
    Yields:
        Chunk dictionaries.
        
    Raises:
        FileNotFoundError: If local file doesn't exist.
        UnsupportedFormatError: If format is not supported.
    """
    if cfg is None:
        cfg = ChunkConfig()
    
    # URL detection
    if input_path.startswith("http://") or input_path.startswith("https://"):
        from .URLChunker import iter_chunks_from_url
        yield from iter_chunks_from_url(input_path, cfg)
        return
    
    # Local file
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
    
    ext = input_path.lower()
    
    if ext.endswith(".csv"):
        from .CSVChunker import iter_chunks_from_csv
        yield from iter_chunks_from_csv(input_path, cfg)
    elif ext.endswith(".json") or ext.endswith(".jsonl") or ext.endswith(".ndjson"):
        from .JSON_JSONLChunker import iter_chunks_from_json
        yield from iter_chunks_from_json(input_path, cfg)
    elif ext.endswith(".pdf"):
        from .PDFChunker import iter_chunks_from_pdf
        yield from iter_chunks_from_pdf(input_path, cfg)
    elif ext.endswith(".docx"):
        from .DOCXChunker import iter_chunks_from_docx
        yield from iter_chunks_from_docx(input_path, cfg)
    elif ext.endswith(".xml"):
        from .XMLChunker import iter_chunks_from_xml
        yield from iter_chunks_from_xml(input_path, cfg)
    elif ext.endswith(".xlsx") or ext.endswith(".xls"):
        from .XLSXChunker import iter_chunks_from_xlsx
        yield from iter_chunks_from_xlsx(input_path, cfg)
    elif ext.endswith(".txt") or ext.endswith(".text"):
        from .TextChunker import iter_chunks_from_text
        yield from iter_chunks_from_text(input_path, cfg)
    elif ext.endswith(".md") or ext.endswith(".markdown"):
        from .TextChunker import iter_chunks_from_markdown
        yield from iter_chunks_from_markdown(input_path, cfg)
    else:
        # Try to detect extension
        _, file_ext = os.path.splitext(input_path)
        raise UnsupportedFormatError(input_path, file_ext)


def stream_chunks_to_sink(
    input_path: str,
    sink: Callable[[List[Dict[str, Any]]], None],
    cfg: Optional[ChunkConfig] = None,
    batch_size: Optional[int] = None,
) -> int:
    """
    Stream chunks to a sink function in batches.
    
    Args:
        input_path: Path to file or URL.
        sink: Callable that receives batches of chunks.
        cfg: Chunk configuration.
        batch_size: Batch size (overrides cfg.sink_batch_size).
        
    Returns:
        Total number of chunks processed.
    """
    if cfg is None:
        cfg = ChunkConfig()
    
    bs = batch_size or cfg.sink_batch_size
    batch: List[Dict[str, Any]] = []
    total = 0
    
    for ch in iter_chunks_auto(input_path, cfg):
        batch.append(ch)
        total += 1
        if len(batch) >= bs:
            sink(batch)
            batch = []
    
    if batch:
        sink(batch)
    
    return total
