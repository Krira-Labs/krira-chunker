"""
JSON/JSONL chunker for Krira_Chunker.
"""

import os
import json
from typing import Generator, Dict, Any, List, Optional, Iterable, Tuple, Iterator

from ..config import ChunkConfig
from ..core import FastChunker, LOGGER
from ..exceptions import DependencyNotInstalledError, FileSizeLimitError, ProcessingError


def _flatten_json(obj: Any, prefix: str = "") -> Iterable[Tuple[str, Any]]:
    """Recursively flatten JSON object to key-value pairs."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            yield from _flatten_json(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{prefix}[{i}]" if prefix else f"[{i}]"
            yield from _flatten_json(v, p)
    else:
        yield prefix or "value", obj


def _format_kv(path: str, value: Any) -> Optional[str]:
    """Format key-value pair as string."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        try:
            value = json.dumps(value, ensure_ascii=False)
        except Exception:
            value = str(value)
    s = str(value).strip()
    if not s:
        return None
    return f"{path}: {s}"


def _iter_jsonl_records(file_path: str) -> Iterable[Tuple[int, Any]]:
    """Stream JSONL records line by line."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for rec_i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield rec_i, json.loads(line)
            except Exception:
                continue


class JSONChunker:
    """
    Class-based JSON/JSONL chunker with lazy ijson loading.
    
    Supports:
    - JSONL/NDJSON: Line-by-line streaming
    - Large JSON arrays: Streaming via ijson if available
    - Regular JSON: Full load with record-wise chunking
    
    Example:
        >>> cfg = ChunkConfig(max_chars=2000)
        >>> chunker = JSONChunker(cfg)
        >>> for chunk in chunker.chunk_file("data.jsonl"):
        ...     print(chunk["text"][:100])
    """
    
    def __init__(self, cfg: ChunkConfig = None):
        """
        Initialize JSON chunker.
        
        Args:
            cfg: Chunk configuration. Uses defaults if None.
        """
        self.cfg = cfg or ChunkConfig()
        self._chunker = None
    
    @property
    def chunker(self) -> FastChunker:
        """Lazy-load FastChunker."""
        if self._chunker is None:
            self._chunker = FastChunker(self.cfg)
        return self._chunker
    
    def _get_ijson(self):
        """Lazy import ijson (optional)."""
        try:
            import ijson
            return ijson
        except ImportError:
            return None
    
    def _iter_json_streaming(self, file_path: str) -> Optional[Iterable[Tuple[int, Any]]]:
        """
        Stream large JSON using ijson if available.
        
        Supports:
        - Top-level array: [{...}, {...}]
        - Object with array: {"items": [...]}
        """
        ijson = self._get_ijson()
        if ijson is None:
            return None
        
        def first_non_ws_char(path: str) -> str:
            with open(path, "rb") as f:
                while True:
                    b = f.read(1)
                    if not b:
                        return ""
                    c = chr(b[0])
                    if c.isspace():
                        continue
                    return c
        
        lead = first_non_ws_char(file_path)
        
        if lead == "[":
            def gen() -> Iterable[Tuple[int, Any]]:
                with open(file_path, "rb") as f:
                    for i, item in enumerate(ijson.items(f, "item")):
                        yield i, item
            return gen()
        
        if lead == "{":
            # Find first array key
            array_key = None
            with open(file_path, "rb") as f:
                parser = ijson.parse(f)
                last_key = None
                for prefix, event, value in parser:
                    if prefix == "" and event == "map_key":
                        last_key = value
                    elif last_key and prefix == last_key and event == "start_array":
                        array_key = last_key
                        break
            
            if not array_key:
                return None
            
            def gen_obj() -> Iterable[Tuple[int, Any]]:
                with open(file_path, "rb") as f:
                    for i, item in enumerate(ijson.items(f, f"{array_key}.item")):
                        yield i, item
            return gen_obj()
        
        return None
    
    def chunk_file(self, file_path: str) -> Iterator[Dict[str, Any]]:
        """
        Chunk a JSON or JSONL file.
        
        Args:
            file_path: Path to JSON/JSONL file.
            
        Yields:
            Chunk dictionaries with record_index metadata.
            
        Raises:
            FileSizeLimitError: If file exceeds size limit.
        """
        cfg = self.cfg
        
        # Security: check file size
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            if size > cfg.security_max_file_bytes:
                raise FileSizeLimitError(file_path, size, cfg.security_max_file_bytes)
        
        ext = file_path.lower()
        is_jsonl = ext.endswith(".jsonl") or ext.endswith(".ndjson")
        
        base_meta = {
            "source": os.path.basename(file_path),
            "source_path": os.path.abspath(file_path),
            "source_type": "jsonl" if is_jsonl else "json",
        }
        
        # JSONL: streaming line-by-line
        if is_jsonl:
            chunk_index = 0
            for rec_i, obj in _iter_jsonl_records(file_path):
                units: List[str] = []
                for p, v in _flatten_json(obj):
                    s = _format_kv(p, v)
                    if s:
                        units.append(s)
                
                meta = dict(base_meta)
                meta["record_index"] = rec_i
                
                for ch in self.chunker.chunk_units(
                    units=units,
                    base_meta=meta,
                    joiner="\n",
                    locator=f"jsonl|record={rec_i}",
                    start_chunk_index=chunk_index,
                ):
                    chunk_index = ch["metadata"]["chunk_index"] + 1
                    yield ch
            return
        
        # JSON: try streaming first
        recs = self._iter_json_streaming(file_path)
        if recs is not None:
            chunk_index = 0
            for rec_i, obj in recs:
                units: List[str] = []
                for p, v in _flatten_json(obj):
                    s = _format_kv(p, v)
                    if s:
                        units.append(s)
                
                meta = dict(base_meta)
                meta["record_index"] = rec_i
                
                for ch in self.chunker.chunk_units(
                    units=units,
                    base_meta=meta,
                    joiner="\n",
                    locator=f"json|stream_record={rec_i}",
                    start_chunk_index=chunk_index,
                ):
                    chunk_index = ch["metadata"]["chunk_index"] + 1
                    yield ch
            return
        
        # Fallback: full load
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
        except Exception as e:
            LOGGER.error("JSON Parse Error in %s: %s", file_path, e)
            raise ProcessingError(f"Failed to parse JSON: {e}", {"path": file_path})
        
        chunk_index = 0
        
        # List of dicts -> record-wise
        if isinstance(data, list) and data and all(isinstance(x, dict) for x in data):
            for rec_i, obj in enumerate(data):
                units: List[str] = []
                for p, v in _flatten_json(obj):
                    s = _format_kv(p, v)
                    if s:
                        units.append(s)
                
                meta = dict(base_meta)
                meta["record_index"] = rec_i
                meta["record_count"] = len(data)
                
                for ch in self.chunker.chunk_units(
                    units=units,
                    base_meta=meta,
                    joiner="\n",
                    locator=f"json|record={rec_i}",
                    start_chunk_index=chunk_index,
                ):
                    chunk_index = ch["metadata"]["chunk_index"] + 1
                    yield ch
            return
        
        # Dict with array inside
        if isinstance(data, dict):
            longest_key = None
            max_len = 0
            for k, v in data.items():
                if isinstance(v, list) and len(v) > max_len:
                    max_len = len(v)
                    longest_key = k
            
            if longest_key and max_len > 0:
                for rec_i, obj in enumerate(data[longest_key]):
                    units: List[str] = []
                    for p, v in _flatten_json(obj):
                        s = _format_kv(f"{longest_key}.{p}" if p else longest_key, v)
                        if s:
                            units.append(s)
                    
                    meta = dict(base_meta)
                    meta["record_index"] = rec_i
                    meta["record_key"] = longest_key
                    meta["record_count"] = max_len
                    
                    for ch in self.chunker.chunk_units(
                        units=units,
                        base_meta=meta,
                        joiner="\n",
                        locator=f"json|{longest_key}|record={rec_i}",
                        start_chunk_index=chunk_index,
                    ):
                        chunk_index = ch["metadata"]["chunk_index"] + 1
                        yield ch
                return
        
        # Whole document flatten
        units: List[str] = []
        for p, v in _flatten_json(data):
            s = _format_kv(p, v)
            if s:
                units.append(s)
        
        for ch in self.chunker.chunk_units(
            units=units,
            base_meta=base_meta,
            joiner="\n",
            locator="json|document",
            start_chunk_index=0,
        ):
            yield ch


# Backward compatibility function
def iter_chunks_from_json(
    file_path: str,
    cfg: ChunkConfig = None
) -> Generator[Dict[str, Any], None, None]:
    """
    Iterate over chunks from a JSON or JSONL file.
    
    Args:
        file_path: Path to JSON/JSONL file.
        cfg: Chunk configuration.
        
    Yields:
        Chunk dictionaries.
    """
    chunker = JSONChunker(cfg)
    yield from chunker.chunk_file(file_path)
