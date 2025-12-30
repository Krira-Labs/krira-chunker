"""
Core chunking logic for Krira_Chunker.

Contains:
- FastChunker: Original fast chunker (fixed/prose mode)
- HybridBoundaryChunker: Advanced boundary-aware chunker
- Utility functions: clean_text, stable_id
"""

import re
import hashlib
import logging
import threading
import queue
from typing import List, Dict, Any, Optional, Generator, Tuple, Iterable
from .config import ChunkConfig

LOGGER = logging.getLogger("krira_chunker")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.WARNING)


# =============================================================================
# Regex Patterns
# =============================================================================

# Optimized Single-Pass Boundary Scanner
_BOUNDARY_SCANNER = re.compile(
    r"(?P<heading>^(?:#{1,6})\s+.+$)|"
    r"(?P<paragraph>\n\s*\n)|"
    r"(?P<sentence>(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+)|"
    r"(?P<line>\n)",
    re.MULTILINE
)

_SENT_SPLIT_RE = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+")
_MULTI_WS_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_SQL_STMT_SPLIT_RE = re.compile(r";\s*(?:\r?\n)+|;\s+$", re.MULTILINE)
_CODE_BLOCK_RE = re.compile(
    r"^(?:\s*(?:def|class|function)\s+|\s*CREATE\s+(?:TABLE|VIEW|FUNCTION|PROCEDURE)\b)",
    re.IGNORECASE | re.MULTILINE,
)

# Markdown patterns
_MD_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_MD_TABLE_LINE_RE = re.compile(r"^\|.*\|$", re.MULTILINE)
_MD_LIST_ITEM_RE = re.compile(r"^(\s*[-*+]|\s*\d+\.)\s+", re.MULTILINE)


# =============================================================================
# Utility Functions
# =============================================================================

def clean_text(s: str) -> str:
    """Clean and normalize text."""
    if not s:
        return ""
    s = s.replace("\x00", " ")
    s = _MULTI_WS_RE.sub(" ", s)
    s = _MULTI_NEWLINE_RE.sub("\n\n", s)
    return s.strip()


def stable_id(source: str, locator: str, chunk_index: int, text: str) -> str:
    """Generate a stable, deterministic chunk ID."""
    raw = f"{source}|{locator}|{chunk_index}|{text}"
    return hashlib.md5(raw.encode("utf-8", errors="ignore")).hexdigest()


def prefetch_generator(gen: Iterable[Any], buffer_size: int = 5) -> Generator[Any, None, None]:
    """
    Threaded prefetcher to improve IO throughput (Producer-Consumer).
    
    Reads from the input generator in a background thread and queues items.
    """
    # Simply wrap the generator in a production loop
    q = queue.Queue(maxsize=buffer_size)
    sentinel = object()

    def producer():
        try:
            for item in gen:
                q.put(item)
        except Exception as e:
            LOGGER.error(f"Prefetch producer error: {e}")
        finally:
            q.put(sentinel)

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    while True:
        item = q.get()
        if item is sentinel:
            return
        yield item


# =============================================================================
# Protected Block Detection
# =============================================================================

def _find_protected_ranges(text: str, cfg: ChunkConfig) -> List[Tuple[int, int]]:
    """Find ranges in text that should not be split."""
    ranges = []
    
    # Code fences
    if cfg.preserve_code_blocks:
        for m in _MD_CODE_FENCE_RE.finditer(text):
            ranges.append((m.start(), m.end()))
    
    # Tables
    if cfg.preserve_tables:
        lines = text.split("\n")
        i = 0
        pos = 0
        while i < len(lines):
            line = lines[i]
            if _MD_TABLE_LINE_RE.match(line):
                start = pos
                while i < len(lines) and _MD_TABLE_LINE_RE.match(lines[i]):
                    pos += len(lines[i]) + 1
                    i += 1
                ranges.append((start, pos - 1))
            else:
                pos += len(line) + 1
                i += 1
    
    return ranges


def _is_in_protected_range(pos: int, ranges: List[Tuple[int, int]]) -> bool:
    """Check if position is inside any protected range."""
    for start, end in ranges:
        if start <= pos < end:
            return True
    return False


# =============================================================================
# Boundary Detection
# =============================================================================

class BoundaryPoint:
    """Represents a potential chunk boundary."""
    __slots__ = ("pos", "type", "priority")
    
    def __init__(self, pos: int, boundary_type: str, priority: int):
        self.pos = pos
        self.type = boundary_type
        self.priority = priority

def _find_boundaries(text: str, cfg: ChunkConfig) -> List[BoundaryPoint]:
    """
    Find all potential chunk boundaries using optimized single-pass scanning.
    """
    boundaries = []
    protected = _find_protected_ranges(text, cfg)
    
    # Priority map
    priorities = {
        "heading": 1,
        "paragraph": 2,
        "sentence": 3,
        "line": 4
    }
    
    # Single pass scan
    for m in _BOUNDARY_SCANNER.finditer(text):
        kind = m.lastgroup
        if not kind:
            continue
            
        pos = m.start() if kind == "heading" else m.end()
        
        if not _is_in_protected_range(pos, protected):
            boundaries.append(BoundaryPoint(pos, kind, priorities[kind]))
            
    return boundaries


# =============================================================================
# FastChunker (Original Implementation)
# =============================================================================

class FastChunker:
    """Fast chunker with support for prose, sql, code, and lines modes."""

    def __init__(self, cfg: ChunkConfig):
        self.cfg = cfg
        self._tok = None
        if cfg.use_tokens:
            try:
                import tiktoken
                self._tok = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._tok = None
                LOGGER.warning("use_tokens=True but tiktoken not available; falling back to char-based.")

    def _count_tokens(self, s: str) -> int:
        if not self._tok:
            return (len(s) + 3) // 4
        return len(self._tok.encode_ordinary(s))

    def _measure_size(self, s: str) -> int:
        if self.cfg.use_tokens and self._tok:
            return self._count_tokens(s)
        return len(s)

    def _split_units(self, text: str, mode: str) -> List[str]:
        text = text or ""
        if not text.strip():
            return []

        if mode == "sql":
            stmts = [x.strip() for x in re.split(_SQL_STMT_SPLIT_RE, text) if x and x.strip()]
            if not stmts:
                return [text.strip()]
            return [s if s.endswith(";") else (s + ";") for s in stmts]

        if mode == "code":
            blocks: List[str] = []
            cur: List[str] = []
            for line in text.splitlines():
                if _CODE_BLOCK_RE.match(line) and cur:
                    blocks.append("\n".join(cur).strip())
                    cur = [line]
                else:
                    cur.append(line)
            if cur:
                blocks.append("\n".join(cur).strip())

            units: List[str] = []
            for b in blocks:
                parts = [p.strip() for p in re.split(r"\n\s*\n+", b) if p.strip()]
                units.extend(parts if parts else [b.strip()])
            return [u for u in units if u]

        if mode == "lines":
            return [ln.strip() for ln in text.splitlines() if ln.strip()]

        paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
        units: List[str] = []
        for p in paras:
            sents = [s.strip() for s in re.split(_SENT_SPLIT_RE, p) if s.strip()]
            units.extend(sents if sents else [p])
        return units

    def _hard_split_text(self, text: str, max_len: int, overlap: int) -> List[str]:
        text = text.strip()
        if not text:
            return []
        if max_len <= 0:
            return [text]
        overlap = max(0, overlap)
        overlap = min(overlap, max_len - 1) if max_len > 1 else 0

        chunks: List[str] = []
        i = 0
        n = len(text)
        step = max_len - overlap if (max_len - overlap) > 0 else max_len
        while i < n:
            chunk = text[i: i + max_len].strip()
            if chunk:
                chunks.append(chunk)
            i += step
        return chunks

    def chunk_units(
        self,
        units: Iterable[str],
        base_meta: Dict[str, Any],
        joiner: str,
        locator: str,
        range_key: Optional[str] = None,
        range_vals: Optional[List[int]] = None,
        start_chunk_index: int = 0,
    ) -> Generator[Dict[str, Any], None, None]:
        """Chunk a list/iterable of text units."""
        cfg = self.cfg
        chunk_index = start_chunk_index
        config_hash = cfg.config_hash()

        max_size = cfg.max_tokens if cfg.use_tokens else cfg.max_chars
        overlap_size = cfg.overlap_tokens if cfg.use_tokens else cfg.overlap_chars

        def fits(curr: str, add: str) -> bool:
            if not curr:
                return True
            cand = curr + joiner + add
            return self._measure_size(cand) <= max_size

        buf: List[str] = []
        buf_range: List[int] = []

        def finalize(boundary_type: str = "natural") -> Optional[Dict[str, Any]]:
            nonlocal chunk_index
            if not buf:
                return None
            content = clean_text(joiner.join(buf))
            if not content or len(content) < cfg.min_chars:
                return None

            meta = dict(base_meta)
            meta["chunk_index"] = chunk_index
            meta["locator"] = locator
            meta["config_hash"] = config_hash
            meta["boundary_type"] = boundary_type

            if range_key and buf_range:
                meta[f"{range_key}_start"] = min(buf_range)
                meta[f"{range_key}_end"] = max(buf_range)

            out = {
                "id": stable_id(str(meta.get("source", "")), locator, chunk_index, content),
                "text": content,
                "metadata": meta,
            }
            chunk_index += 1
            return out

        def make_overlap() -> Tuple[List[str], List[int]]:
            if not buf:
                return [], []
            target = overlap_size
            acc: List[str] = []
            acc_r: List[int] = []
            total = 0
            for i in range(len(buf) - 1, -1, -1):
                t = buf[i]
                size = self._measure_size(t)
                if total + size <= target:
                    acc.append(t)
                    if buf_range:
                        acc_r.append(buf_range[i])
                    total += size
                else:
                    break
            acc.reverse()
            acc_r.reverse()
            return acc, acc_r

        # Use prefetcher if it's a generator (not a list) to parallelize consumption
        if not isinstance(units, list):
             units = prefetch_generator(units)

        for idx, u in enumerate(units):
            u = clean_text(u)
            if not u:
                continue

            # Oversize unit => hard split
            if self._measure_size(u) > max_size:
                done = finalize("natural")
                if done:
                    yield done
                buf, buf_range = [], []
                for piece in self._hard_split_text(u, cfg.max_chars, cfg.overlap_chars):
                    if len(piece) >= cfg.min_chars:
                        meta = dict(base_meta)
                        meta["chunk_index"] = chunk_index
                        meta["locator"] = locator + "|hard_split"
                        meta["config_hash"] = config_hash
                        meta["boundary_type"] = "hard"
                        yield {
                            "id": stable_id(str(meta.get("source", "")), meta["locator"], chunk_index, piece),
                            "text": piece,
                            "metadata": meta,
                        }
                        chunk_index += 1
                continue

            if buf:
                prospective = buf[-1] if len(buf) == 1 else joiner.join(buf)
                if not fits(prospective, u):
                    done = finalize("natural")
                    if done:
                        yield done
                    buf, buf_range = make_overlap()

            buf.append(u)
            if range_vals is not None and idx < len(range_vals):
                buf_range.append(range_vals[idx])

        done = finalize("natural")
        if done:
            yield done

    def chunk_text(
        self,
        text: str,
        base_meta: Dict[str, Any],
        mode: str,
        locator: str,
        joiner: str = " ",
        start_chunk_index: int = 0,
    ) -> Generator[Dict[str, Any], None, None]:
        units = self._split_units(text, mode=mode)
        yield from self.chunk_units(
            units=units,
            base_meta=base_meta,
            joiner=joiner,
            locator=locator,
            start_chunk_index=start_chunk_index,
        )


# =============================================================================
# HybridBoundaryChunker
# =============================================================================

class HybridBoundaryChunker:
    """Advanced boundary-aware chunker that respects structure."""

    def __init__(self, cfg: ChunkConfig):
        self.cfg = cfg
        self._tok = None
        if cfg.use_tokens:
            try:
                import tiktoken
                self._tok = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._tok = None
                LOGGER.warning("use_tokens=True but tiktoken not available; falling back to char-based.")

    def _measure_size(self, s: str) -> int:
        if self.cfg.use_tokens and self._tok:
            return len(self._tok.encode_ordinary(s))
        return len(s)

    def _find_best_split_point(
        self,
        text: str,
        target_size: int,
        boundaries: List[BoundaryPoint],
        protected: List[Tuple[int, int]],
    ) -> Tuple[int, str]:
        candidates = []
        for b in boundaries:
            if b.pos <= 0:
                continue
            # Approximate size check using position (optimization)
            # If measuring tokens, this is inaccurate, but for chars it's exact
            if not self.cfg.use_tokens:
                size = b.pos
            else:
                size = self._measure_size(text[:b.pos])
                
            if size <= target_size:
                distance_score = target_size - size
                candidates.append((b.pos, b.type, b.priority, distance_score))
        
        if not candidates:
            return -1, "hard"
        
        candidates.sort(key=lambda x: (x[2], x[3]))
        best = candidates[0]
        return best[0], best[1]

    def chunk_text(
        self,
        text: str,
        base_meta: Dict[str, Any],
        locator: str,
        start_chunk_index: int = 0,
    ) -> Generator[Dict[str, Any], None, None]:
        cfg = self.cfg
        chunk_index = start_chunk_index
        config_hash = cfg.config_hash()
        
        max_size = cfg.max_tokens if cfg.use_tokens else cfg.max_chars
        overlap_size = cfg.overlap_tokens if cfg.use_tokens else cfg.overlap_chars
        
        text = clean_text(text)
        if not text:
            return
            
        # Optimization: Early exit if text fits
        if self._measure_size(text) <= max_size:
            meta = dict(base_meta)
            meta.update({
                "chunk_index": chunk_index,
                "locator": locator,
                "config_hash": config_hash,
                "boundary_type": "natural"
            })
            yield {
                "id": stable_id(str(meta.get("source", "")), locator, chunk_index, text),
                "text": text,
                "metadata": meta,
            }
            return

        boundaries = _find_boundaries(text, cfg)
        protected = _find_protected_ranges(text, cfg)
        
        pos = 0
        overlap_text = ""
        
        while pos < len(text):
            remaining = text[pos:]
            remaining_size = self._measure_size(remaining)
            
            if remaining_size <= max_size:
                chunk_text = (overlap_text + " " + remaining).strip() if overlap_text else remaining
                if len(chunk_text) >= cfg.min_chars:
                    meta = dict(base_meta)
                    meta["chunk_index"] = chunk_index
                    meta["locator"] = locator
                    meta["config_hash"] = config_hash
                    meta["boundary_type"] = "natural"
                    yield {
                        "id": stable_id(str(meta.get("source", "")), locator, chunk_index, chunk_text),
                        "text": chunk_text,
                        "metadata": meta,
                    }
                break
            
            effective_max = max_size - self._measure_size(overlap_text) - 1 if overlap_text else max_size
            
            rel_boundaries = [
                BoundaryPoint(b.pos - pos, b.type, b.priority)
                for b in boundaries
                if b.pos > pos
            ]
            rel_protected = [(s - pos, e - pos) for s, e in protected if e > pos]
            
            split_pos, boundary_type = self._find_best_split_point(
                remaining, effective_max, rel_boundaries, rel_protected
            )
            
            # Code Block Fix Strategy:
            # If optimal split is in protected range OR no split found (-1), try specialized protected handling.
            if split_pos == -1 or _is_in_protected_range(pos + split_pos, protected):
                found_safe_split = False
                
                # Check if we are inside a protected block (or overlap one)
                for start_p, end_p in protected:
                    if start_p <= pos and end_p > pos:
                        # Current position is within a protected block.
                        # It's an oversized block we must split.
                        
                        # Calculate where the block ends relative to current pos
                        rel_block_end = end_p - pos
                        
                        # We must split somewhere in [0, effective_max]
                        # Prefer newline split
                        search_limit = min(effective_max, rel_block_end)
                        last_newline = remaining.rfind('\n', 0, search_limit)
                        
                        if last_newline > 0:
                            split_pos = last_newline + 1 # Include newline in chunk
                            boundary_type = "line"
                            found_safe_split = True
                        else:
                            # No newline? 
                            # If we can't find newline, force split at max_size (hard split)
                            # but mark it as 'line' or 'hard'
                            split_pos = min(effective_max, rel_block_end)
                            boundary_type = "hard"
                            found_safe_split = True
                        break
                
                # If we weren't inside a block but split_pos landed in one ahead...
                if not found_safe_split:
                     # e.g. split_pos falls in protected range [pos+100, pos+200]
                     # We should split BEFORE that range starts.
                     for start_p, end_p in protected:
                         if start_p > pos and start_p < pos + split_pos:
                             rel_start = start_p - pos
                             # Back off to the start of the protected block (split right before it)
                             split_pos = rel_start
                             boundary_type = "paragraph"
                             found_safe_split = True
                             break
                
                if not found_safe_split:
                    split_pos = min(effective_max, len(remaining))
                    boundary_type = "hard"

            chunk_content = remaining[:split_pos].strip()
            chunk_text = (overlap_text + " " + chunk_content).strip() if overlap_text else chunk_content
            
            if len(chunk_text) >= cfg.min_chars:
                meta = dict(base_meta)
                meta["chunk_index"] = chunk_index
                meta["locator"] = locator
                meta["config_hash"] = config_hash
                meta["boundary_type"] = boundary_type
                yield {
                    "id": stable_id(str(meta.get("source", "")), locator, chunk_index, chunk_text),
                    "text": chunk_text,
                    "metadata": meta,
                }
                chunk_index += 1
            
            if overlap_size > 0:
                overlap_start = max(0, split_pos - overlap_size)
                overlap_text = remaining[overlap_start:split_pos].strip()
            else:
                overlap_text = ""
            
            pos += split_pos

def get_chunker(cfg: ChunkConfig) -> FastChunker:
    if cfg.chunk_strategy == "hybrid":
        return FastChunker(cfg)
    return FastChunker(cfg)

def get_hybrid_chunker(cfg: ChunkConfig) -> HybridBoundaryChunker:
    return HybridBoundaryChunker(cfg)
