"""
CSV chunker for Krira_Chunker.
"""

import os
from typing import Generator, Dict, Any, Iterator, List

from ..config import ChunkConfig
from ..core import FastChunker, LOGGER
from ..exceptions import DependencyNotInstalledError, FileSizeLimitError


def _detect_csv_sep(header_line: str) -> str:
    """Auto-detect CSV separator."""
    if header_line.count("\t") > max(header_line.count(","), header_line.count(";")):
        return "\t"
    return "," if header_line.count(",") >= header_line.count(";") else ";"


class CSVChunker:
    """
    Class-based CSV chunker with lazy polars loading.
    
    Each row is treated as an atomic unit that will not be split
    during chunking (rows form the minimum unit boundary).
    
    Example:
        >>> cfg = ChunkConfig(max_chars=2000)
        >>> chunker = CSVChunker(cfg)
        >>> for chunk in chunker.chunk_file("data.csv"):
        ...     print(chunk["text"][:100])
    """
    
    def __init__(self, cfg: ChunkConfig = None):
        """
        Initialize CSV chunker.
        
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
    
    def _get_polars(self):
        """Lazy import polars."""
        try:
            import polars as pl
            return pl
        except ImportError:
            raise DependencyNotInstalledError("polars", "csv", "CSV processing")
    
    def _csv_row_text_expr(self, pl, cols: List[str]):
        """Build polars expression for row text."""
        parts = []
        for c in cols:
            v = pl.col(c).cast(pl.Utf8)
            non_blank = v.is_not_null() & (v.str.strip_chars() != "")
            parts.append(pl.when(non_blank).then(pl.lit(f"{c}: ") + v).otherwise(None))
        return pl.concat_str(parts, separator=" | ", ignore_nulls=True)
    
    def chunk_file(self, file_path: str) -> Iterator[Dict[str, Any]]:
        """
        Chunk a CSV file.
        
        Each row is treated as an atomic unit. Rows are grouped into
        chunks but never split across chunk boundaries.
        
        Args:
            file_path: Path to CSV file.
            
        Yields:
            Chunk dictionaries with row_start/row_end metadata.
            
        Raises:
            DependencyNotInstalledError: If polars is not installed.
            FileSizeLimitError: If file exceeds size limit.
        """
        pl = self._get_polars()
        cfg = self.cfg
        
        # Security: check file size
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            if size > cfg.security_max_file_bytes:
                raise FileSizeLimitError(file_path, size, cfg.security_max_file_bytes)
        
        base_meta = {
            "source": os.path.basename(file_path),
            "source_path": os.path.abspath(file_path),
            "source_type": "csv",
        }
        
        # Detect separator
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            header = f.readline()
        sep = _detect_csv_sep(header)
        
        row_offset = 0
        chunk_index = 0
        
        try:
            reader = pl.read_csv_batched(
                file_path,
                batch_size=cfg.csv_batch_rows,
                separator=sep,
                ignore_errors=True,
                encoding="utf8-lossy",
                try_parse_dates=True,
            )
            
            def process_df(df) -> Generator[Dict[str, Any], None, None]:
                nonlocal row_offset, chunk_index
                if df.height == 0:
                    return
                
                df = df.with_row_index("row_index", offset=row_offset)
                row_offset += df.height
                
                cols = [c for c in df.columns if c != "row_index"]
                df2 = df.select(
                    pl.col("row_index"),
                    self._csv_row_text_expr(pl, cols).alias("row_text"),
                )
                
                units: List[str] = []
                row_ids: List[int] = []
                for rid, txt in df2.iter_rows():
                    if txt:
                        units.append(str(txt))
                        row_ids.append(int(rid))
                
                for ch in self.chunker.chunk_units(
                    units=units,
                    base_meta=base_meta,
                    joiner="\n",
                    locator="csv",
                    range_key="row",
                    range_vals=row_ids,
                    start_chunk_index=chunk_index,
                ):
                    chunk_index = ch["metadata"]["chunk_index"] + 1
                    yield ch
            
            # Use batched reading
            if hasattr(reader, "next_batches"):
                while True:
                    batches = reader.next_batches(1)
                    if not batches:
                        break
                    yield from process_df(batches[0])
                return
            
            # Fallback for different polars versions
            for df in reader:
                yield from process_df(df)
            return
            
        except Exception as e:
            LOGGER.warning("Batched CSV reading failed, trying lazy scan: %s", e)
            # Fallback: lazy scan
            try:
                q = pl.scan_csv(
                    file_path,
                    separator=sep,
                    ignore_errors=True,
                    try_parse_dates=True,
                    encoding="utf8-lossy"
                )
            except Exception:
                q = pl.scan_csv(
                    file_path,
                    separator=sep,
                    ignore_errors=True,
                    encoding="utf8-lossy"
                )
            
            cols = q.collect_schema().names()
            df = q.select(self._csv_row_text_expr(pl, cols).alias("row_text")).collect(engine="streaming")
            units = [t for t in df["row_text"].to_list() if t]
            
            for ch in self.chunker.chunk_units(
                units=units,
                base_meta=base_meta,
                joiner="\n",
                locator="csv",
                start_chunk_index=0,
            ):
                yield ch


# Backward compatibility function
def iter_chunks_from_csv(
    file_path: str,
    cfg: ChunkConfig = None
) -> Generator[Dict[str, Any], None, None]:
    """
    Iterate over chunks from a CSV file.
    
    Args:
        file_path: Path to CSV file.
        cfg: Chunk configuration.
        
    Yields:
        Chunk dictionaries.
    """
    chunker = CSVChunker(cfg)
    yield from chunker.chunk_file(file_path)
