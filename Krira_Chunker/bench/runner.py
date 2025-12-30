"""
Benchmark runner for Krira_Chunker.

Compares chunking performance and quality against LangChain and LlamaIndex baselines.
"""

import os
import sys
import time
import json
import platform
from typing import Dict, Any, List, Optional, Iterator, Tuple
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    library: str
    strategy: str
    file_path: str
    file_type: str
    file_bytes: int
    duration_s: float
    chunk_count: int
    chars_total: int
    chunks_per_s: float
    mb_per_s: float
    avg_chunk_len_chars: float
    empty_chunk_count: int
    very_large_chunk_count: int
    codeblock_break_rate: float
    sentence_break_rate: float
    peak_rss_mb: Optional[float] = None
    chunk_previews: List[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""
    streaming: bool = True
    error: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass 
class BenchmarkReport:
    """Complete benchmark report."""
    timestamp: str
    system_info: Dict[str, str]
    krira_config: Dict[str, Any]
    baseline_config: Dict[str, Any]
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


def get_system_info() -> Dict[str, str]:
    """Get system information."""
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "machine": platform.machine(),
    }


def get_memory_usage() -> Optional[float]:
    """Get current memory usage in MB using psutil if available."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return None
    except Exception:
        return None


def count_codeblock_breaks(text: str) -> bool:
    """
    Check if text has an odd number of code fences (indicates broken code block).
    
    Returns:
        True if code block is broken (odd number of ```).
    """
    fence_count = text.count("```")
    return fence_count % 2 != 0


def check_sentence_ending(text: str) -> bool:
    """
    Check if text ends with a sentence terminator.
    
    Returns:
        True if ends properly with .?! or quote.
    """
    text = text.strip()
    if not text:
        return True
    last_char = text[-1]
    return last_char in ".?!\"'"


def read_file_text(file_path: str) -> str:
    """Read full text from a file (for baseline comparisons)."""
    ext = file_path.lower()
    
    if ext.endswith(".pdf"):
        try:
            import pypdf
            reader = pypdf.PdfReader(file_path)
            texts = []
            for page in reader.pages:
                try:
                    texts.append(page.extract_text() or "")
                except Exception:
                    pass
            return "\n".join(texts)
        except ImportError:
            return ""
        except Exception:
            return ""
    
    if ext.endswith((".txt", ".md", ".csv", ".json", ".jsonl", ".xml")):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""
    
    if ext.endswith(".xlsx"):
        # Skip for baselines - complex format
        return ""
    
    # Default: try reading as text
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def run_krira_benchmark(
    file_path: str,
    cfg: Any,  # ChunkConfig
    file_bytes: int,
) -> BenchmarkResult:
    """
    Run benchmark using Krira_Chunker.
    
    Processes in streaming mode (no full text load).
    """
    from Krira_Chunker import iter_chunks_auto
    
    file_type = Path(file_path).suffix.lower().lstrip(".")
    
    mem_before = get_memory_usage()
    start = time.perf_counter()
    
    chunk_count = 0
    chars_total = 0
    empty_count = 0
    very_large_count = 0
    codeblock_breaks = 0
    sentence_breaks = 0
    previews: List[str] = []
    target_size = cfg.max_chars
    
    try:
        for chunk in iter_chunks_auto(file_path, cfg):
            chunk_text = chunk.get("text", "")
            chunk_count += 1
            chars_total += len(chunk_text)
            
            # Quality metrics
            if not chunk_text.strip():
                empty_count += 1
            
            if len(chunk_text) > target_size * 2:
                very_large_count += 1
            
            if count_codeblock_breaks(chunk_text):
                codeblock_breaks += 1
            
            if not check_sentence_ending(chunk_text):
                sentence_breaks += 1
            
            # Store first 2 previews
            if len(previews) < 2:
                previews.append(chunk_text[:80].replace("\n", " "))
        
        elapsed = time.perf_counter() - start
        mem_after = get_memory_usage()
        
        return BenchmarkResult(
            library="krira_chunker",
            strategy=cfg.chunk_strategy,
            file_path=file_path,
            file_type=file_type,
            file_bytes=file_bytes,
            duration_s=elapsed,
            chunk_count=chunk_count,
            chars_total=chars_total,
            chunks_per_s=chunk_count / elapsed if elapsed > 0 else 0,
            mb_per_s=(file_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0,
            avg_chunk_len_chars=chars_total / chunk_count if chunk_count > 0 else 0,
            empty_chunk_count=empty_count,
            very_large_chunk_count=very_large_count,
            codeblock_break_rate=codeblock_breaks / chunk_count if chunk_count > 0 else 0,
            sentence_break_rate=sentence_breaks / chunk_count if chunk_count > 0 else 0,
            peak_rss_mb=(mem_after - mem_before) if mem_before and mem_after else None,
            chunk_previews=previews,
            streaming=True,
        )
        
    except Exception as e:
        return BenchmarkResult(
            library="krira_chunker",
            strategy=cfg.chunk_strategy,
            file_path=file_path,
            file_type=file_type,
            file_bytes=file_bytes,
            duration_s=0,
            chunk_count=0,
            chars_total=0,
            chunks_per_s=0,
            mb_per_s=0,
            avg_chunk_len_chars=0,
            empty_chunk_count=0,
            very_large_chunk_count=0,
            codeblock_break_rate=0,
            sentence_break_rate=0,
            error=str(e),
        )


def run_langchain_benchmark(
    file_path: str,
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    file_bytes: int,
) -> BenchmarkResult:
    """
    Run benchmark using LangChain RecursiveCharacterTextSplitter.
    
    Note: Requires full text (not streaming).
    """
    file_type = Path(file_path).suffix.lower().lstrip(".")
    
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        return BenchmarkResult(
            library="langchain",
            strategy="recursive_character",
            file_path=file_path,
            file_type=file_type,
            file_bytes=file_bytes,
            duration_s=0,
            chunk_count=0,
            chars_total=0,
            chunks_per_s=0,
            mb_per_s=0,
            avg_chunk_len_chars=0,
            empty_chunk_count=0,
            very_large_chunk_count=0,
            codeblock_break_rate=0,
            sentence_break_rate=0,
            skipped=True,
            skip_reason="langchain not installed",
            streaming=False,
        )
    
    if not text:
        return BenchmarkResult(
            library="langchain",
            strategy="recursive_character",
            file_path=file_path,
            file_type=file_type,
            file_bytes=file_bytes,
            duration_s=0,
            chunk_count=0,
            chars_total=0,
            chunks_per_s=0,
            mb_per_s=0,
            avg_chunk_len_chars=0,
            empty_chunk_count=0,
            very_large_chunk_count=0,
            codeblock_break_rate=0,
            sentence_break_rate=0,
            skipped=True,
            skip_reason=f"Cannot extract text from {file_type}",
            streaming=False,
        )
    
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        mem_before = get_memory_usage()
        start = time.perf_counter()
        
        chunks = splitter.split_text(text)
        
        elapsed = time.perf_counter() - start
        mem_after = get_memory_usage()
        
        # Quality metrics
        chunk_count = len(chunks)
        chars_total = sum(len(c) for c in chunks)
        empty_count = sum(1 for c in chunks if not c.strip())
        very_large_count = sum(1 for c in chunks if len(c) > chunk_size * 2)
        codeblock_breaks = sum(1 for c in chunks if count_codeblock_breaks(c))
        sentence_breaks = sum(1 for c in chunks if not check_sentence_ending(c))
        previews = [c[:80].replace("\n", " ") for c in chunks[:2]]
        
        return BenchmarkResult(
            library="langchain",
            strategy="recursive_character",
            file_path=file_path,
            file_type=file_type,
            file_bytes=file_bytes,
            duration_s=elapsed,
            chunk_count=chunk_count,
            chars_total=chars_total,
            chunks_per_s=chunk_count / elapsed if elapsed > 0 else 0,
            mb_per_s=(file_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0,
            avg_chunk_len_chars=chars_total / chunk_count if chunk_count > 0 else 0,
            empty_chunk_count=empty_count,
            very_large_chunk_count=very_large_count,
            codeblock_break_rate=codeblock_breaks / chunk_count if chunk_count > 0 else 0,
            sentence_break_rate=sentence_breaks / chunk_count if chunk_count > 0 else 0,
            peak_rss_mb=(mem_after - mem_before) if mem_before and mem_after else None,
            chunk_previews=previews,
            streaming=False,
        )
        
    except Exception as e:
        return BenchmarkResult(
            library="langchain",
            strategy="recursive_character",
            file_path=file_path,
            file_type=file_type,
            file_bytes=file_bytes,
            duration_s=0,
            chunk_count=0,
            chars_total=0,
            chunks_per_s=0,
            mb_per_s=0,
            avg_chunk_len_chars=0,
            empty_chunk_count=0,
            very_large_chunk_count=0,
            codeblock_break_rate=0,
            sentence_break_rate=0,
            error=str(e),
            streaming=False,
        )


def run_llamaindex_sentence_benchmark(
    file_path: str,
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    file_bytes: int,
) -> BenchmarkResult:
    """
    Run benchmark using LlamaIndex SentenceSplitter.
    
    Note: Requires full text (not streaming).
    """
    file_type = Path(file_path).suffix.lower().lstrip(".")
    
    # Try different import paths for LlamaIndex versions
    SentenceSplitter = None
    try:
        from llama_index.core.node_parser import SentenceSplitter
    except ImportError:
        try:
            from llama_index.node_parser import SentenceSplitter
        except ImportError:
            pass
    
    if SentenceSplitter is None:
        return BenchmarkResult(
            library="llama_index",
            strategy="sentence_splitter",
            file_path=file_path,
            file_type=file_type,
            file_bytes=file_bytes,
            duration_s=0,
            chunk_count=0,
            chars_total=0,
            chunks_per_s=0,
            mb_per_s=0,
            avg_chunk_len_chars=0,
            empty_chunk_count=0,
            very_large_chunk_count=0,
            codeblock_break_rate=0,
            sentence_break_rate=0,
            skipped=True,
            skip_reason="llama_index not installed",
            streaming=False,
        )
    
    if not text:
        return BenchmarkResult(
            library="llama_index",
            strategy="sentence_splitter",
            file_path=file_path,
            file_type=file_type,
            file_bytes=file_bytes,
            duration_s=0,
            chunk_count=0,
            chars_total=0,
            chunks_per_s=0,
            mb_per_s=0,
            avg_chunk_len_chars=0,
            empty_chunk_count=0,
            very_large_chunk_count=0,
            codeblock_break_rate=0,
            sentence_break_rate=0,
            skipped=True,
            skip_reason=f"Cannot extract text from {file_type}",
            streaming=False,
        )
    
    try:
        splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        mem_before = get_memory_usage()
        start = time.perf_counter()
        
        # LlamaIndex split_text method
        try:
            chunks = splitter.split_text(text)
        except AttributeError:
            # Fallback for different API
            from llama_index.core import Document
            doc = Document(text=text)
            nodes = splitter.get_nodes_from_documents([doc])
            chunks = [n.get_content() for n in nodes]
        
        elapsed = time.perf_counter() - start
        mem_after = get_memory_usage()
        
        # Quality metrics
        chunk_count = len(chunks)
        chars_total = sum(len(c) for c in chunks)
        empty_count = sum(1 for c in chunks if not c.strip())
        very_large_count = sum(1 for c in chunks if len(c) > chunk_size * 2)
        codeblock_breaks = sum(1 for c in chunks if count_codeblock_breaks(c))
        sentence_breaks = sum(1 for c in chunks if not check_sentence_ending(c))
        previews = [c[:80].replace("\n", " ") for c in chunks[:2]]
        
        return BenchmarkResult(
            library="llama_index",
            strategy="sentence_splitter",
            file_path=file_path,
            file_type=file_type,
            file_bytes=file_bytes,
            duration_s=elapsed,
            chunk_count=chunk_count,
            chars_total=chars_total,
            chunks_per_s=chunk_count / elapsed if elapsed > 0 else 0,
            mb_per_s=(file_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0,
            avg_chunk_len_chars=chars_total / chunk_count if chunk_count > 0 else 0,
            empty_chunk_count=empty_count,
            very_large_chunk_count=very_large_count,
            codeblock_break_rate=codeblock_breaks / chunk_count if chunk_count > 0 else 0,
            sentence_break_rate=sentence_breaks / chunk_count if chunk_count > 0 else 0,
            peak_rss_mb=(mem_after - mem_before) if mem_before and mem_after else None,
            chunk_previews=previews,
            streaming=False,
        )
        
    except Exception as e:
        return BenchmarkResult(
            library="llama_index",
            strategy="sentence_splitter",
            file_path=file_path,
            file_type=file_type,
            file_bytes=file_bytes,
            duration_s=0,
            chunk_count=0,
            chars_total=0,
            chunks_per_s=0,
            mb_per_s=0,
            avg_chunk_len_chars=0,
            empty_chunk_count=0,
            very_large_chunk_count=0,
            codeblock_break_rate=0,
            sentence_break_rate=0,
            error=str(e),
            streaming=False,
        )


def run_llamaindex_token_benchmark(
    file_path: str,
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    file_bytes: int,
) -> BenchmarkResult:
    """
    Run benchmark using LlamaIndex TokenTextSplitter.
    
    Note: Requires full text (not streaming).
    """
    file_type = Path(file_path).suffix.lower().lstrip(".")
    
    # Try different import paths
    TokenTextSplitter = None
    try:
        from llama_index.core.node_parser import TokenTextSplitter
    except ImportError:
        try:
            from llama_index.node_parser import TokenTextSplitter
        except ImportError:
            pass
    
    if TokenTextSplitter is None:
        return BenchmarkResult(
            library="llama_index",
            strategy="token_text_splitter",
            file_path=file_path,
            file_type=file_type,
            file_bytes=file_bytes,
            duration_s=0,
            chunk_count=0,
            chars_total=0,
            chunks_per_s=0,
            mb_per_s=0,
            avg_chunk_len_chars=0,
            empty_chunk_count=0,
            very_large_chunk_count=0,
            codeblock_break_rate=0,
            sentence_break_rate=0,
            skipped=True,
            skip_reason="llama_index not installed",
            streaming=False,
        )
    
    if not text:
        return BenchmarkResult(
            library="llama_index",
            strategy="token_text_splitter",
            file_path=file_path,
            file_type=file_type,
            file_bytes=file_bytes,
            duration_s=0,
            chunk_count=0,
            chars_total=0,
            chunks_per_s=0,
            mb_per_s=0,
            avg_chunk_len_chars=0,
            empty_chunk_count=0,
            very_large_chunk_count=0,
            codeblock_break_rate=0,
            sentence_break_rate=0,
            skipped=True,
            skip_reason=f"Cannot extract text from {file_type}",
            streaming=False,
        )
    
    try:
        # TokenTextSplitter uses token counts, approximate conversion
        token_chunk_size = chunk_size // 4  # ~4 chars per token
        token_overlap = chunk_overlap // 4
        
        splitter = TokenTextSplitter(
            chunk_size=token_chunk_size,
            chunk_overlap=token_overlap,
        )
        
        mem_before = get_memory_usage()
        start = time.perf_counter()
        
        try:
            chunks = splitter.split_text(text)
        except AttributeError:
            from llama_index.core import Document
            doc = Document(text=text)
            nodes = splitter.get_nodes_from_documents([doc])
            chunks = [n.get_content() for n in nodes]
        
        elapsed = time.perf_counter() - start
        mem_after = get_memory_usage()
        
        # Quality metrics
        chunk_count = len(chunks)
        chars_total = sum(len(c) for c in chunks)
        empty_count = sum(1 for c in chunks if not c.strip())
        very_large_count = sum(1 for c in chunks if len(c) > chunk_size * 2)
        codeblock_breaks = sum(1 for c in chunks if count_codeblock_breaks(c))
        sentence_breaks = sum(1 for c in chunks if not check_sentence_ending(c))
        previews = [c[:80].replace("\n", " ") for c in chunks[:2]]
        
        return BenchmarkResult(
            library="llama_index",
            strategy="token_text_splitter",
            file_path=file_path,
            file_type=file_type,
            file_bytes=file_bytes,
            duration_s=elapsed,
            chunk_count=chunk_count,
            chars_total=chars_total,
            chunks_per_s=chunk_count / elapsed if elapsed > 0 else 0,
            mb_per_s=(file_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0,
            avg_chunk_len_chars=chars_total / chunk_count if chunk_count > 0 else 0,
            empty_chunk_count=empty_count,
            very_large_chunk_count=very_large_count,
            codeblock_break_rate=codeblock_breaks / chunk_count if chunk_count > 0 else 0,
            sentence_break_rate=sentence_breaks / chunk_count if chunk_count > 0 else 0,
            peak_rss_mb=(mem_after - mem_before) if mem_before and mem_after else None,
            chunk_previews=previews,
            streaming=False,
        )
        
    except Exception as e:
        return BenchmarkResult(
            library="llama_index",
            strategy="token_text_splitter",
            file_path=file_path,
            file_type=file_type,
            file_bytes=file_bytes,
            duration_s=0,
            chunk_count=0,
            chars_total=0,
            chunks_per_s=0,
            mb_per_s=0,
            avg_chunk_len_chars=0,
            empty_chunk_count=0,
            very_large_chunk_count=0,
            codeblock_break_rate=0,
            sentence_break_rate=0,
            error=str(e),
            streaming=False,
        )


def collect_corpus_files(corpus_path: str) -> List[Tuple[str, int]]:
    """
    Collect files from corpus path.
    
    Returns:
        List of (file_path, file_bytes) tuples.
    """
    extensions = {".pdf", ".md", ".txt", ".csv", ".jsonl", ".json", ".xml", ".xlsx"}
    files = []
    
    corpus = Path(corpus_path)
    
    if corpus.is_file():
        if corpus.suffix.lower() in extensions:
            files.append((str(corpus), corpus.stat().st_size))
    elif corpus.is_dir():
        for f in corpus.rglob("*"):
            if f.is_file() and f.suffix.lower() in extensions:
                files.append((str(f), f.stat().st_size))
    
    return files


def collect_urls(urls_file: str) -> List[str]:
    """Collect URLs from a urls.txt file."""
    urls = []
    try:
        with open(urls_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and line.startswith(("http://", "https://")):
                    urls.append(line)
    except Exception:
        pass
    return urls


def run_full_benchmark(
    corpus_path: str,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
    verbose: bool = False,
) -> BenchmarkReport:
    """
    Run complete benchmark suite.
    
    Args:
        corpus_path: Path to corpus directory or file.
        chunk_size: Target chunk size in chars.
        chunk_overlap: Overlap in chars.
        verbose: Print progress.
        
    Returns:
        BenchmarkReport with all results.
    """
    from Krira_Chunker import ChunkConfig
    
    # Create Krira config
    cfg = ChunkConfig(
        max_chars=chunk_size,
        overlap_chars=chunk_overlap,
        chunk_strategy="hybrid",
    )
    
    # Collect files
    files = collect_corpus_files(corpus_path)
    
    # Check for urls.txt
    urls_file = os.path.join(corpus_path, "urls.txt") if os.path.isdir(corpus_path) else None
    urls = collect_urls(urls_file) if urls_file and os.path.exists(urls_file) else []
    
    results: List[Dict[str, Any]] = []
    
    if verbose:
        print(f"Found {len(files)} files and {len(urls)} URLs to benchmark")
    
    for file_path, file_bytes in files:
        if verbose:
            print(f"\nBenchmarking: {file_path} ({file_bytes:,} bytes)")
        
        # Run Krira benchmark (streaming)
        krira_result = run_krira_benchmark(file_path, cfg, file_bytes)
        results.append(krira_result.to_dict())
        
        if verbose:
            print(f"  Krira: {krira_result.chunk_count} chunks in {krira_result.duration_s:.3f}s")
        
        # Read text for baselines (not streaming)
        text = read_file_text(file_path)
        
        # LangChain baseline
        lc_result = run_langchain_benchmark(file_path, text, chunk_size, chunk_overlap, file_bytes)
        results.append(lc_result.to_dict())
        
        if verbose:
            if lc_result.skipped:
                print(f"  LangChain: skipped ({lc_result.skip_reason})")
            else:
                print(f"  LangChain: {lc_result.chunk_count} chunks in {lc_result.duration_s:.3f}s")
        
        # LlamaIndex SentenceSplitter
        li_sent_result = run_llamaindex_sentence_benchmark(file_path, text, chunk_size, chunk_overlap, file_bytes)
        results.append(li_sent_result.to_dict())
        
        if verbose:
            if li_sent_result.skipped:
                print(f"  LlamaIndex Sentence: skipped ({li_sent_result.skip_reason})")
            else:
                print(f"  LlamaIndex Sentence: {li_sent_result.chunk_count} chunks in {li_sent_result.duration_s:.3f}s")
        
        # LlamaIndex TokenTextSplitter
        li_tok_result = run_llamaindex_token_benchmark(file_path, text, chunk_size, chunk_overlap, file_bytes)
        results.append(li_tok_result.to_dict())
        
        if verbose:
            if li_tok_result.skipped:
                print(f"  LlamaIndex Token: skipped ({li_tok_result.skip_reason})")
            else:
                print(f"  LlamaIndex Token: {li_tok_result.chunk_count} chunks in {li_tok_result.duration_s:.3f}s")
    
    # Process URLs
    for url in urls:
        if verbose:
            print(f"\nBenchmarking URL: {url}")
        
        # Krira URL chunking
        krira_result = run_krira_benchmark(url, cfg, 0)
        results.append(krira_result.to_dict())
        
        if verbose:
            print(f"  Krira: {krira_result.chunk_count} chunks in {krira_result.duration_s:.3f}s")
    
    # Compute summary
    krira_results = [r for r in results if r["library"] == "krira_chunker" and not r.get("skipped")]
    lc_results = [r for r in results if r["library"] == "langchain" and not r.get("skipped")]
    li_results = [r for r in results if r["library"] == "llama_index" and not r.get("skipped")]
    
    summary = {
        "total_files": len(files),
        "total_urls": len(urls),
        "krira_runs": len(krira_results),
        "langchain_runs": len(lc_results),
        "llamaindex_runs": len(li_results),
    }
    
    if krira_results:
        summary["krira_avg_chunks_per_s"] = sum(r["chunks_per_s"] for r in krira_results) / len(krira_results)
        summary["krira_avg_mb_per_s"] = sum(r["mb_per_s"] for r in krira_results) / len(krira_results)
    
    if lc_results:
        summary["langchain_avg_chunks_per_s"] = sum(r["chunks_per_s"] for r in lc_results) / len(lc_results)
    
    if li_results:
        summary["llamaindex_avg_chunks_per_s"] = sum(r["chunks_per_s"] for r in li_results) / len(li_results)
    
    return BenchmarkReport(
        timestamp=datetime.utcnow().isoformat() + "Z",
        system_info=get_system_info(),
        krira_config=cfg.to_dict() if hasattr(cfg, "to_dict") else {
            "max_chars": cfg.max_chars,
            "overlap_chars": cfg.overlap_chars,
            "chunk_strategy": cfg.chunk_strategy,
        },
        baseline_config={
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
        results=results,
        summary=summary,
    )
