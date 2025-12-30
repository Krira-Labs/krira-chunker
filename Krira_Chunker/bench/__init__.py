"""
Benchmark module for Krira_Chunker.

Usage:
    python -m Krira_Chunker.bench --corpus ./data --report results.json
"""

from .runner import (
    run_full_benchmark,
    run_krira_benchmark,
    run_langchain_benchmark,
    run_llamaindex_sentence_benchmark,
    run_llamaindex_token_benchmark,
    BenchmarkResult,
    BenchmarkReport,
    collect_corpus_files,
    get_system_info,
    get_memory_usage,
)
from .__main__ import main

__all__ = [
    "main",
    "run_full_benchmark",
    "run_krira_benchmark",
    "run_langchain_benchmark",
    "run_llamaindex_sentence_benchmark",
    "run_llamaindex_token_benchmark",
    "BenchmarkResult",
    "BenchmarkReport",
    "collect_corpus_files",
    "get_system_info",
    "get_memory_usage",
]
