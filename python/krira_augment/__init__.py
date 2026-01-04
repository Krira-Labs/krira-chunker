"""
Krira Augment - High Performance RAG Framework

A production-grade Python library for document chunking in RAG pipelines,
backed by a highly optimized Rust core for maximum performance.
"""

from .krira_chunker import (
    Pipeline, 
    PipelineConfig, 
    SplitStrategy, 
    PipelineStats, 
    StreamingChunkIterator,
    KriraLoader,
    TextSplitter
)

__all__ = [
    "Pipeline", 
    "PipelineConfig", 
    "SplitStrategy", 
    "PipelineStats", 
    "StreamingChunkIterator",
    "KriraLoader",
    "TextSplitter"
]
