"""
Smoke tests for the benchmark module.

These tests verify that the benchmark runner works correctly
without requiring LangChain or LlamaIndex to be installed.
"""

import os
import json
import tempfile
import pytest
from pathlib import Path


class TestBenchmarkRunner:
    """Smoke tests for benchmark runner."""
    
    @pytest.fixture
    def temp_corpus(self):
        """Create a temporary corpus with test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple text file
            txt_file = os.path.join(tmpdir, "test.txt")
            with open(txt_file, "w") as f:
                f.write("This is a test document. " * 100)
            
            # Create a simple markdown file
            md_file = os.path.join(tmpdir, "test.md")
            with open(md_file, "w") as f:
                f.write("""# Test Heading

This is a paragraph with some content.

## Code Example

```python
def hello():
    print("Hello")
```

More text here.
""" + "Additional content. " * 50)
            
            yield tmpdir
    
    @pytest.fixture
    def single_file(self):
        """Create a single temp file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Single file test content. " * 50)
            yield f.name
        os.unlink(f.name)
    
    def test_run_krira_benchmark(self, single_file):
        """Test Krira benchmark runs successfully."""
        from Krira_Chunker import ChunkConfig
        from Krira_Chunker.bench.runner import run_krira_benchmark
        
        cfg = ChunkConfig(max_chars=500, overlap_chars=50)
        file_bytes = os.path.getsize(single_file)
        
        result = run_krira_benchmark(single_file, cfg, file_bytes)
        
        assert result.library == "krira_chunker"
        assert result.chunk_count > 0
        assert result.chars_total > 0
        assert result.duration_s > 0
        assert result.streaming is True
        assert not result.skipped
    
    def test_langchain_skipped_when_not_installed(self, single_file):
        """Test LangChain benchmark is skipped gracefully when not installed."""
        from Krira_Chunker.bench.runner import run_langchain_benchmark
        
        # Read file text
        with open(single_file, "r") as f:
            text = f.read()
        
        file_bytes = os.path.getsize(single_file)
        result = run_langchain_benchmark(single_file, text, 500, 50, file_bytes)
        
        # Either runs successfully OR is skipped with proper reason
        assert result.library == "langchain"
        if result.skipped:
            assert "not installed" in result.skip_reason.lower() or "cannot" in result.skip_reason.lower()
        else:
            assert result.chunk_count > 0
    
    def test_llamaindex_skipped_when_not_installed(self, single_file):
        """Test LlamaIndex benchmark is skipped gracefully when not installed."""
        from Krira_Chunker.bench.runner import run_llamaindex_sentence_benchmark
        
        with open(single_file, "r") as f:
            text = f.read()
        
        file_bytes = os.path.getsize(single_file)
        result = run_llamaindex_sentence_benchmark(single_file, text, 500, 50, file_bytes)
        
        assert result.library == "llama_index"
        if result.skipped:
            assert "not installed" in result.skip_reason.lower() or "cannot" in result.skip_reason.lower()
        else:
            assert result.chunk_count > 0
    
    def test_full_benchmark_with_directory(self, temp_corpus):
        """Test full benchmark with a directory corpus."""
        from Krira_Chunker.bench.runner import run_full_benchmark
        
        report = run_full_benchmark(
            corpus_path=temp_corpus,
            chunk_size=500,
            chunk_overlap=50,
            verbose=False,
        )
        
        assert report.timestamp
        assert report.system_info["python_version"]
        assert report.krira_config["max_chars"] == 500
        assert len(report.results) > 0
        
        # Check that Krira results exist
        krira_results = [r for r in report.results if r["library"] == "krira_chunker"]
        assert len(krira_results) > 0
        
        for r in krira_results:
            assert r["chunk_count"] > 0
            assert r["streaming"] is True
    
    def test_full_benchmark_with_single_file(self, single_file):
        """Test full benchmark with a single file."""
        from Krira_Chunker.bench.runner import run_full_benchmark
        
        report = run_full_benchmark(
            corpus_path=single_file,
            chunk_size=500,
            chunk_overlap=50,
            verbose=False,
        )
        
        assert len(report.results) >= 1
        
        krira_results = [r for r in report.results if r["library"] == "krira_chunker"]
        assert len(krira_results) == 1
    
    def test_report_json_serializable(self, single_file):
        """Test that report can be serialized to JSON."""
        from Krira_Chunker.bench.runner import run_full_benchmark
        
        report = run_full_benchmark(
            corpus_path=single_file,
            chunk_size=500,
            chunk_overlap=50,
        )
        
        # Should not raise
        json_str = json.dumps(report.to_dict(), default=str)
        assert len(json_str) > 0
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert "results" in parsed
        assert "system_info" in parsed
    
    def test_collect_corpus_files(self, temp_corpus):
        """Test corpus file collection."""
        from Krira_Chunker.bench.runner import collect_corpus_files
        
        files = collect_corpus_files(temp_corpus)
        
        assert len(files) >= 2  # At least .txt and .md
        
        for file_path, file_bytes in files:
            assert os.path.exists(file_path)
            assert file_bytes > 0
    
    def test_quality_metrics_computed(self, single_file):
        """Test that quality metrics are computed."""
        from Krira_Chunker import ChunkConfig
        from Krira_Chunker.bench.runner import run_krira_benchmark
        
        cfg = ChunkConfig(max_chars=100, overlap_chars=20)
        file_bytes = os.path.getsize(single_file)
        
        result = run_krira_benchmark(single_file, cfg, file_bytes)
        
        # Quality metrics should be defined
        assert result.avg_chunk_len_chars >= 0
        assert result.empty_chunk_count >= 0
        assert result.very_large_chunk_count >= 0
        assert 0 <= result.codeblock_break_rate <= 1
        assert 0 <= result.sentence_break_rate <= 1
    
    def test_chunk_previews_captured(self, single_file):
        """Test that chunk previews are captured."""
        from Krira_Chunker import ChunkConfig
        from Krira_Chunker.bench.runner import run_krira_benchmark
        
        cfg = ChunkConfig(max_chars=100, overlap_chars=20)
        file_bytes = os.path.getsize(single_file)
        
        result = run_krira_benchmark(single_file, cfg, file_bytes)
        
        # Should have up to 2 previews
        assert len(result.chunk_previews) <= 2
        if result.chunk_count > 0:
            assert len(result.chunk_previews) >= 1
            assert len(result.chunk_previews[0]) <= 80


class TestBenchmarkQualityMetrics:
    """Tests for quality metric calculations."""
    
    def test_codeblock_break_detection(self):
        """Test code block break detection."""
        from Krira_Chunker.bench.runner import count_codeblock_breaks
        
        # Complete code block - not broken
        assert count_codeblock_breaks("```python\ncode\n```") is False
        
        # Odd number of fences - broken
        assert count_codeblock_breaks("```python\ncode") is True
        
        # Two complete blocks - not broken
        assert count_codeblock_breaks("```\na\n```\n```\nb\n```") is False
        
        # No fences - not broken
        assert count_codeblock_breaks("just text") is False
    
    def test_sentence_ending_detection(self):
        """Test sentence ending detection."""
        from Krira_Chunker.bench.runner import check_sentence_ending
        
        assert check_sentence_ending("This is a sentence.") is True
        assert check_sentence_ending("Is this a question?") is True
        assert check_sentence_ending("Wow!") is True
        assert check_sentence_ending("He said \"hello.\"") is True
        
        # Broken sentences
        assert check_sentence_ending("This is incomplete") is False
        assert check_sentence_ending("The end") is False


class TestSystemInfo:
    """Tests for system info collection."""
    
    def test_get_system_info(self):
        """Test system info collection."""
        from Krira_Chunker.bench.runner import get_system_info
        
        info = get_system_info()
        
        assert "python_version" in info
        assert "platform" in info
        assert info["python_version"]  # Not empty
    
    def test_get_memory_usage(self):
        """Test memory usage collection."""
        from Krira_Chunker.bench.runner import get_memory_usage
        
        mem = get_memory_usage()
        
        # Returns None if psutil not installed, otherwise a positive number
        assert mem is None or mem > 0
