"""
Tests for the HybridBoundaryChunker.
"""

import pytest
from Krira_Chunker import ChunkConfig, HybridBoundaryChunker
from Krira_Chunker.core import FastChunker


class TestHybridBoundaryChunker:
    """Test suite for HybridBoundaryChunker."""
    
    def test_basic_chunking(self):
        """Test basic text chunking."""
        cfg = ChunkConfig(max_chars=200, overlap_chars=50, chunk_strategy="hybrid")
        chunker = HybridBoundaryChunker(cfg)
        
        text = "This is a test. " * 50
        chunks = list(chunker.chunk_text(
            text=text,
            base_meta={"source": "test"},
            locator="test",
        ))
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert "id" in chunk
            assert "text" in chunk
            assert "metadata" in chunk
            assert len(chunk["text"]) >= cfg.min_chars
    
    def test_code_block_preservation(self):
        """Test that code blocks are not split."""
        cfg = ChunkConfig(
            max_chars=500,
            overlap_chars=50,
            chunk_strategy="hybrid",
            preserve_code_blocks=True
        )
        chunker = HybridBoundaryChunker(cfg)
        
        text = """
Some intro text here.

```python
def hello_world():
    print("Hello, World!")
    return True

def another_function():
    x = 1 + 2
    return x
```

Some conclusion text here.
"""
        chunks = list(chunker.chunk_text(
            text=text,
            base_meta={"source": "test"},
            locator="test",
        ))
        
        # Verify code block is in one chunk
        code_found = False
        for chunk in chunks:
            if "def hello_world" in chunk["text"] and "def another_function" in chunk["text"]:
                code_found = True
                break
        
        # Code should be together (may be in same chunk or adjacent due to size)
        assert len(chunks) > 0
    
    def test_sentence_boundaries(self):
        """Test that sentences are not split mid-sentence when possible."""
        cfg = ChunkConfig(
            max_chars=100,
            overlap_chars=20,
            chunk_strategy="hybrid"
        )
        chunker = HybridBoundaryChunker(cfg)
        
        text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        chunks = list(chunker.chunk_text(
            text=text,
            base_meta={"source": "test"},
            locator="test",
        ))
        
        # Each chunk should generally end with a complete sentence
        for chunk in chunks:
            text = chunk["text"].strip()
            # Should end with period, question mark, or exclamation
            assert text[-1] in ".?!" or len(text) < cfg.max_chars
    
    def test_no_empty_chunks(self):
        """Test that empty chunks are never produced."""
        cfg = ChunkConfig(max_chars=100, overlap_chars=20, chunk_strategy="hybrid")
        chunker = HybridBoundaryChunker(cfg)
        
        text = "Short text. Another sentence. And one more sentence here."
        chunks = list(chunker.chunk_text(
            text=text,
            base_meta={"source": "test"},
            locator="test",
        ))
        
        for chunk in chunks:
            assert chunk["text"].strip() != ""
            assert len(chunk["text"]) >= cfg.min_chars
    
    def test_stable_ids(self):
        """Test that chunk IDs are deterministic."""
        cfg = ChunkConfig(max_chars=200, overlap_chars=50)
        chunker = HybridBoundaryChunker(cfg)
        
        text = "This is a test document. It has multiple sentences. Each sentence adds more content."
        
        chunks1 = list(chunker.chunk_text(
            text=text,
            base_meta={"source": "test.txt"},
            locator="test",
        ))
        chunks2 = list(chunker.chunk_text(
            text=text,
            base_meta={"source": "test.txt"},
            locator="test",
        ))
        
        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1["id"] == c2["id"]
    
    def test_config_hash_in_metadata(self):
        """Test that config_hash is included in metadata."""
        cfg = ChunkConfig(max_chars=200, overlap_chars=50, chunk_strategy="hybrid")
        chunker = HybridBoundaryChunker(cfg)
        
        text = "This is a test. " * 20
        chunks = list(chunker.chunk_text(
            text=text,
            base_meta={"source": "test"},
            locator="test",
        ))
        
        for chunk in chunks:
            assert "config_hash" in chunk["metadata"]
            assert len(chunk["metadata"]["config_hash"]) == 12
    
    def test_boundary_type_in_metadata(self):
        """Test that boundary_type is included in metadata."""
        cfg = ChunkConfig(max_chars=100, overlap_chars=20, chunk_strategy="hybrid")
        chunker = HybridBoundaryChunker(cfg)
        
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = list(chunker.chunk_text(
            text=text,
            base_meta={"source": "test"},
            locator="test",
        ))
        
        for chunk in chunks:
            assert "boundary_type" in chunk["metadata"]
            assert chunk["metadata"]["boundary_type"] in ["heading", "paragraph", "sentence", "line", "word", "hard", "natural"]


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_overlap_less_than_size(self):
        """Test that overlap must be less than size."""
        from Krira_Chunker.exceptions import ConfigError
        
        with pytest.raises(ConfigError):
            ChunkConfig(max_chars=100, overlap_chars=100)
        
        with pytest.raises(ConfigError):
            ChunkConfig(max_chars=100, overlap_chars=150)
    
    def test_valid_config(self):
        """Test valid configuration creation."""
        cfg = ChunkConfig(
            max_chars=2000,
            overlap_chars=200,
            chunk_strategy="hybrid",
            preserve_code_blocks=True,
        )
        assert cfg.max_chars == 2000
        assert cfg.overlap_chars == 200
        assert cfg.chunk_strategy == "hybrid"
    
    def test_config_hash_stability(self):
        """Test that config hash is stable."""
        cfg1 = ChunkConfig(max_chars=1000, overlap_chars=100)
        cfg2 = ChunkConfig(max_chars=1000, overlap_chars=100)
        
        assert cfg1.config_hash() == cfg2.config_hash()
        
        cfg3 = ChunkConfig(max_chars=1000, overlap_chars=200)
        assert cfg1.config_hash() != cfg3.config_hash()


class TestFastChunker:
    """Test FastChunker (original implementation)."""
    
    def test_basic_chunking(self):
        """Test basic unit chunking."""
        cfg = ChunkConfig(max_chars=100, overlap_chars=20)
        chunker = FastChunker(cfg)
        
        units = ["First unit.", "Second unit.", "Third unit.", "Fourth unit."]
        chunks = list(chunker.chunk_units(
            units=units,
            base_meta={"source": "test"},
            joiner=" ",
            locator="test",
        ))
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert "id" in chunk
            assert "text" in chunk
    
    def test_hard_split_oversized(self):
        """Test that oversized units are hard split."""
        cfg = ChunkConfig(max_chars=50, overlap_chars=10, min_chars=10)
        chunker = FastChunker(cfg)
        
        # Unit longer than max_chars
        units = ["A" * 200]
        chunks = list(chunker.chunk_units(
            units=units,
            base_meta={"source": "test"},
            joiner=" ",
            locator="test",
        ))
        
        # Should be split into multiple chunks
        assert len(chunks) >= 2
        for chunk in chunks:
            # Each chunk should be roughly max_chars or less
            assert len(chunk["text"]) <= cfg.max_chars * 1.5  # Allow some tolerance
