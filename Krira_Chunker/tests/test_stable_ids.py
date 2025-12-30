"""
Tests for stable ID determinism.
"""

import pytest
from Krira_Chunker import ChunkConfig, HybridBoundaryChunker
from Krira_Chunker.core import stable_id, FastChunker


class TestStableIds:
    """Test that chunk IDs are deterministic and stable."""
    
    def test_stable_id_function(self):
        """Test the stable_id function directly."""
        id1 = stable_id("source.txt", "page=1", 0, "Hello World")
        id2 = stable_id("source.txt", "page=1", 0, "Hello World")
        
        assert id1 == id2
        assert len(id1) == 32  # MD5 hex length
    
    def test_different_source_different_id(self):
        """Test that different sources produce different IDs."""
        id1 = stable_id("source1.txt", "page=1", 0, "Hello World")
        id2 = stable_id("source2.txt", "page=1", 0, "Hello World")
        
        assert id1 != id2
    
    def test_different_locator_different_id(self):
        """Test that different locators produce different IDs."""
        id1 = stable_id("source.txt", "page=1", 0, "Hello World")
        id2 = stable_id("source.txt", "page=2", 0, "Hello World")
        
        assert id1 != id2
    
    def test_different_chunk_index_different_id(self):
        """Test that different chunk indices produce different IDs."""
        id1 = stable_id("source.txt", "page=1", 0, "Hello World")
        id2 = stable_id("source.txt", "page=1", 1, "Hello World")
        
        assert id1 != id2
    
    def test_different_text_different_id(self):
        """Test that different text produces different IDs."""
        id1 = stable_id("source.txt", "page=1", 0, "Hello World")
        id2 = stable_id("source.txt", "page=1", 0, "Goodbye World")
        
        assert id1 != id2
    
    def test_chunker_produces_stable_ids(self):
        """Test that chunker produces stable IDs across runs."""
        cfg = ChunkConfig(max_chars=100, overlap_chars=20)
        chunker = FastChunker(cfg)
        
        text = "This is a test. Another sentence. More text here."
        
        chunks1 = list(chunker.chunk_text(
            text=text,
            base_meta={"source": "test.txt"},
            mode="prose",
            locator="test",
        ))
        
        chunks2 = list(chunker.chunk_text(
            text=text,
            base_meta={"source": "test.txt"},
            mode="prose",
            locator="test",
        ))
        
        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1["id"] == c2["id"]
            assert c1["text"] == c2["text"]
    
    def test_hybrid_chunker_produces_stable_ids(self):
        """Test that HybridBoundaryChunker produces stable IDs."""
        cfg = ChunkConfig(max_chars=100, overlap_chars=20, chunk_strategy="hybrid")
        chunker = HybridBoundaryChunker(cfg)
        
        text = "This is a test. Another sentence here. And more text."
        
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
    
    def test_chunk_ordering_consistent(self):
        """Test that chunk ordering is consistent."""
        cfg = ChunkConfig(max_chars=50, overlap_chars=10)
        chunker = FastChunker(cfg)
        
        text = "One. Two. Three. Four. Five. Six. Seven. Eight."
        
        for _ in range(5):  # Run multiple times
            chunks = list(chunker.chunk_text(
                text=text,
                base_meta={"source": "test"},
                mode="prose",
                locator="test",
            ))
            
            # Check indices are sequential
            indices = [c["metadata"]["chunk_index"] for c in chunks]
            assert indices == list(range(len(chunks)))
