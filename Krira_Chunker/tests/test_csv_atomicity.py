"""
Tests for CSV atomicity - rows should never be split.
"""

import pytest
import tempfile
import os

from Krira_Chunker import ChunkConfig
from Krira_Chunker.exceptions import DependencyNotInstalledError


class TestCSVAtomicity:
    """Test that CSV rows are treated as atomic units."""
    
    @pytest.fixture
    def sample_csv(self):
        """Create a sample CSV file."""
        content = """name,email,description
John Doe,john@example.com,A software engineer with 10 years of experience
Jane Smith,jane@example.com,A data scientist specializing in machine learning
Bob Johnson,bob@example.com,A product manager leading cross-functional teams
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(content)
            return f.name
    
    @pytest.fixture
    def large_csv(self):
        """Create a larger CSV file."""
        lines = ["id,name,value"]
        for i in range(100):
            lines.append(f"{i},Item {i},Description for item {i} with some additional text")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("\n".join(lines))
            return f.name
    
    def test_row_not_split(self, sample_csv):
        """Test that individual rows are not split across chunks."""
        try:
            from Krira_Chunker.CSVChunker import CSVChunker
        except DependencyNotInstalledError:
            pytest.skip("polars not installed")
        
        cfg = ChunkConfig(max_chars=100, overlap_chars=20)
        chunker = CSVChunker(cfg)
        
        try:
            chunks = list(chunker.chunk_file(sample_csv))
        except DependencyNotInstalledError:
            pytest.skip("polars not installed")
        finally:
            os.unlink(sample_csv)
        
        # Check that email addresses are complete (not split)
        all_text = " ".join(c["text"] for c in chunks)
        assert "john@example.com" in all_text
        assert "jane@example.com" in all_text
        
        # Email should not be split across chunks
        for chunk in chunks:
            text = chunk["text"]
            if "@" in text:
                # If @ is present, the full email should be there
                assert "example.com" in text or text.endswith("@")
    
    def test_row_metadata(self, large_csv):
        """Test that row range metadata is correct."""
        try:
            from Krira_Chunker.CSVChunker import CSVChunker
        except DependencyNotInstalledError:
            pytest.skip("polars not installed")
        
        cfg = ChunkConfig(max_chars=500, overlap_chars=50)
        chunker = CSVChunker(cfg)
        
        try:
            chunks = list(chunker.chunk_file(large_csv))
        except DependencyNotInstalledError:
            pytest.skip("polars not installed")
        finally:
            os.unlink(large_csv)
        
        for chunk in chunks:
            meta = chunk["metadata"]
            assert "source_type" in meta
            assert meta["source_type"] == "csv"
            
            # Should have row range if multiple rows
            if "row_start" in meta:
                assert "row_end" in meta
                assert meta["row_start"] <= meta["row_end"]
    
    def test_chunk_index_sequential(self, sample_csv):
        """Test that chunk indices are sequential."""
        try:
            from Krira_Chunker.CSVChunker import CSVChunker
        except DependencyNotInstalledError:
            pytest.skip("polars not installed")
        
        cfg = ChunkConfig(max_chars=100, overlap_chars=20)
        chunker = CSVChunker(cfg)
        
        try:
            chunks = list(chunker.chunk_file(sample_csv))
        except DependencyNotInstalledError:
            pytest.skip("polars not installed")
        finally:
            os.unlink(sample_csv)
        
        indices = [c["metadata"]["chunk_index"] for c in chunks]
        expected = list(range(len(chunks)))
        assert indices == expected
