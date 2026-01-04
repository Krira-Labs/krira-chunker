<<<<<<< HEAD
"""
Comprehensive Tests for Krira Augment

Tests both file-based and streaming modes across various file formats.
"""

=======
>>>>>>> c85f1727f4e8aad48e031764a3067eb4374a5a9f
import os
import json
import csv
import shutil
import logging
import pytest
<<<<<<< HEAD
from krira_augment import Pipeline, PipelineConfig
=======
from krira_augment import Pipeline
>>>>>>> c85f1727f4e8aad48e031764a3067eb4374a5a9f

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "test_data_comprehensive"
OUTPUT_DIR = "test_output_comprehensive"

<<<<<<< HEAD

@pytest.fixture(scope="session", autouse=True)
def setup_dirs():
    """Setup and teardown test directories."""
=======
@pytest.fixture(scope="session", autouse=True)
def setup_dirs():
>>>>>>> c85f1727f4e8aad48e031764a3067eb4374a5a9f
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    yield
<<<<<<< HEAD
    # Cleanup after tests? Keep for inspection.
    # shutil.rmtree(DATA_DIR)
    # shutil.rmtree(OUTPUT_DIR)


# =============================================================================
# Test Data Generators
# =============================================================================

def create_dummy_csv():
    """Create a test CSV file."""
=======
    # Cleanup after tests? Maybe keep for inspection.
    # shutil.rmtree(DATA_DIR)
    # shutil.rmtree(OUTPUT_DIR)

def create_dummy_csv():
>>>>>>> c85f1727f4e8aad48e031764a3067eb4374a5a9f
    path = os.path.join(DATA_DIR, "test.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text", "category"])
        for i in range(10):
<<<<<<< HEAD
            writer.writerow([i, f"This is row {i} of the CSV file with some longer text to chunk.", "test"])
    return path


def create_dummy_jsonl():
    """Create a test JSONL file."""
    path = os.path.join(DATA_DIR, "test.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for i in range(10):
            f.write(json.dumps({"id": i, "text": f"This is JSONL line {i} with additional content."}) + "\n")
    return path


def create_dummy_json():
    """Create a test JSON file."""
    path = os.path.join(DATA_DIR, "test.json")
    data = [{"id": i, "text": f"This is JSON object {i} with some text content."} for i in range(10)]
=======
            writer.writerow([i, f"This is row {i} of the CSV file.", "test"])
    return path

def create_dummy_jsonl():
    path = os.path.join(DATA_DIR, "test.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for i in range(10):
            f.write(json.dumps({"id": i, "text": f"This is JSONL line {i}"}) + "\n")
    return path

def create_dummy_json():
    path = os.path.join(DATA_DIR, "test.json")
    data = [{"id": i, "text": f"This is JSON object {i}"} for i in range(10)]
>>>>>>> c85f1727f4e8aad48e031764a3067eb4374a5a9f
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path

<<<<<<< HEAD

def create_dummy_txt():
    """Create a test text file."""
    path = os.path.join(DATA_DIR, "test.txt")
    with open(path, "w", encoding="utf-8") as f:
        for i in range(20):
            f.write(f"This is line {i} of the text file with some content for testing.\n")
    return path


def create_dummy_xml():
    """Create a test XML file."""
=======
def create_dummy_txt():
    path = os.path.join(DATA_DIR, "test.txt")
    with open(path, "w", encoding="utf-8") as f:
        for i in range(20):
            f.write(f"This is line {i} of the text file.\n")
    return path

def create_dummy_xml():
>>>>>>> c85f1727f4e8aad48e031764a3067eb4374a5a9f
    path = os.path.join(DATA_DIR, "test.xml")
    with open(path, "w", encoding="utf-8") as f:
        f.write("<root>\n")
        for i in range(10):
<<<<<<< HEAD
            f.write(f"  <item id='{i}'>This is XML item {i} with content</item>\n")
        f.write("</root>")
    return path


def create_large_txt(num_lines=1000):
    """Create a larger text file for performance testing."""
    path = os.path.join(DATA_DIR, "large_test.txt")
    with open(path, "w", encoding="utf-8") as f:
        for i in range(num_lines):
            f.write(f"Line {i}: " + "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 5 + "\n")
    return path


# =============================================================================
# File-Based Mode Tests
# =============================================================================

def test_pipeline_csv():
    """Test file-based processing of CSV files."""
    pipeline = Pipeline()
    path = create_dummy_csv()
    stats = pipeline.process(input_path=path)
    assert stats.output_file is not None
    assert os.path.exists(stats.output_file)
    assert stats.mb_per_second >= 0
    assert stats.chunks_created > 0


def test_pipeline_jsonl():
    """Test file-based processing of JSONL files."""
=======
            f.write(f"  <item id='{i}'>This is XML item {i}</item>\n")
        f.write("</root>")
    return path

def test_pipeline_csv():
    pipeline = Pipeline()
    path = create_dummy_csv()
    stats = pipeline.process(input_path=path) # Optional output path test
    assert stats.output_file is not None
    assert os.path.exists(stats.output_file)
    assert stats.mb_per_second >= 0

def test_pipeline_jsonl():
>>>>>>> c85f1727f4e8aad48e031764a3067eb4374a5a9f
    pipeline = Pipeline()
    path = create_dummy_jsonl()
    output = os.path.join(OUTPUT_DIR, "output_jsonl.jsonl")
    stats = pipeline.process(input_path=path, output_path=output)
    assert os.path.exists(output)
    
    with open(output, 'r', encoding='utf-8') as f:
        lines = f.readlines()
<<<<<<< HEAD
        assert len(lines) > 0


def test_pipeline_json():
    """Test file-based processing of JSON files."""
    pipeline = Pipeline()
    path = create_dummy_json()
=======
        assert len(lines) > 0 # Should have chunks

def test_pipeline_json():
    pipeline = Pipeline()
    path = create_dummy_json()
    # Using explicit output path
>>>>>>> c85f1727f4e8aad48e031764a3067eb4374a5a9f
    output = os.path.join(OUTPUT_DIR, "output_json.jsonl")
    stats = pipeline.process(input_path=path, output_path=output)
    assert os.path.exists(output)

<<<<<<< HEAD

def test_pipeline_txt():
    """Test file-based processing of text files."""
    pipeline = Pipeline()
    path = create_dummy_txt()
=======
def test_pipeline_txt():
    pipeline = Pipeline()
    path = create_dummy_txt()
    # Using auto output path
>>>>>>> c85f1727f4e8aad48e031764a3067eb4374a5a9f
    stats = pipeline.process(input_path=path)
    assert os.path.exists(stats.output_file)
    assert stats.output_file.endswith("_processed.jsonl")

<<<<<<< HEAD

def test_pipeline_xml():
    """Test file-based processing of XML files."""
=======
def test_pipeline_xml():
>>>>>>> c85f1727f4e8aad48e031764a3067eb4374a5a9f
    pipeline = Pipeline()
    path = create_dummy_xml()
    output = os.path.join(OUTPUT_DIR, "output_xml.jsonl")
    stats = pipeline.process(input_path=path, output_path=output)
    assert os.path.exists(output)

<<<<<<< HEAD

# =============================================================================
# Streaming Mode Tests
# =============================================================================

def test_streaming_basic():
    """Test basic streaming functionality."""
    config = PipelineConfig(chunk_size=512, chunk_overlap=50)
    pipeline = Pipeline(config=config)
    path = create_dummy_txt()
    
    chunks = list(pipeline.process_stream(path))
    
    assert len(chunks) > 0
    assert "text" in chunks[0]
    assert "metadata" in chunks[0]
    assert chunks[0]["metadata"]["chunk_index"] == 0


def test_streaming_csv():
    """Test streaming with CSV files."""
    config = PipelineConfig(chunk_size=256, chunk_overlap=25)
    pipeline = Pipeline(config=config)
    path = create_dummy_csv()
    
    chunk_count = 0
    for chunk in pipeline.process_stream(path):
        assert "text" in chunk
        assert "metadata" in chunk
        assert len(chunk["text"]) > 0
        chunk_count += 1
    
    assert chunk_count > 0
    print(f"✅ Streamed {chunk_count} chunks from CSV")


def test_streaming_jsonl():
    """Test streaming with JSONL files."""
    config = PipelineConfig(chunk_size=512, chunk_overlap=50)
    pipeline = Pipeline(config=config)
    path = create_dummy_jsonl()
    
    chunks = list(pipeline.process_stream(path))
    
    assert len(chunks) > 0
    for i, chunk in enumerate(chunks):
        assert chunk["metadata"]["chunk_index"] == i


def test_streaming_large_file():
    """Test streaming with a larger file."""
    config = PipelineConfig(chunk_size=1000, chunk_overlap=100)
    pipeline = Pipeline(config=config)
    path = create_large_txt(500)  # 500 lines
    
    chunk_count = 0
    total_chars = 0
    
    for chunk in pipeline.process_stream(path):
        chunk_count += 1
        total_chars += len(chunk["text"])
        
        # Verify chunk structure
        assert "text" in chunk
        assert "metadata" in chunk
        assert "source" in chunk["metadata"]
        assert "chunk_index" in chunk["metadata"]
        assert "char_count" in chunk["metadata"]
    
    assert chunk_count > 0
    print(f"✅ Streamed {chunk_count} chunks, {total_chars:,} total characters")


def test_streaming_file_not_found():
    """Test streaming with non-existent file raises error."""
    config = PipelineConfig()
    pipeline = Pipeline(config=config)
    
    with pytest.raises(FileNotFoundError):
        list(pipeline.process_stream("nonexistent_file.txt"))


def test_streaming_early_stop():
    """Test that streaming can be stopped early without issues."""
    config = PipelineConfig(chunk_size=100, chunk_overlap=10)
    pipeline = Pipeline(config=config)
    path = create_large_txt(100)
    
    count = 0
    for chunk in pipeline.process_stream(path):
        count += 1
        if count >= 5:
            break
    
    assert count == 5
    print(f"✅ Stopped streaming early after {count} chunks")


# =============================================================================
# Configuration Tests
# =============================================================================

def test_pipeline_config_custom():
    """Test custom pipeline configuration."""
    config = PipelineConfig(
        chunk_size=256,
        chunk_overlap=25,
        clean_html=True,
        clean_unicode=True,
        min_chunk_len=10
    )
    
    assert config.chunk_size == 256
    assert config.chunk_overlap == 25
    
    pipeline = Pipeline(config=config)
    path = create_dummy_txt()
    stats = pipeline.process(input_path=path)
    
    assert stats.chunks_created > 0


def test_streaming_with_custom_config():
    """Test streaming with custom configuration."""
    config = PipelineConfig(
        chunk_size=100,
        chunk_overlap=10
    )
    pipeline = Pipeline(config=config)
    path = create_dummy_txt()
    
    chunks = list(pipeline.process_stream(path))
    
    # With smaller chunk size, we should get more chunks
    assert len(chunks) > 0
    for chunk in chunks:
        # Most chunks should be around chunk_size (allow some flexibility)
        assert len(chunk["text"]) <= config.chunk_size + 50  # Some margin for boundaries


# =============================================================================
# Performance Tests
# =============================================================================

def test_streaming_performance():
    """Basic performance test for streaming."""
    import time
    
    config = PipelineConfig(chunk_size=1000, chunk_overlap=100)
    pipeline = Pipeline(config=config)
    path = create_large_txt(1000)  # 1000 lines
    
    start_time = time.time()
    chunk_count = sum(1 for _ in pipeline.process_stream(path))
    duration = time.time() - start_time
    
    chunks_per_second = chunk_count / duration if duration > 0 else 0
    
    print(f"\n📊 Performance Results:")
    print(f"   Chunks: {chunk_count}")
    print(f"   Duration: {duration:.3f}s")
    print(f"   Speed: {chunks_per_second:.1f} chunks/sec")
    
    assert chunk_count > 0
    assert duration < 10  # Should complete in under 10 seconds


# =============================================================================
# Main Entry Point
# =============================================================================
=======
# Skip PDF/DOCX/XLSX/URL in basic unit tests unless we know libs are there and we have valid dummy files.
# But we can try to call them if we had dummy file generators for them. 
# For now, this covers the Logic change (optional output) + basic formats.
>>>>>>> c85f1727f4e8aad48e031764a3067eb4374a5a9f

if __name__ == "__main__":
    # If run directly, run manual checks
    try:
<<<<<<< HEAD
        # Create dirs manually for direct run
        if os.path.exists(DATA_DIR):
            shutil.rmtree(DATA_DIR)
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Create test files
=======
        setup_dirs()
>>>>>>> c85f1727f4e8aad48e031764a3067eb4374a5a9f
        create_dummy_csv()
        create_dummy_json()
        create_dummy_jsonl()
        create_dummy_txt()
        create_dummy_xml()
<<<<<<< HEAD
        create_large_txt(500)
        
        # File-based tests
        print("\n" + "=" * 60)
        print("📁 FILE-BASED MODE TESTS")
        print("=" * 60)
=======
>>>>>>> c85f1727f4e8aad48e031764a3067eb4374a5a9f
        
        test_pipeline_csv()
        print("✅ CSV Test Passed")
        test_pipeline_jsonl()
        print("✅ JSONL Test Passed")
        test_pipeline_json()
        print("✅ JSON Test Passed")
        test_pipeline_txt()
        print("✅ TXT Test Passed")
        test_pipeline_xml()
        print("✅ XML Test Passed")
<<<<<<< HEAD
        
        # Streaming tests
        print("\n" + "=" * 60)
        print("🌊 STREAMING MODE TESTS")
        print("=" * 60)
        
        test_streaming_basic()
        print("✅ Basic Streaming Test Passed")
        test_streaming_csv()
        print("✅ CSV Streaming Test Passed")
        test_streaming_jsonl()
        print("✅ JSONL Streaming Test Passed")
        test_streaming_large_file()
        print("✅ Large File Streaming Test Passed")
        test_streaming_early_stop()
        print("✅ Early Stop Test Passed")
        
        # Performance test
        print("\n" + "=" * 60)
        print("⚡ PERFORMANCE TESTS")
        print("=" * 60)
        
        test_streaming_performance()
        print("✅ Performance Test Passed")
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Tests Failed: {e}")
        import traceback
        traceback.print_exc()
        raise

=======
    except Exception as e:
        print(f"❌ Tests Failed: {e}")
        raise
>>>>>>> c85f1727f4e8aad48e031764a3067eb4374a5a9f
