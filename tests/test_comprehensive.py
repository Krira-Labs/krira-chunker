import os
import json
import csv
import shutil
import logging
import pytest
from krira_augment import Pipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "test_data_comprehensive"
OUTPUT_DIR = "test_output_comprehensive"

@pytest.fixture(scope="session", autouse=True)
def setup_dirs():
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    yield
    # Cleanup after tests? Maybe keep for inspection.
    # shutil.rmtree(DATA_DIR)
    # shutil.rmtree(OUTPUT_DIR)

def create_dummy_csv():
    path = os.path.join(DATA_DIR, "test.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text", "category"])
        for i in range(10):
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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path

def create_dummy_txt():
    path = os.path.join(DATA_DIR, "test.txt")
    with open(path, "w", encoding="utf-8") as f:
        for i in range(20):
            f.write(f"This is line {i} of the text file.\n")
    return path

def create_dummy_xml():
    path = os.path.join(DATA_DIR, "test.xml")
    with open(path, "w", encoding="utf-8") as f:
        f.write("<root>\n")
        for i in range(10):
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
    pipeline = Pipeline()
    path = create_dummy_jsonl()
    output = os.path.join(OUTPUT_DIR, "output_jsonl.jsonl")
    stats = pipeline.process(input_path=path, output_path=output)
    assert os.path.exists(output)
    
    with open(output, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        assert len(lines) > 0 # Should have chunks

def test_pipeline_json():
    pipeline = Pipeline()
    path = create_dummy_json()
    # Using explicit output path
    output = os.path.join(OUTPUT_DIR, "output_json.jsonl")
    stats = pipeline.process(input_path=path, output_path=output)
    assert os.path.exists(output)

def test_pipeline_txt():
    pipeline = Pipeline()
    path = create_dummy_txt()
    # Using auto output path
    stats = pipeline.process(input_path=path)
    assert os.path.exists(stats.output_file)
    assert stats.output_file.endswith("_processed.jsonl")

def test_pipeline_xml():
    pipeline = Pipeline()
    path = create_dummy_xml()
    output = os.path.join(OUTPUT_DIR, "output_xml.jsonl")
    stats = pipeline.process(input_path=path, output_path=output)
    assert os.path.exists(output)

# Skip PDF/DOCX/XLSX/URL in basic unit tests unless we know libs are there and we have valid dummy files.
# But we can try to call them if we had dummy file generators for them. 
# For now, this covers the Logic change (optional output) + basic formats.

if __name__ == "__main__":
    # If run directly, run manual checks
    try:
        setup_dirs()
        create_dummy_csv()
        create_dummy_json()
        create_dummy_jsonl()
        create_dummy_txt()
        create_dummy_xml()
        
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
    except Exception as e:
        print(f"❌ Tests Failed: {e}")
        raise
