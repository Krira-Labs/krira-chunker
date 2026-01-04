import os
import json
import csv
import logging
from krira_augment import Pipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = "test_data"
OUTPUT_DIR = "test_output"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        for i in range(10):
            f.write(f"This is line {i} of the text file.\n")
    return path

def test_pipeline():
    pipeline = Pipeline()
    
    files = {
        "CSV": create_dummy_csv(),
        "JSONL": create_dummy_jsonl(),
        "JSON": create_dummy_json(), # Might fail if treated as lines
        "TXT": create_dummy_txt(),
    }
    
    # These require extra libs, we will just simulate input paths if we can't create them easily here
    # without installing libs in this script. But for now let's test basics + fail on others.
    
    for fmt, path in files.items():
        output = os.path.join(OUTPUT_DIR, f"output_{fmt}.jsonl")
        print(f"Testing {fmt} -> {path}")
        try:
            stats = pipeline.process(input_path=path, output_path=output)
            print(f"✅ {fmt} Success: {stats}")
        except Exception as e:
            print(f"❌ {fmt} Failed: {e}")

if __name__ == "__main__":
    test_pipeline()
