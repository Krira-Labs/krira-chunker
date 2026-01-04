import time
import os
import json
from krira_augment import Pipeline, PipelineConfig, SplitStrategy

def create_dummy_data(filename, rows=10000):
    print(f"Generating {rows} rows of dummy data...")
    with open(filename, "w", encoding="utf-8") as f:
        f.write("id,content,timestamp\n")
        for i in range(rows):
            text = f"Row {i}. " + "Krira is fast. " * 50
            f.write(f"{i},\"{text}\",2024-01-01\n")

def main():
    input_file = "demo_data.csv"
    output_file = "demo_output.jsonl"
    
    # 1. Generate Input
    if not os.path.exists(input_file):
        create_dummy_data(input_file, rows=100_000)
    
    # 2. Configure (Advanced)
    config = PipelineConfig(
        chunk_size=500,
        strategy=SplitStrategy.SMART,
        clean_html=True
    )
    
    # 3. Initialize Pipeline
    pipeline = Pipeline(config)

    print(f"Processing {input_file}...")
    
    try:
        # 4. Process
        result = pipeline.process(input_file, output_file)
        
        print(f"\n✅ Done!")
        print(f"Job ID: {result.job_id}")
        print(f"Throughput: {result.mb_per_second:.2f} MB/s")
        print(f"Output: {output_file}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
