"""
Benchmark Script for Krira Augment

Compares performance and memory usage between:
1. File-based processing (`process()`)
2. Streaming processing (`process_stream()`)

Metrics:
- Execution Time (seconds)
- Peak Memory Usage (MB)
- Throughput (MB/s)
"""

import os
import time
import tracemalloc
import tempfile
import random
import string
from krira_augment import Pipeline, PipelineConfig

# Configuration
SIZES_MB = [1, 10, 100]  # File sizes to test
CHUNK_SIZE = 512
OVERLAP = 50

def generate_random_file(size_mb, path):
    """Generate a file with random text of specified size in MB."""
    print(f"  Generating {size_mb}MB file...")
    chunk_size = 1024 * 1024  # 1 MB chunks
    chars = string.ascii_letters + string.digits + " " * 20  # Bias towards spaces for realistic text
    
    with open(path, 'w', encoding='utf-8') as f:
        for _ in range(size_mb):
            # Generate 1MB of text
            text = ''.join(random.choices(chars, k=chunk_size))
            f.write(text)

def measure_file_based(pipeline, input_path):
    """Measure file-based processing."""
    tracemalloc.start()
    start_time = time.time()
    
    try:
        # File-based creates an output file
        result = pipeline.process(input_path)
        # Verify it really worked
        if not result.chunks_created:
            raise Exception("No chunks created!")
    finally:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
    duration = time.time() - start_time
    peak_mb = peak / (1024 * 1024)
    
    # Clean up output file
    if result.output_file and os.path.exists(result.output_file):
        os.unlink(result.output_file)
        
    return duration, peak_mb, result.chunks_created

def measure_streaming(pipeline, input_path):
    """Measure streaming processing."""
    tracemalloc.start()
    start_time = time.time()
    chunk_count = 0
    
    try:
        # Iterate through all chunks to emulate full consumption
        for _ in pipeline.process_stream(input_path):
            chunk_count += 1
    finally:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
    duration = time.time() - start_time
    peak_mb = peak / (1024 * 1024)
    
    return duration, peak_mb, chunk_count

def run_benchmarks():
    print("=" * 80)
    print(f"🚀 Krira Augment Benchmark (v2.1.0)")
    print("=" * 80)
    print(f"{'Size (MB)':<10} | {'Mode':<12} | {'Time (s)':<10} | {'Throughput':<12} | {'Peak Mem':<10} | {'Chunks':<8}")
    print("-" * 80)

    config = PipelineConfig(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP)
    pipeline = Pipeline(config=config)
    
    results = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for size_mb in SIZES_MB:
            input_path = os.path.join(temp_dir, f"test_{size_mb}mb.txt")
            generate_random_file(size_mb, input_path)
            
            # --- File-Based ---
            time_fb, mem_fb, chunks_fb = measure_file_based(pipeline, input_path)
            speed_fb = size_mb / time_fb if time_fb > 0 else 0
            
            print(f"{size_mb:<10} | {'File':<12} | {time_fb:<10.3f} | {speed_fb:<6.1f} MB/s | {mem_fb:<6.1f} MB | {chunks_fb:<8}")
            
            # --- Streaming ---
            time_st, mem_st, chunks_st = measure_streaming(pipeline, input_path)
            speed_st = size_mb / time_st if time_st > 0 else 0
            
            print(f"{size_mb:<10} | {'Stream':<12} | {time_st:<10.3f} | {speed_st:<6.1f} MB/s | {mem_st:<6.1f} MB | {chunks_st:<8}")
            print("-" * 80)
            
            results.append({
                "size": size_mb,
                "file_time": time_fb,
                "stream_time": time_st,
                "file_mem": mem_fb,
                "stream_mem": mem_st
            })

    print("\n✅ Benchmark Complete!")
    
    # Summary Analysis
    print("\n📊 Analysis:")
    for res in results:
        speedup = (res["file_time"] - res["stream_time"]) / res["file_time"] * 100
        mem_diff = res["file_mem"] - res["stream_mem"]
        print(f"Size {res['size']}MB:")
        print(f"  - Speedup: {speedup:+.1f}% {'(Faster)' if speedup > 0 else '(Slower)'}")
        print(f"  - Memory Delta: {mem_diff:+.1f} MB")

if __name__ == "__main__":
    run_benchmarks()
