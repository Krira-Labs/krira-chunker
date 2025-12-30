"""
CLI entry point for Krira_Chunker benchmark.

Usage:
    python -m Krira_Chunker.bench --corpus ./data --report results.json
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from .runner import run_full_benchmark, BenchmarkReport


def generate_sample_corpus(output_dir: str) -> None:
    """Generate sample files for benchmarking."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample markdown with code
    md_content = """# Sample Document

This is a sample markdown document for benchmarking.

## Introduction

Python is a versatile programming language widely used for:

- Web development
- Data science  
- Machine learning
- Automation scripts

## Code Example

Here's a simple Python function:

```python
def calculate_sum(numbers):
    \"\"\"Calculate the sum of a list of numbers.\"\"\"
    total = 0
    for n in numbers:
        total += n
    return total

def main():
    nums = [1, 2, 3, 4, 5]
    result = calculate_sum(nums)
    print(f"Sum: {result}")
```

## Data Table

| Name | Value | Description |
|------|-------|-------------|
| Alpha | 100 | First item |
| Beta | 200 | Second item |
| Gamma | 300 | Third item |

## Conclusion

""" + ("This is additional text to make the document longer. " * 50)
    
    with open(os.path.join(output_dir, "sample.md"), "w") as f:
        f.write(md_content)
    
    # Sample plain text
    txt_content = ("This is a sample text file. " * 200)
    with open(os.path.join(output_dir, "sample.txt"), "w") as f:
        f.write(txt_content)
    
    # Sample JSONL
    import json as json_module
    jsonl_lines = []
    for i in range(50):
        jsonl_lines.append(json_module.dumps({
            "id": i,
            "name": f"Item {i}",
            "description": f"This is the description for item number {i}.",
            "value": i * 10
        }))
    with open(os.path.join(output_dir, "sample.jsonl"), "w") as f:
        f.write("\n".join(jsonl_lines))
    
    # Sample CSV
    csv_content = "id,name,email,description\n"
    for i in range(100):
        csv_content += f"{i},User {i},user{i}@example.com,Description for user {i}\n"
    with open(os.path.join(output_dir, "sample.csv"), "w") as f:
        f.write(csv_content)


def print_summary(report: BenchmarkReport) -> None:
    """Print a summary of benchmark results."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    print(f"\nSystem: {report.system_info['platform']}")
    print(f"Python: {report.system_info['python_version']}")
    print(f"Timestamp: {report.timestamp}")
    
    print(f"\nKrira Config: chunk_size={report.krira_config.get('max_chars', 'N/A')}, "
          f"overlap={report.krira_config.get('overlap_chars', 'N/A')}, "
          f"strategy={report.krira_config.get('chunk_strategy', 'N/A')}")
    
    summary = report.summary
    print(f"\nFiles benchmarked: {summary.get('total_files', 0)}")
    print(f"URLs benchmarked: {summary.get('total_urls', 0)}")
    
    print("\n--- Results by Library ---")
    
    if summary.get("krira_runs", 0) > 0:
        print(f"\nKrira_Chunker ({summary['krira_runs']} runs):")
        print(f"  Avg chunks/sec: {summary.get('krira_avg_chunks_per_s', 0):,.1f}")
        print(f"  Avg MB/sec: {summary.get('krira_avg_mb_per_s', 0):,.2f}")
        print("  Mode: STREAMING (memory efficient)")
    
    if summary.get("langchain_runs", 0) > 0:
        print(f"\nLangChain ({summary['langchain_runs']} runs):")
        print(f"  Avg chunks/sec: {summary.get('langchain_avg_chunks_per_s', 0):,.1f}")
        print("  Mode: Full text load (not streaming)")
    else:
        print("\nLangChain: SKIPPED (not installed)")
    
    if summary.get("llamaindex_runs", 0) > 0:
        print(f"\nLlamaIndex ({summary['llamaindex_runs']} runs):")
        print(f"  Avg chunks/sec: {summary.get('llamaindex_avg_chunks_per_s', 0):,.1f}")
        print("  Mode: Full text load (not streaming)")
    else:
        print("\nLlamaIndex: SKIPPED (not installed)")
    
    # Quality comparison
    print("\n--- Quality Metrics (Krira) ---")
    krira_results = [r for r in report.results if r["library"] == "krira_chunker" and not r.get("skipped")]
    if krira_results:
        avg_codeblock_break = sum(r["codeblock_break_rate"] for r in krira_results) / len(krira_results)
        avg_sentence_break = sum(r["sentence_break_rate"] for r in krira_results) / len(krira_results)
        avg_chunk_len = sum(r["avg_chunk_len_chars"] for r in krira_results) / len(krira_results)
        total_empty = sum(r["empty_chunk_count"] for r in krira_results)
        total_very_large = sum(r["very_large_chunk_count"] for r in krira_results)
        
        print(f"  Avg chunk length: {avg_chunk_len:,.0f} chars")
        print(f"  Code block break rate: {avg_codeblock_break:.1%}")
        print(f"  Sentence break rate: {avg_sentence_break:.1%}")
        print(f"  Empty chunks: {total_empty}")
        print(f"  Very large chunks (>2x target): {total_very_large}")
    
    print("\n" + "=" * 70)


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Krira_Chunker Benchmark - Compare chunking performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m Krira_Chunker.bench --corpus ./data --report results.json
  python -m Krira_Chunker.bench --corpus document.pdf --verbose
  python -m Krira_Chunker.bench --generate-sample --corpus ./bench_data
        """
    )
    
    parser.add_argument(
        "--corpus",
        type=str,
        default=".",
        help="Path to corpus directory or single file (default: current directory)"
    )
    
    parser.add_argument(
        "--report",
        type=str,
        default="bench_report.json",
        help="Output JSON report path (default: bench_report.json)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Target chunk size in chars (default: 2000)"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap in chars (default: 200)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress"
    )
    
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Generate sample benchmark files in corpus directory"
    )
    
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip printing summary to console"
    )
    
    args = parser.parse_args()
    
    # Generate sample files if requested
    if args.generate_sample:
        print(f"Generating sample files in: {args.corpus}")
        generate_sample_corpus(args.corpus)
        print("Sample files generated successfully.")
        if not args.verbose:
            return 0
    
    # Check corpus exists
    if not os.path.exists(args.corpus):
        print(f"Error: Corpus path does not exist: {args.corpus}", file=sys.stderr)
        return 1
    
    print("=" * 70)
    print("KRIRA_CHUNKER BENCHMARK")
    print("=" * 70)
    print(f"\nCorpus: {args.corpus}")
    print(f"Chunk size: {args.chunk_size}, Overlap: {args.chunk_overlap}")
    print(f"Report: {args.report}")
    
    # Run benchmark
    try:
        report = run_full_benchmark(
            corpus_path=args.corpus,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"\nError running benchmark: {e}", file=sys.stderr)
        return 1
    
    # Write report
    try:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        print(f"\nReport written to: {args.report}")
    except Exception as e:
        print(f"\nError writing report: {e}", file=sys.stderr)
        return 1
    
    # Print summary
    if not args.no_summary:
        print_summary(report)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
