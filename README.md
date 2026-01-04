# Krira Augment ⚡🦀

**The High-Performance Rust Chunking Engine for RAG Pipelines**

[![PyPI version](https://badge.fury.io/py/krira-augment.svg)](https://badge.fury.io/py/krira-augment)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Built_with-Rust-orange)](https://www.rust-lang.org/)

**Krira Augment** is a production-grade Python library backed by a highly optimized Rust core. It is designed to replace slow, memory-intensive preprocessing steps in large-scale Retrieval Augmented Generation (RAG) systems.

It processes gigabytes of raw unstructured data (CSV, PDF, DOCX, JSON, URLs, etc.) into high-quality, clean chunks in seconds—utilizing **zero-copy memory mapping** and **segment-based parallel CPU execution**.

---

## 🚀 Performance Benchmarks

Benchmarks run on a standard 8-core machine (M2 Air equivalent).

| Dataset Size | Legacy (LangChain/Pandas) | Krira V2 (Rust Core) | Speedup |
| :--- | :--- | :--- | :--- |
| **100 MB** | ~45 sec | **~0.8 sec** | **56x** 🚀 |
| **1 GB** | ~8.0 min | **~12.0 sec** | **40x** 🚀 |
| **5.28 GB** | *Crash / OOM* | **~58.0 sec** | **Stable** ✅ |
| **10 GB+** | *N/A* | **~2.1 min** | **Scalable** ✅ |

> **Note:** Krira uses a segment-based parallel strategy. It divides large files into 32MB chunks to ensure CPU saturation while maintaining a strict, low memory footprint.

---

## 📦 Installation

```bash
# Basic installation
pip install krira-augment

# Install with optional multi-format support
pip install "krira-augment[all]"
```

*Requirements: Python 3.8+*

---

## 🛠️ Usage

### 1. Quick Start
The `process` method is now fully flexible. If no `output_path` is provided, Krira automatically generates one based on the input filename.

```python
from krira_augment import Pipeline

# Initialize the pipeline
pipeline = Pipeline()

# Process any file (CSV, JSONL, TXT, XML, etc.)
stats = pipeline.process(input_path="my_data.csv")

# Print the beautiful formatted output
print(stats)
```

**Output:**
```
============================================================
✅ KRIRA AUGMENT - Processing Complete
============================================================
📊 Chunks Created:  1,247
⏱️  Execution Time:  0.85 seconds
🚀 Throughput:      118.24 MB/s
📁 Output File:     my_data_processed.jsonl
============================================================

📝 Preview (Top 3 Chunks):
------------------------------------------------------------
[1] This is the first chunk of processed text from your file...
[2] Here is the second chunk with more content from the data...
[3] And the third chunk showing a sample of the output...
------------------------------------------------------------
```

You can also access individual stats:
```python
print(f"Chunks: {stats.chunks_created}")
print(f"Time: {stats.execution_time:.2f}s")
print(f"Output: {stats.output_file}")
print(f"Preview: {stats.preview_chunks}")
```

### 2. Multi-Format Support

Krira Augment handles the heavy lifting of extracting text from complex formats and passing it to the high-speed Rust core.

---

#### 📄 **CSV Files**
Process CSV files directly. Each row is treated as a separate text unit for chunking.

```python
from krira_augment import Pipeline

pipeline = Pipeline()

# Process CSV - output auto-generated as 'data_processed.jsonl'
stats = pipeline.process("data.csv")
print(f"Output: {stats.output_file}")

# Or specify custom output path
stats = pipeline.process("data.csv", output_path="chunked_data.jsonl")
```

---

#### 📕 **PDF Documents**
Extract text from PDF files page by page. Requires: `pip install pdfplumber`

```python
from krira_augment import Pipeline

pipeline = Pipeline()

# Process PDF - extracts text from all pages
stats = pipeline.process("document.pdf")
print(f"Output: {stats.output_file}")

# With custom output
stats = pipeline.process("report.pdf", output_path="report_chunks.jsonl")
```

---

#### 📗 **Excel Spreadsheets (.xlsx)**
Process Excel files with automatic sheet and row handling. Requires: `pip install openpyxl`

```python
from krira_augment import Pipeline

pipeline = Pipeline()

# Process Excel - each row becomes a text chunk
stats = pipeline.process("spreadsheet.xlsx")
print(f"Output: {stats.output_file}")

# With custom output
stats = pipeline.process("data.xlsx", output_path="excel_chunks.jsonl")
```

---

#### 📘 **Word Documents (.docx)**
Extract paragraphs from Word documents. Requires: `pip install python-docx`

```python
from krira_augment import Pipeline

pipeline = Pipeline()

# Process DOCX - each paragraph becomes a text unit
stats = pipeline.process("document.docx")
print(f"Output: {stats.output_file}")

# With custom output
stats = pipeline.process("contract.docx", output_path="contract_chunks.jsonl")
```

---

#### 🌐 **Website URLs**
Fetch and process web pages. Requires: `pip install requests beautifulsoup4`

```python
from krira_augment import Pipeline

pipeline = Pipeline()

# Process URL - auto-generates output filename from URL hash
stats = pipeline.process("https://example.com/docs")
print(f"Output: {stats.output_file}")

# With custom output
stats = pipeline.process("https://example.com/article", output_path="article_chunks.jsonl")
```

---

#### 📙 **XML Files**
Process XML files by extracting text from each child element.

```python
from krira_augment import Pipeline

pipeline = Pipeline()

# Process XML - each child element text becomes a chunk
stats = pipeline.process("data.xml")
print(f"Output: {stats.output_file}")

# With custom output
stats = pipeline.process("config.xml", output_path="xml_chunks.jsonl")
```

---

#### 📋 **JSON Files**
Process JSON arrays or objects by flattening to JSONL.

```python
from krira_augment import Pipeline

pipeline = Pipeline()

# Process JSON - arrays are flattened, objects are chunked
stats = pipeline.process("data.json")
print(f"Output: {stats.output_file}")

# With custom output
stats = pipeline.process("config.json", output_path="json_chunks.jsonl")
```

---

#### 📝 **JSONL Files**
Process JSONL files directly (native format for Rust core).

```python
from krira_augment import Pipeline

pipeline = Pipeline()

# Process JSONL - direct pass-through to Rust core
stats = pipeline.process("data.jsonl")
print(f"Output: {stats.output_file}")

# With custom output
stats = pipeline.process("logs.jsonl", output_path="processed_logs.jsonl")
```

---

#### 📃 **Text Files (.txt)**
Process plain text files line by line.

```python
from krira_augment import Pipeline

pipeline = Pipeline()

# Process TXT - each line is processed
stats = pipeline.process("notes.txt")
print(f"Output: {stats.output_file}")

# With custom output
stats = pipeline.process("corpus.txt", output_path="corpus_chunks.jsonl")
```

### 3. Advanced Configuration (Professional)
For production RAG, you need fine-grained control over chunking strategies and data cleaning.

```python
from krira_augment import Pipeline, PipelineConfig, SplitStrategy

# Define a robust configuration
config = PipelineConfig(
    chunk_size=512,               # Target characters per chunk
    strategy=SplitStrategy.SMART, # Respects sentence/paragraph boundaries
    clean_html=True,              # Remove <div>, <br>, etc.
    clean_unicode=True,           # Normalize whitespace and emojis
)

pipeline = Pipeline(config=config)

# Execute
result = pipeline.process("large_corpus.csv", output_path="custom_output.jsonl")

# Beautiful formatted output
print(result)

# Or access individual stats
print(f"Chunks Created: {result.chunks_created}")
print(f"Execution Time: {result.execution_time:.2f}s")
print(f"Throughput: {result.mb_per_second:.2f} MB/s")
print(f"Preview: {result.preview_chunks[:2]}")  # First 2 chunks
```

---

## 🏗️ Architecture

Krira differs from standard Python loaders by offloading the entire ETL process to a compiled Rust binary with industrial-strength safety.

1.  **Memory Mapping (mmap):** Files are mapped directly from disk. No loading massive files into Python RAM.
2.  **Segmented Parallelism:** The file is sliced into 32MB segments processed via the Rayon work-stealing scheduler.
3.  **Bounded Backpressure:** A 1024-item bounded MPSC channel manages data flow from processing threads to the disk writer, preventing runaway memory growth even if processing speed exceeds disk I/O.
4.  **Serde Serialization:** Chunks are serialized to JSONL directly on Rust threads, bypassing the Python GIL.

---

## 🤝 Integration Example

```python
import json

def stream_chunks(jsonl_path):
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

# Usage
for chunk in stream_chunks("my_data_processed.jsonl"):
    # Send to Vector DB or OpenAI Embedding API
    pass
```

---

## 🧑‍💻 Development

1.  **Clone the repo**
2.  **Install Maturin**
    ```bash
    pip install maturin
    ```
3.  **Build and Install locally**
    ```bash
    python -m build
    pip install dist/*.whl --force-reinstall
    ```

---

## License

MIT License. (c) 2024 Krira Labs.
