<<<<<<< HEAD
# Krira Chunker

[![PyPI version](https://img.shields.io/pypi/v/krira-chunker.svg)](https://pypi.org/project/krira-chunker/)
[![Python versions](https://img.shields.io/pypi/pyversions/krira-chunker.svg)](https://pypi.org/project/krira-chunker/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Krira Chunker** is a high-performance, production-grade document chunking library specifically engineered for Retrieval-Augmented Generation (RAG) pipelines. It prioritizes semantic integrity, memory efficiency, and security, ensuring your LLM applications receive contextually coherent information.

---

## Key Highlights

- **Hybrid Boundary-Aware Chunking**: Intelligently avoids splitting critical structures like code blocks, tables, and sentences.
- **Streaming-First Architecture**: Process gigabyte-scale datasets with minimal memory footprint through generator-based ingestion.
- **Enterprise Security**: Built-in SSRF protection, safe file extraction (Zip-Slip prevention), and configurable resource limits.
- **Deterministic Output**: Stable MD5-based UUIDs for chunks, enabling efficient upserts and caching in vector databases.
- **Zero-Config Ingestion**: Automatic format detection for local files and remote URLs.
- **Modular Design**: Lightweight core with optional extras to keep your environment clean.

---

## Installation

```bash
# Core installation (High-performance text chunking)
pip install krira-chunker

# Install with specific format support
pip install krira-chunker[pdf]      # PDF extraction
pip install krira-chunker[csv]      # Polars-powered CSV streaming
pip install krira-chunker[xlsx]     # Excel/Spreadsheet support
pip install krira-chunker[url]      # Web scraping with SSRF protection
pip install krira-chunker[tokens]   # Tiktoken-based token counting

# Install everything
pip install krira-chunker[all]
=======
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
>>>>>>> c85f172 (first commit)
```

---

<<<<<<< HEAD
## Supported Formats

| Format | Extension | Engine/Method | Batch Support |
| :--- | :--- | :--- | :---: |
| **PDF** | `.pdf` | Multi-layer text extraction | Yes |
| **Word** | `.docx` | Structural XML parsing | Yes |
| **Excel** | `.xlsx`, `.xls` | Sequential row streaming | Yes |
| **CSV** | `.csv` | High-speed Polars engine | Yes |
| **JSON** | `.json`, `.jsonl` | Ijson streaming parser | Yes |
| **XML** | `.xml` | Incremental tree walking | Yes |
| **Web** | `http://`, `https://` | Trafilatura / Clean HTML | Yes |
| **Markdown**| `.md`, `.markdown` | Semantic structure aware | Yes |
| **Text** | `.txt`, `.text` | Prose-optimized splitting | Yes |

---

## Quick Start

### The Magic Ingestor
One function to rule them all. Detects format, applies strategy, and yields chunks.

```python
from Krira_Chunker import iter_chunks_auto, ChunkConfig

# Configure for your specific LLM window
cfg = ChunkConfig(
    max_chars=1500,
    overlap_chars=150,
    chunk_strategy="hybrid"
)

# Process PDF, CSV, or a URL seamlessly
for chunk in iter_chunks_auto("knowledge_base.pdf", cfg):
    print(f"ID: {chunk['id']}")
    print(f"Content: {chunk['text'][:100]}...")
    print(f"Metadata: {chunk['metadata']}")
=======
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
>>>>>>> c85f172 (first commit)
```

---

<<<<<<< HEAD
## Detailed Usage by Format

### PDF Documents
Uses `pypdf` for layered text extraction and keeps track of page numbers.
```python
from Krira_Chunker import iter_chunks_from_pdf, ChunkConfig

cfg = ChunkConfig(max_chars=2000)
for chunk in iter_chunks_from_pdf("report.pdf", cfg):
    print(f"Page: {chunk['metadata']['page']}")
    print(chunk['text'])
```

### Word Documents (DOCX)
Safe XML-based parsing that respects paragraphs and prevents Zip-Slip vulnerabilities.
```python
from Krira_Chunker import iter_chunks_from_docx, ChunkConfig

cfg = ChunkConfig(chunk_strategy="hybrid")
for chunk in iter_chunks_from_docx("contract.docx", cfg):
    print(chunk['text'])
```

### CSV Files
Powered by **Polars** for high-speed streaming. Ideal for massive datasets.
```python
from Krira_Chunker import iter_chunks_from_csv, ChunkConfig

cfg = ChunkConfig(rows_per_chunk=100) # Chunk by number of rows
for chunk in iter_chunks_from_csv("data.csv", cfg):
    print(f"Rows: {chunk['metadata']['row_start']} to {chunk['metadata']['row_end']}")
    print(chunk['text'])
```

### Excel Spreadsheets (XLSX/XLS)
Memory-efficient row-by-row streaming for multi-sheet workbooks.
```python
from Krira_Chunker import iter_chunks_from_xlsx, ChunkConfig

for chunk in iter_chunks_from_xlsx("budget.xlsx", cfg):
    print(f"Sheet: {chunk['metadata']['sheet_name']}")
    print(chunk['text'])
```

### JSON / JSONL
Uses `ijson` for incremental parsing, allowing you to process multi-GB JSON files without loading them into memory.
```python
from Krira_Chunker import iter_chunks_from_json, ChunkConfig

for chunk in iter_chunks_from_json("events.jsonl", cfg):
    print(chunk['text'])
```

### XML Data
Incremental tree walking for structured data extraction.
```python
from Krira_Chunker import iter_chunks_from_xml, ChunkConfig

for chunk in iter_chunks_from_xml("data.xml", cfg):
    print(chunk['text'])
```

### Web Content (URLs)
Fetches and cleans HTML using `Trafilatura`, with built-in SSRF protection to block private IP ranges.
```python
from Krira_Chunker import iter_chunks_from_url, ChunkConfig

cfg = ChunkConfig(url_allow_private=False) # Protection ON
for chunk in iter_chunks_from_url("https://docs.example.com", cfg):
    print(chunk['text'])
```

### Markdown & Text
Intelligently respects headers (#), code blocks (```), and list items.
```python
from Krira_Chunker import iter_chunks_from_markdown, iter_chunks_from_text, ChunkConfig

cfg = ChunkConfig(chunk_strategy="hybrid", preserve_code_blocks=True)

# For Markdown
for chunk in iter_chunks_from_markdown("README.md", cfg):
    print(chunk['text'])

# For Plain Text
for chunk in iter_chunks_from_text("notes.txt", cfg):
    print(chunk['text'])
=======
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
>>>>>>> c85f172 (first commit)
```

---

<<<<<<< HEAD
## High-Throughput Batching

Stream chunks directly to your vector database in optimized batches.

```python
from Krira_Chunker import stream_chunks_to_sink

def upsert_to_db(batch):
    # db.upsert(batch)
    print(f"Upserting {len(batch)} chunks...")

total = stream_chunks_to_sink(
    input_path="knowledge_base.pdf",
    sink=upsert_to_db,
    batch_size=100
)
```

---

## Advanced Configuration

```python
from Krira_Chunker import ChunkConfig

cfg = ChunkConfig(
    # --- Sizing ---
    max_chars=2000,
    overlap_chars=200,
    min_chars=50,          # Filter out noise/empty chunks
    
    # --- Token-Based (Optional) ---
    use_tokens=True,       # Requires [tokens] extra
    max_tokens=512,
    
    # --- Strategies ---
    chunk_strategy="hybrid", # "hybrid", "fixed", "sentence", "markdown"
    
    # --- Preservation Flags ---
    preserve_code_blocks=True, 
    preserve_tables=True,
    preserve_lists=True,
    
    # --- Performance ---
    sink_batch_size=256,
    csv_batch_rows=50000,
)
```

---

## Performance Benchmark

Internal benchmarks against standard RAG splitters (Measured on 1GB mixed-format corpus):

| Metric | Krira Chunker | Generic Splitters |
| :--- | :---: | :---: |
| **Throughput (MB/s)** | **12.4** | 4.1 |
| **Memory Peak (MB)** | **42** | 210 |
| **Code Block Breakage** | **0%** | 18% |
=======
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
>>>>>>> c85f172 (first commit)

---

## License

<<<<<<< HEAD
Distributed under the **MIT License**. See `LICENSE` for more information.

---

<p align="center">
  Developed by <b>Krira Labs</b>
</p>
=======
MIT License. (c) 2024 Krira Labs.
>>>>>>> c85f172 (first commit)
