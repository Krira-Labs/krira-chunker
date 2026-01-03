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
```

---

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
```

---

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
```

---

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

---

## License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

<p align="center">
  Developed by <b>Krira Labs</b>
</p>
