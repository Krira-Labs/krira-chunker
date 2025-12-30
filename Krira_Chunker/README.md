# Krira Chunker

Production-grade document chunking library for RAG (Retrieval-Augmented Generation) applications.

## Features

- **Hybrid Boundary-Aware Chunking**: Avoids splitting sentences, code blocks, and tables
- **Streaming-First**: Memory-efficient processing for large files
- **Multi-Format Support**: PDF, DOCX, CSV, XLSX, JSON/JSONL, XML, URLs, Markdown, Text
- **Secure by Default**: SSRF protection, file size limits, zip-slip prevention
- **Deterministic**: Stable chunk IDs and consistent ordering
- **Optional Dependencies**: Install only what you need

## Installation

```bash
# Core installation (no heavy dependencies)
pip install krira-chunker

# With specific format support
pip install krira-chunker[pdf]      # PDF support
pip install krira-chunker[url]      # URL fetching
pip install krira-chunker[csv]      # CSV processing
pip install krira-chunker[xlsx]     # Excel support
pip install krira-chunker[all]      # Everything
```

## Quick Start

### Simple Usage

```python
from Krira_Chunker import iter_chunks_auto, ChunkConfig

cfg = ChunkConfig(
    max_chars=2000,
    overlap_chars=200,
    chunk_strategy="hybrid"
)

for chunk in iter_chunks_auto("document.pdf", cfg):
    print(chunk["text"][:100])
    print(chunk["metadata"])
```

### Class-Based API

```python
from Krira_Chunker.PDFChunker import PDFChunker
from Krira_Chunker.URLChunker import URLChunker

# PDF processing
pdf = PDFChunker(cfg)
for chunk in pdf.chunk_file("report.pdf"):
    print(chunk["text"])

# URL processing (with SSRF protection)
url_chunker = URLChunker(cfg, allow_private=False)
for chunk in url_chunker.chunk_url("https://example.com"):
    print(chunk["text"])
```

### Facade Pattern

```python
from Krira_Chunker import KriraChunker

engine = KriraChunker(cfg)
chunks = list(engine.process("any_file.pdf"))
```

## Configuration

```python
from Krira_Chunker import ChunkConfig

cfg = ChunkConfig(
    # Chunk sizing
    max_chars=2200,
    overlap_chars=250,
    
    # Token-based (optional)
    use_tokens=False,
    max_tokens=512,
    overlap_tokens=64,
    
    # Strategy
    chunk_strategy="hybrid",  # fixed, sentence, markdown, hybrid
    
    # Preservation
    preserve_code_blocks=True,
    preserve_tables=True,
    preserve_lists=True,
    
    # Security
    url_allow_private=False,
    url_max_bytes=8_000_000,
    security_max_file_bytes=50_000_000,
)
```

## Chunk Output Format

```python
{
    "id": "a1b2c3d4...",  # Stable MD5 hash
    "text": "Chunk content...",
    "metadata": {
        "source": "document.pdf",
        "source_type": "pdf",
        "chunk_index": 0,
        "config_hash": "abc123...",
        "boundary_type": "paragraph",  # heading/paragraph/sentence/hard
        "page": 1,  # PDF
        "row_start": 5, "row_end": 10,  # CSV/XLSX
        # ... more format-specific metadata
    }
}
```

## Chunking Strategies

### Hybrid (Recommended)
Best quality, avoids splitting:
- Code blocks (fenced with ```)
- Tables (markdown pipe tables)
- Sentences (when possible)
- Lists

### Fixed
Simple fixed-size chunks with overlap.

### Sentence
Respects sentence boundaries.

### Markdown
Respects markdown structure (headings, paragraphs).

## Security Features

- **SSRF Protection**: Blocks private IPs, localhost, .local domains
- **Content-Type Validation**: Allowlist for URL content types
- **File Size Limits**: Configurable max file size
- **Zip-Slip Prevention**: Safe DOCX extraction

## Benchmarking

```bash
python -m Krira_Chunker.bench --corpus ./data --report results.json
```

Compares against LangChain and LlamaIndex when installed.

## Testing

```bash
pip install krira-chunker[test]
pytest
```

## License

MIT
