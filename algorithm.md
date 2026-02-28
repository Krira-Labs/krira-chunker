# Krira Chunker — Algorithm Documentation

> **Version:** 2.1.13  
> **Architecture:** Hybrid Rust + Python  
> **Purpose:** Production-grade document chunking for RAG (Retrieval-Augmented Generation) pipelines

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Text Chunking Algorithms](#2-text-chunking-algorithms)
   - 2.1 [Hybrid 3-Tier Chunking](#21-hybrid-3-tier-chunking-rust-core)
   - 2.2 [Sliding Window with Overlap](#22-sliding-window-chunking-with-overlap)
   - 2.3 [Chunking Strategies](#23-chunking-strategies)
3. [Text Cleaning Algorithms](#3-text-cleaning-algorithms)
   - 3.1 [Rust Fast Cleaner](#31-rust-fast-cleaner)
   - 3.2 [Python Advanced Cleaner](#32-python-advanced-cleaner)
   - 3.3 [Streaming Cleaner](#33-streaming-cleaner-sliding-window-buffer)
4. [Data Transformation Algorithms](#4-data-transformation-algorithms)
   - 4.1 [CSV → Markdown Table](#41-csv--markdown-table-conversion)
   - 4.2 [JSON → Markdown Flattening](#42-json--markdown-recursive-flattening)
   - 4.3 [Row Transformation](#43-row-transformation)
5. [Parallel Processing & I/O](#5-parallel-processing--io-algorithms)
   - 5.1 [Memory-Mapped I/O](#51-memory-mapped-file-io-zero-copy)
   - 5.2 [Rayon Parallel Segments](#52-rayon-parallel-segment-processing)
   - 5.3 [Newline-Aligned Splitting](#53-newline-aligned-segment-splitting)
   - 5.4 [Producer-Consumer Streaming](#54-producer-consumer-streaming)
6. [Format Conversion Pipeline](#6-format-conversion-pipeline)
7. [Hashing & ID Generation](#7-hashing--id-generation)
8. [Security Algorithms](#8-security-algorithms)
9. [Performance Optimizations](#9-performance-optimizations)
10. [Complexity Summary](#10-complexity-summary)

---

## 1. Architecture Overview

The library processes documents through a **Clean → Transform → Chunk** pipeline:

```
┌──────────────────────────────────────────────────────────────┐
│                        INPUT SOURCES                         │
│   CSV │ XLSX │ PDF │ DOCX │ JSON │ XML │ URL │ TXT │ JSONL  │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  FORMAT CONVERSION  │  (Python Layer)
              │  All → JSONL / Text │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │    DATA CLEANING    │  (Rust + Python)
              │  Regex, Unicode,    │
              │  PII Redaction      │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  DATA TRANSFORMATION│  (Rust + Python)
              │  CSV → Markdown     │
              │  JSON → Markdown    │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   CHUNKING ENGINE   │  (Rust Core)
              │  Hybrid / Sliding   │
              │  Window / Fixed     │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │       OUTPUT        │
              │  JSONL File or      │
              │  Streaming Iterator │
              └─────────────────────┘
```

**Two execution modes:**

| Mode | Description | Memory Usage |
|------|-------------|-------------|
| **File-based** (`process()`) | Parallel processing via Rayon, writes JSONL to disk | O(segment_size × cores) |
| **Streaming** (`process_stream()`) | Background Rust thread → bounded channel → Python iterator | O(1) constant |

---

## 2. Text Chunking Algorithms

### 2.1 Hybrid 3-Tier Chunking (Rust Core)

**Source:** `src/chunker.rs` → `RustChunker::chunk()`

The primary algorithm uses a **greedy accumulation** strategy with 3 hierarchical split tiers:

```
┌───────────────────────────────────────────────────────┐
│                   INPUT TEXT                           │
│  "Paragraph 1...\n\nParagraph 2...\n\nParagraph 3..." │
└───────────────────────┬───────────────────────────────┘
                        │
                        ▼
           ┌────────────────────────┐
           │  TIER 1: PARAGRAPH     │
           │  Split on "\n\n"       │
           │  Accumulate greedily   │
           └────────────┬───────────┘
                        │
              Paragraph > max_chars?
                   │          │
                  NO         YES
                   │          │
                   ▼          ▼
             Keep in     ┌────────────────────┐
             buffer      │  TIER 2: SENTENCE  │
                         │  Split on ". "     │
                         │  Accumulate        │
                         └────────┬───────────┘
                                  │
                        Sentence > max_chars?
                             │          │
                            NO         YES
                             │          │
                             ▼          ▼
                       Keep in     ┌────────────────────┐
                       buffer      │  TIER 3: CHARACTER  │
                                   │  Hard split at      │
                                   │  max_chars boundary  │
                                   └─────────────────────┘
```

**Pseudocode:**

```python
def chunk(text, max_chars):
    chunks = []
    buffer = ""

    for paragraph in text.split("\n\n"):

        # TIER 1: Try to fit paragraph into current buffer
        if len(buffer) + len(paragraph) + 2 <= max_chars:
            buffer += "\n\n" + paragraph
        else:
            # Flush buffer if non-empty
            if buffer:
                chunks.append(buffer)
                buffer = ""

            # TIER 2: Paragraph too large → split by sentences
            if len(paragraph) > max_chars:
                for sentence in paragraph.split(". "):

                    if len(buffer) + len(sentence) + 2 <= max_chars:
                        buffer += ". " + sentence
                    else:
                        if buffer:
                            chunks.append(buffer)
                            buffer = ""

                        # TIER 3: Sentence too large → hard character split
                        if len(sentence) > max_chars:
                            for i in range(0, len(sentence), max_chars):
                                chunks.append(sentence[i : i + max_chars])
                        else:
                            buffer = sentence
            else:
                buffer = paragraph

    if buffer:
        chunks.append(buffer)

    return chunks
```

**Key Properties:**

| Property | Value |
|----------|-------|
| Time complexity | O(n) — single pass |
| Space complexity | O(max_chars) — buffer bounded |
| Splitting priority | Paragraph > Sentence > Character |
| Accumulation | Greedy (fills buffer before flushing) |
| Memory allocation | Pre-allocated `String::with_capacity(max_chars)` |

---

### 2.2 Sliding Window Chunking with Overlap

**Source:** `src/pipeline.rs` → `KriraPipeline::chunk_text()`

Used for texts exceeding `max_size`. Provides **overlap** between consecutive chunks to prevent information loss at boundaries.

```
Algorithm:
──────────

Input: text, max_size, overlap_size, min_chars
Step = max_size - overlap_size

Position: 0
While position < len(text):
    1. Set candidate_end = position + max_size
    2. Clamp to text length
    3. If not at end, search BACKWARD for natural boundary:
       a. Try: rfind('\n') → split at newline     (PREFERRED)
       b. Try: rfind(' ')  → split at word break   (FALLBACK)
       c. Else: split at candidate_end              (HARD SPLIT)
    4. Extract chunk_text = text[position..split_pos].trim()
    5. If len(chunk_text) >= min_chars → emit chunk
    6. Advance: position += step (or split_pos if it jumped ahead)
```

**Visual representation:**

```
Text:   |==============================================|
         ├── Chunk 1 ──────┤
                      ├─overlap─┤
                      ├── Chunk 2 ──────┤
                                   ├─overlap─┤
                                   ├── Chunk 3 ──────┤
```

**Configuration:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_chars` | 2200 | Maximum chunk size in characters |
| `overlap_chars` | 250 | Overlap between consecutive chunks |
| `min_chars` | 30 | Minimum chunk size (filter out tiny fragments) |
| `use_tokens` | false | Switch to token-based sizing |
| `max_tokens` | 512 | Maximum chunk size in tokens |
| `overlap_tokens` | 64 | Overlap in tokens |

---

### 2.3 Chunking Strategies

**Source:** `src/config.rs` → `ChunkStrategy`

| Strategy | Algorithm | Best For |
|----------|-----------|----------|
| `fixed` | Hard character splits at exact `max_chars` boundaries | Uniform chunk sizes for embedding models |
| `sentence` | Split at sentence boundaries (`. `, `? `, `! `) | Narrative/prose text |
| `markdown` | Aware of `#` headers, ` ``` ` code blocks, `|` tables, `- ` lists | Markdown/documentation |
| `hybrid` | **Default.** 3-tier: Paragraph → Sentence → Character | General purpose, best quality |
| `llm` | Reserved for future LLM-assisted semantic splitting | Semantic coherence |

---

## 3. Text Cleaning Algorithms

### 3.1 Rust Fast Cleaner

**Source:** `src/cleaning.rs` → `RustCleaner::clean()`

Uses `lazy_static!` pre-compiled regexes for maximum performance:

```
Input Text
    │
    ▼
┌──────────────────────────────────────────────────────┐
│ Step 1: Remove Headers                               │
│   Regex: (?i)Page \d+ of \d+                         │
│   Removes: "Page 1 of 10", "page 23 of 100"         │
├──────────────────────────────────────────────────────┤
│ Step 2: Remove Footers                               │
│   Regex: (?i)© \d{4}                                 │
│   Removes: "© 2024", "© 1999"                        │
├──────────────────────────────────────────────────────┤
│ Step 3: Normalize Whitespace                         │
│   Regex: [ \t]+                                      │
│   Replaces multiple spaces/tabs → single space       │
├──────────────────────────────────────────────────────┤
│ Step 4: Trim                                         │
│   Strip leading and trailing whitespace              │
└──────────────────────────────────────────────────────┘
    │
    ▼
Output (cleaned text)
```

**Efficiency:** Uses `Cow<str>` (Copy-on-Write) — if no regex matches, **zero allocation** occurs.

---

### 3.2 Python Advanced Cleaner

**Source:** `python/krira_augment/_python/cleaning.py` → `DataCleaner`

A more comprehensive, configurable cleaning pipeline:

```
Step 1: Unicode Normalization (NFKC)
   │  unicodedata.normalize('NFKC', text)
   │  Fixes: \u00a0 → space, ﬀ → ff, ² → 2
   ▼
Step 2: Remove Headers (3 patterns)
   │  "Page X of Y"   →  removed
   │  "Page X"         →  removed
   │  "X / Y"          →  removed
   ▼
Step 3: Remove Footers (6 patterns)
   │  "© YYYY Company"        →  removed
   │  "Copyright YYYY"        →  removed
   │  "Confidential"          →  removed
   │  "All Rights Reserved"   →  removed
   │  "CONFIDENTIAL"          →  removed
   │  "PROPRIETARY..."        →  removed
   ▼
Step 4: Custom Pattern Removal
   │  User-supplied regex patterns applied in order
   ▼
Step 5: PII Redaction (optional)
   │  email@domain.com     →  <EMAIL>
   │  +1-555-123-4567      →  <PHONE>
   ▼
Step 6: Whitespace Normalization
   │  If preserve_line_breaks:
   │    Multiple spaces → single space (per line)
   │    3+ newlines → double newline
   │  Else:
   │    All whitespace → single space
   ▼
Step 7: Final Trim
   │  .strip()
   ▼
Output
```

**All regex patterns are pre-compiled in `__init__`** for performance.

---

### 3.3 Streaming Cleaner (Sliding Window Buffer)

**Source:** `python/krira_augment/_python/cleaning.py` → `DataCleaner.clean_stream()`

Handles large files without loading everything into memory, using an **overlap buffer** to catch patterns spanning chunk boundaries:

```
Algorithm:
──────────

buffer_size = config.chunk_buffer_size  (default: 10,000 chars)
overlap_size = min(100, buffer_size / 10)

buffer = ""
for chunk in text_stream:
    buffer += chunk

    while len(buffer) >= buffer_size + overlap_size:
        to_process = buffer[:buffer_size]
        cleaned = clean_text(to_process)
        yield cleaned
        buffer = buffer[buffer_size:]   # keep overlap portion

# Process remaining buffer
if buffer:
    yield clean_text(buffer)
```

---

## 4. Data Transformation Algorithms

### 4.1 CSV → Markdown Table Conversion

**Source:** `src/transformation.rs` and `python/krira_augment/_python/transformation.py`

```
Algorithm:
──────────

Input: CSV text, has_header flag

1. PARSE
   Parse CSV using flexible reader (handles irregular row lengths)

2. ANALYZE
   max_cols = max column count across all rows
   effective_cols = min(max_cols, config.max_table_columns)
   truncated = (max_cols > max_table_columns)

3. NORMALIZE
   For each row:
     - Take first effective_cols columns
     - Pad with empty strings if row is shorter
     - Strip newlines from cell contents

4. COMPUTE WIDTHS
   For each column:
     width[i] = max(header_width, max(data_widths), 3)

5. RENDER
   ┌─────────────────────────────────────────┐
   │ *Note: Table truncated from N to M...*  │  ← if truncated
   │ | Col1   | Col2   | Col3   |            │  ← header row
   │ |--------|--------|--------|            │  ← separator
   │ | data1  | data2  | data3  |            │  ← data rows
   │ | data4  | data5  | data6  |            │
   └─────────────────────────────────────────┘
```

**Alternate output:** If `output_format = "plain_text"`:
```
Col1: data1 | Col2: data2 | Col3: data3
Col1: data4 | Col2: data5 | Col3: data6
```

---

### 4.2 JSON → Markdown Recursive Flattening

**Source:** `src/transformation.rs` and `python/krira_augment/_python/transformation.py`

```
Algorithm: Depth-First Recursive Traversal
──────────────────────────────────────────

function format_json_value(value, depth):

    if depth >= max_json_depth:
        return "[...truncated...]"

    match typeof(value):
        null    → "None"
        bool    → "true" / "false"
        number  → string(value)
        string  → value

        array   → Numbered list:
                   "1. item_formatted"
                   "2. item_formatted"
                   For nested objects/arrays: indented sub-section

        object  → Bullet list with bold keys:
                   "- **key1**: value_formatted"
                   "- **key2**:"
                   "    - **nested_key**: value"
```

**Example:**

```json
{
  "user": "Alice",
  "scores": [95, 87, 92],
  "address": {
    "city": "NYC",
    "zip": "10001"
  }
}
```

**→ Markdown Output:**

```markdown
- **user**: Alice
- **scores**:
  1. 95
  2. 87
  3. 92
- **address**:
  - **city**: NYC
  - **zip**: 10001
```

---

### 4.3 Row Transformation

**Source:** `src/transformation.rs` → `DataTransformer::transform_row()`

Converts a key-value row (from CSV/XLSX) into formatted text:

```
Markdown mode:  **Name**: Alice | **Age**: 30 | **City**: NYC
Plain text:     Name: Alice | Age: 30 | City: NYC
```

Empty/null values are filtered out before joining.

---

## 5. Parallel Processing & I/O Algorithms

### 5.1 Memory-Mapped File I/O (Zero-Copy)

**Source:** `src/lib.rs` — uses `memmap2` crate

```
Algorithm:
──────────

1. Open file descriptor
2. mmap() → OS maps file pages into virtual memory
3. Cast bytes to &str (UTF-8 validation)
4. Process directly from virtual memory pages
   - No heap allocation
   - OS handles page faults and caching
   - File never fully loaded into RAM
```

**Benefit:** For a 1 GB file, the process uses minimal RSS because only accessed pages are loaded by the OS on demand.

---

### 5.2 Rayon Parallel Segment Processing

**Source:** `src/lib.rs` → `process_file_rust()`

```
Algorithm:
──────────

1. Memory-map the input file

2. Split content into ~32 MB segments (newline-aligned)
   segments = split_into_chunks(content, 32 * 1024 * 1024)

3. Create bounded channel: sync_channel<Vec<ChunkObj>>(128)
   - 128 batch slots max → limits memory to ~6 MB buffer

4. Spawn writer thread:
   - Reads from channel
   - Writes JSONL via BufWriter(64 KB buffer)

5. Parallel processing (Rayon):
   segments.par_iter().for_each(|segment| {
       batch = []
       for line in segment.lines():
           cleaned = RustCleaner::clean(line)
           sub_chunks = chunker.chunk(cleaned)
           batch.extend(sub_chunks)

           if |batch| >= 100:
               channel.send(batch)  // blocks if channel full
               batch = []
       channel.send(remaining_batch)
   })

6. Writer thread finishes and flushes

Architecture Diagram:
─────────────────────

  ┌────────────────────┐
  │  Memory-Mapped File │
  │  (mmap, zero-copy)  │
  └────────┬───────────┘
           │ split into ~32 MB
           ▼
  ┌────────────────────┐
  │ Segment 1 │ Seg 2  │ ... │ Seg N │
  └────┬──────┴───┬────┘     └───┬───┘
       │          │              │
       ▼          ▼              ▼
  ┌─────────┐ ┌─────────┐  ┌─────────┐
  │ Thread 1│ │ Thread 2│  │ Thread N│   ← Rayon thread pool
  │ Clean   │ │ Clean   │  │ Clean   │
  │ Chunk   │ │ Chunk   │  │ Chunk   │
  │ Batch   │ │ Batch   │  │ Batch   │
  └────┬────┘ └────┬────┘  └────┬────┘
       │           │             │
       └───────────┼─────────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  sync_channel(128)  │  ← bounded, backpressure
        │  (batch of 100 each)│
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │    Writer Thread     │
        │  BufWriter (64 KB)   │
        │  → output.jsonl      │
        └─────────────────────┘
```

**GIL Release:** The entire Rust processing block runs inside `py.allow_threads()`, ensuring the Python GIL is released during computation.

---

### 5.3 Newline-Aligned Segment Splitting

**Source:** `src/lib.rs` → `split_into_chunks()`

Ensures no line is split across segments, which would corrupt line-based processing:

```
Algorithm:
──────────

function split_into_chunks(text, target_size):
    chunks = []
    start = 0

    while start < len(text):
        end = start + target_size

        if end >= len(text):
            end = len(text)
        else:
            # Scan FORWARD from 'end' to find next newline
            newline_pos = text[end:].find('\n')
            if newline_pos exists:
                end = end + newline_pos + 1  # include the newline
            else:
                end = len(text)  # last segment

        chunks.append(text[start:end])
        start = end

    return chunks
```

**Key:** Scans **forward** (not backward) to find the next newline, ensuring each segment contains only complete lines.

---

### 5.4 Producer-Consumer Streaming

**Source:** `src/lib.rs` → `process_stream()` and `ChunkIterator`

For streaming mode, a background Rust thread pipes chunks to a Python iterator:

```
Architecture:
─────────────

  ┌─────────────────────────┐
  │   Background Rust Thread │
  │                          │
  │   1. mmap file           │
  │   2. split segments      │
  │   3. FOR each segment:   │
  │      FOR each line:      │
  │        clean → chunk     │
  │        → sender.send()   │──── sync_channel(100) ────┐
  │                          │     (backpressure)         │
  │   4. sender dropped      │                            │
  │      → channel closes    │                            │
  └─────────────────────────┘                            │
                                                          │
                                                          ▼
                                              ┌─────────────────────┐
                                              │  ChunkIterator      │
                                              │  (Python __next__)  │
                                              │                     │
                                              │  receiver.recv()    │
                                              │  → build dict:      │
                                              │    {text, metadata}  │
                                              │  → yield to Python  │
                                              └─────────────────────┘
```

**Properties:**

| Property | Detail |
|----------|--------|
| Memory | O(1) — bounded to ~100 × chunk_size bytes |
| Backpressure | If Python is slow, Rust blocks on `sender.send()` |
| Thread safety | `Mutex<Option<Receiver>>` protects the channel receiver |
| Termination | If Python drops the iterator, `sender.send()` returns `Err` → Rust thread exits |

---

## 6. Format Conversion Pipeline

**Source:** `python/krira_augment/krira_chunker.py` → `Pipeline._convert_to_jsonl()`

All input formats are normalized to JSONL before being passed to the Rust core:

| Format | Library Used | Conversion Algorithm |
|--------|-------------|---------------------|
| **CSV** | — (pass-through) | Direct to Rust core (line-by-line processing) |
| **TXT** | — (pass-through) | Direct to Rust core |
| **JSONL** | — (pass-through) | Direct to Rust core |
| **JSON** | `json` (stdlib) | Flatten list/dict → per-item JSONL rows |
| **PDF** | `pdfplumber` | Page-by-page `extract_text()` → JSONL with page metadata |
| **DOCX** | `python-docx` | Paragraph-by-paragraph extraction → JSONL |
| **XLSX** | `openpyxl` | Read-only mode, row-by-row `"key: value"` text → JSONL |
| **XML** | `ElementTree` | Recursive `itertext()` per root child element → JSONL |
| **URL** | `requests` + `BeautifulSoup` | HTTP GET → strip `<script>`/`<style>` → extract text → JSONL |

**URL Processing Detail:**

```
1. HTTP GET with 10s timeout
2. Parse HTML with BeautifulSoup
3. Remove all <script> and <style> elements
4. Extract text with newline separator
5. Split multi-headlines (double-space split)
6. Drop blank lines
7. Write as single JSONL record with URL metadata
```

---

## 7. Hashing & ID Generation

### 7.1 Deterministic Chunk IDs

**Source:** Referenced as `stable_id()` in `pipeline.rs`

Each chunk gets a deterministic ID computed from:

```
stable_id(source, source_path, chunk_index, content) → unique string
```

**Purpose:** Same input + same config → same chunk IDs across runs. This enables:
- Deduplication
- Cache invalidation
- Incremental updates

### 7.2 Configuration Fingerprinting

**Source:** `src/config.rs` → `ChunkConfig::config_hash()`

Uses Rust's `DefaultHasher` (SipHash 2-4 internally) to create a 12-character hex fingerprint:

```
Hashed fields:
  max_chars + overlap_chars + use_tokens + max_tokens +
  overlap_tokens + min_chars + chunk_strategy + preserve_code_blocks +
  preserve_tables + preserve_lists

  → SipHash → format as hex → truncate to 12 chars

Example: "a1b2c3d4e5f6"
```

**Purpose:** Embed config hash in chunk metadata so you can trace which configuration produced each chunk.

---

## 8. Security Algorithms

| Feature | Algorithm | Source |
|---------|-----------|--------|
| **SSRF Protection** | Validate URL against private IP ranges (10.x, 172.16-31.x, 192.168.x, 127.x) before HTTP request | `URLChunker` |
| **File Size Limit** | Reject files exceeding `security_max_file_bytes` (default: 50 MB) before processing | `config.rs` |
| **Zip Slip Protection** | Verify extracted paths don't escape target directory via `../` traversal | `exceptions.py` |
| **Content-Type Deny List** | Block dangerous MIME types when fetching URLs | `URLChunker` |
| **Encoding Safety** | Fallback chain: UTF-8 → Latin-1 → CP1252 → UTF-16 → UTF-8 with replace | `pipeline.py` |

---

## 9. Performance Optimizations

| Technique | Location | Impact |
|-----------|----------|--------|
| **Memory-mapped I/O** | `lib.rs` (memmap2) | Zero-copy file access, OS-managed page cache |
| **Rayon data parallelism** | `lib.rs` (par_iter) | Automatic work-stealing across CPU cores |
| **Pre-compiled regex** | `cleaning.rs` (lazy_static!) | Compile once at startup, match O(n) per pattern |
| **Bounded sync channels** | `lib.rs` (sync_channel 128) | Backpressure prevents unbounded memory growth |
| **Buffered I/O** | `lib.rs` (BufWriter 64 KB) | Batches small writes into larger syscalls |
| **Batch flushing** | `lib.rs` (100-item batches) | Reduces channel send/receive overhead |
| **Link-Time Optimization (LTO)** | `Cargo.toml` (lto = "fat") | Whole-program optimization across crate boundaries |
| **Single codegen unit** | `Cargo.toml` (codegen-units = 1) | Maximum optimization within the crate |
| **Panic abort** | `Cargo.toml` (panic = "abort") | Eliminates unwinding tables, smaller binary |
| **opt-level = 3** | `Cargo.toml` | Aggressive compiler optimizations |
| **Copy-on-Write strings** | `cleaning.rs` (Cow<str>) | Avoids allocation when regex doesn't match |
| **Generator streaming** | `pipeline.py` (yield) | O(1) Python-side memory for any file size |
| **GIL release** | `lib.rs` (py.allow_threads) | Rust threads run without Python GIL contention |
| **Lazy imports** | `__init__.py` (__getattr__) | Format-specific deps loaded only when used |
| **CSV separator auto-detect** | `pipeline.py` | Heuristic: count(`,`) vs count(`\t`) vs count(`;`) in header |
| **Adaptive batch sizing** | `pipeline.py` | On MemoryError, batch size reduced by 4× and retried |

---

## 10. Complexity Summary

| Algorithm | Time Complexity | Space Complexity |
|-----------|----------------|-----------------|
| Hybrid 3-tier chunking | O(n) | O(max_chars) buffer |
| Sliding window with overlap | O(n) | O(max_chars + overlap) |
| Regex cleaning (Rust) | O(n × p), p = pattern count | O(n) worst case |
| Unicode NFKC normalization | O(n) | O(n) |
| JSON recursive flattening | O(n) bounded by depth | O(depth) stack |
| CSV → Markdown table | O(rows × cols) | O(rows × cols) |
| Memory-mapped file I/O | O(1) setup | O(1) virtual memory |
| Newline-aligned splitting | O(n) | O(segments) |
| Parallel segment processing | O(n / cores) | O(segments × batch_size) |
| Streaming (bounded channel) | O(n) | O(channel_capacity × chunk_size) |
| SipHash config fingerprint | O(fields) | O(1) |
| Deterministic chunk ID | O(content_length) | O(1) |
| CSV separator auto-detect | O(header_length) | O(1) |
| Encoding fallback | O(file_size × attempts) | O(file_size) |

---

## Appendix: Default Configuration Values

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_chars` | 2200 | Maximum characters per chunk |
| `overlap_chars` | 250 | Character overlap between chunks |
| `min_chars` | 30 | Minimum chunk size (filter threshold) |
| `chunk_strategy` | `"hybrid"` | Chunking algorithm selection |
| `use_tokens` | `false` | Use token-based instead of char-based sizing |
| `max_tokens` | 512 | Maximum tokens per chunk |
| `overlap_tokens` | 64 | Token overlap |
| `preserve_code_blocks` | `true` | Avoid splitting code blocks |
| `preserve_tables` | `true` | Avoid splitting markdown tables |
| `preserve_lists` | `true` | Avoid splitting list items |
| `csv_batch_rows` | 50,000 | Rows per processing batch (CSV) |
| `xlsx_batch_rows` | 25,000 | Rows per processing batch (XLSX) |
| `sink_batch_size` | 256 | Chunks per sink write batch |
| `http_timeout_s` | 15 | URL fetch timeout |
| `url_max_bytes` | 8 MB | Maximum URL response size |
| `security_max_file_bytes` | 50 MB | Maximum allowed file size |
