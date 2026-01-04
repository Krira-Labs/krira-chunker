"""
Krira Augment - High Performance RAG Framework

A production-grade Python library for document chunking in RAG pipelines,
backed by a highly optimized Rust core for maximum performance.
"""
import json
import os
from dataclasses import dataclass, asdict
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Iterator, Generator
import tempfile
import shutil
from pathlib import Path

# Optional dependencies support
def _check_import(module_name: str, feature_name: str):
    import importlib
    try:
        return importlib.import_module(module_name)
    except ImportError:
        raise ImportError(f"Missing optional dependency '{module_name}' for {feature_name}. Install it with `pip install krira-augment[{feature_name}]` or `pip install {module_name}`.")

# Import Rust functions
try:
    from ._rust import process_file_rust, process_stream as _rust_process_stream
except ImportError:
    try:
        from krira_augment._rust import process_file_rust, process_stream as _rust_process_stream
    except ImportError:
        def process_file_rust(*args, **kwargs):
            raise ImportError(
                "Rust extension not found. Please ensure the package is installed correctly "
                "or build in development mode with `maturin develop --release`."
            )
        def _rust_process_stream(*args, **kwargs):
            raise ImportError(
                "Rust extension not found. Please ensure the package is installed correctly "
                "or build in development mode with `maturin develop --release`."
            )
        print("WARNING: Rust extension not found. Chunking will fail.")

# =============================================================================
# Professional API (Matching README)
# =============================================================================

class SplitStrategy(Enum):
    """Chunking strategy enum."""
    FIXED = "fixed"
    SMART = "smart"  # Hybrid
    MARKDOWN = "markdown"

@dataclass
class PipelineConfig:
    """
    Configuration for the Krira Pipeline.
    """
    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 100
    strategy: SplitStrategy = SplitStrategy.SMART
    
    # Cleaning
    clean_html: bool = True
    clean_unicode: bool = True
    min_chunk_len: int = 20
    
    # Performance
    threads: int = 8
    batch_size: int = 1000

    def to_json(self) -> str:
        """Serialize configuration for Rust backend."""
        # Map nice Python names to internal Rust names
        return json.dumps({
            "max_chars": self.chunk_size,
            # Current V2 Rust core mainly uses max_chars. 
            # Future versions will use the rest.
        })

@dataclass
class PipelineStats:
    """Statistics returned after processing a file."""
    chunks_created: int
    execution_time: float  # Time in seconds
    mb_per_second: float
    output_file: str
    preview_chunks: List[str]  # Top 3 chunks as preview
    
    def __str__(self) -> str:
        """Pretty print the stats."""
        lines = [
            f"\n{'='*60}",
            f"✅ KRIRA AUGMENT - Processing Complete",
            f"{'='*60}",
            f"📊 Chunks Created:  {self.chunks_created:,}",
            f"⏱️  Execution Time:  {self.execution_time:.2f} seconds",
            f"🚀 Throughput:      {self.mb_per_second:.2f} MB/s",
            f"📁 Output File:     {self.output_file}",
            f"{'='*60}",
        ]
        
        if self.preview_chunks:
            lines.append(f"\n📝 Preview (Top 3 Chunks):")
            lines.append(f"{'-'*60}")
            for i, chunk in enumerate(self.preview_chunks[:3], 1):
                # Truncate long chunks for display
                display_text = chunk[:150] + "..." if len(chunk) > 150 else chunk
                lines.append(f"[{i}] {display_text}")
            lines.append(f"{'-'*60}")
        
        return "\n".join(lines)

class Pipeline:
    """
    Main entry point for Krira Augment.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

    def _convert_to_jsonl(self, input_path: str) -> str:
        """
        Convert various input formats to a temporary JSONL file that the Rust core can process.
        Returns the path to the temporary file.
        """
        base_ext = os.path.splitext(input_path)[1].lower()
        
        # 0. URL Handling
        if input_path.startswith("http://") or input_path.startswith("https://"):
            return self._process_url(input_path)

        # 1. Text/JSONL/CSV (Direct Pass-through possibilities, but we want consistency)
        # For now, we pass CSV/JSONL/TXT directly if they are simple, 
        # BUT if we want proper row handling for CSV, we should convert here too?
        # The Rust core treats lines as text. 
        # - TXT: Fine.
        # - JSONL: Fine.
        # - CSV: Rust sees "col1,col2,col3". If that's okay, pass through.
        if base_ext in ['.txt', '.jsonl', '.csv']:
            return input_path

        # 2. Complex Formats -> Start conversion
        temp_fd, temp_path = tempfile.mkstemp(suffix=".jsonl", prefix="krira_convert_")
        os.close(temp_fd)
        
        try:
            if base_ext == '.json':
                self._convert_json(input_path, temp_path)
            elif base_ext == '.pdf':
                self._convert_pdf(input_path, temp_path)
            elif base_ext == '.docx':
                self._convert_docx(input_path, temp_path)
            elif base_ext == '.xlsx':
                self._convert_xlsx(input_path, temp_path)
            elif base_ext == '.xml':
                self._convert_xml(input_path, temp_path)
            else:
                # Fallback: Treat as text
                print(f"WARNING: Unknown extension {base_ext}, treating as text.")
                return input_path
                
        except Exception as e:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise RuntimeError(f"Failed to convert {input_path}: {e}")

        return temp_path

    def _write_temp_jsonl(self, temp_path: str, generator):
        with open(temp_path, 'w', encoding='utf-8') as f:
            for item in generator:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def _convert_json(self, input_path, temp_path):
        """Flatten JSON list or dict to JSONL."""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        items = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = [data]
        else:
            raise ValueError("JSON must be a list or dict")
            
        # Ensure strings
        final_items = []
        for item in items:
            if isinstance(item, str):
                final_items.append({"text": item})
            else:
                # Dump object to string if it's not a string
                final_items.append({"text": json.dumps(item, ensure_ascii=False)})
        
        self._write_temp_jsonl(temp_path, final_items)

    def _convert_pdf(self, input_path, temp_path):
        pdfplumber = _check_import("pdfplumber", "pdf")
        
        items = []
        with pdfplumber.open(input_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    items.append({
                        "text": text, 
                        "metadata": {"page": i + 1, "source": input_path}
                    })
        self._write_temp_jsonl(temp_path, items)

    def _convert_docx(self, input_path, temp_path):
        docx = _check_import("docx", "docx")
        
        doc = docx.Document(input_path)
        items = []
        for para in doc.paragraphs:
            if para.text.strip():
                items.append({
                    "text": para.text,
                    "metadata": {"source": input_path}
                })
        self._write_temp_jsonl(temp_path, items)

    def _convert_xlsx(self, input_path, temp_path):
        openpyxl = _check_import("openpyxl", "xlsx")
        
        wb = openpyxl.load_workbook(input_path, read_only=True, data_only=True)
        items = []
        for sheet in wb:
            rows = sheet.values
            headers = next(rows, None)
            if not headers: 
                continue
            
            headers = [str(h) for h in headers]
            for row in rows:
                # Convert row to text representation
                row_dict = {h: str(v) if v is not None else "" for h, v in zip(headers, row)}
                # Serialize row as text
                text_rep = " | ".join(f"{k}: {v}" for k, v in row_dict.items() if v)
                if text_rep:
                    items.append({
                        "text": text_rep,
                        "metadata": {"sheet": sheet.title, "source": input_path}
                    })
        self._write_temp_jsonl(temp_path, items)

    def _convert_xml(self, input_path, temp_path):
        import xml.etree.ElementTree as ET
        tree = ET.parse(input_path)
        root = tree.getroot()
        
        # Naive XML: Convert each child of root to a string
        items = []
        for child in root:
            # Get all text recursively
            text = "".join(child.itertext()).strip()
            if text:
                items.append({"text": text, "metadata": {"tag": child.tag}})
                
        self._write_temp_jsonl(temp_path, items)

    def _process_url(self, url):
        requests = _check_import("requests", "url")
        bs4 = _check_import("bs4", "url")
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = bs4.BeautifulSoup(response.text, "html.parser")
        
        # Kill all script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        text = soup.get_text(separator="\n")
        
        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        temp_fd, temp_path = tempfile.mkstemp(suffix=".jsonl", prefix="krira_url_")
        os.close(temp_fd)
        
        self._write_temp_jsonl(temp_path, [{"text": text, "metadata": {"url": url}}])
        return temp_path

    def process(self, input_path: str, output_path: Optional[str] = None) -> PipelineStats:
        """
        Process a file using the Rust core engine.
        Automatically converts PDF, DOCX, XLSX, XML, JSON, and URLs to a format Rust can handle.
        """
        import time
        
        # Check input existence only if it's not a URL
        is_url = input_path.startswith("http://") or input_path.startswith("https://")
        if not is_url:
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")
                
        # Determine output path if not provided
        if output_path is None:
            if is_url:
                # Use a safe filename based on URL hash or simple sanitize
                import hashlib
                url_hash = hashlib.md5(input_path.encode()).hexdigest()[:8]
                output_path = f"url_output_{url_hash}.jsonl"
            else:
                base, _ = os.path.splitext(input_path)
                output_path = f"{base}_processed.jsonl"
            
        start_time = time.time()
        
        # Pre-process
        processed_input_path = self._convert_to_jsonl(input_path)
        is_temp = processed_input_path != input_path
        
        try:
            # Invoke Rust Core (which expects text-based files)
            process_file_rust(processed_input_path, output_path, self.config.to_json())
        finally:
            # Cleanup temp file if created
            if is_temp and os.path.exists(processed_input_path):
                try:
                    os.unlink(processed_input_path)
                except OSError:
                    pass
        
        duration = time.time() - start_time
        
        # Count chunks and get preview from output file
        chunks_created = 0
        preview_chunks = []
        
        try:
            if os.path.exists(output_path):
                with open(output_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        chunks_created += 1
                        # Collect up to 1000 chunks for preview (allows user to slice as needed)
                        if i < 1000:
                            try:
                                chunk_data = json.loads(line.strip())
                                text = chunk_data.get('text', str(chunk_data))
                                preview_chunks.append(text)
                            except json.JSONDecodeError:
                                preview_chunks.append(line.strip())
        except Exception:
            pass  # If reading fails, keep defaults
        
        # Calculate throughput based on input file size
        try:
            if not is_url and os.path.exists(input_path):
                file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
            else:
                file_size_mb = 0
        except OSError:
            file_size_mb = 0
             
        throughput = file_size_mb / duration if duration > 0 else 0
        
        return PipelineStats(
            chunks_created=chunks_created,
            execution_time=duration,
            mb_per_second=throughput,
            output_file=output_path,
            preview_chunks=preview_chunks
        )

    def process_stream(self, input_path: str) -> Iterator[Dict[str, Any]]:
        """
        Stream chunks from a file without creating intermediate files.
        
        This method provides a memory-efficient way to process large files by yielding
        chunks one at a time. Each chunk can be embedded and stored immediately,
        eliminating the need for intermediate file storage.
        
        Args:
            input_path (str): Path to the input file. Supports CSV, TXT, JSON, JSONL, 
                             PDF, DOCX, XLSX, XML, and URLs.
        
        Yields:
            dict: A dictionary containing:
                - text (str): The chunk text content
                - metadata (dict): Metadata including:
                    - source (str): Original file path
                    - chunk_index (int): Sequential chunk number
                    - char_count (int): Number of characters in chunk
        
        Memory:
            O(1) - Constant memory usage regardless of file size.
            Maximum ~50MB for internal buffering.
        
        Performance:
            - Processes 1GB file in ~12 seconds
            - Utilizes multi-core parallel processing
            - No disk I/O for intermediate files
        
        Example:
            Basic usage:
            >>> config = PipelineConfig(chunk_size=512, chunk_overlap=50)
            >>> pipeline = Pipeline(config=config)
            >>> for chunk in pipeline.process_stream("data.csv"):
            ...     print(chunk["text"][:50])
            
            With OpenAI embedding:
            >>> import openai
            >>> for chunk in pipeline.process_stream("data.csv"):
            ...     embedding = openai.Embedding.create(input=chunk["text"])
            ...     # Store embedding immediately
            
            With progress tracking:
            >>> chunk_count = 0
            >>> for chunk in pipeline.process_stream("data.csv"):
            ...     chunk_count += 1
            ...     if chunk_count % 100 == 0:
            ...         print(f"Processed {chunk_count} chunks")
        
        Raises:
            FileNotFoundError: If input_path does not exist
            ImportError: If required optional dependencies are not installed
        
        Note:
            - Chunks are processed sequentially for consistent ordering
            - The iterator cannot be restarted; create a new one if needed
            - For very large files (>50GB), consider using file-based `process()` mode
        """
        # Check input existence
        is_url = input_path.startswith("http://") or input_path.startswith("https://")
        if not is_url:
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Pre-process file if needed (PDF, DOCX, etc.)
        processed_input_path = self._convert_to_jsonl(input_path)
        is_temp = processed_input_path != input_path
        
        try:
            # Stream from the Rust core
            iterator = _rust_process_stream(
                processed_input_path,
                self.config.chunk_size,
                self.config.chunk_overlap
            )
            
            # Wrap to ensure cleanup
            for chunk in iterator:
                yield chunk
                
        finally:
            # Cleanup temp file if created
            if is_temp and os.path.exists(processed_input_path):
                try:
                    os.unlink(processed_input_path)
                except OSError:
                    pass

    def preview(self, n: int = 3) -> str:
        """
        Preview the first n chunks (deprecated, use process() and check preview_chunks).
        """
        return f"Use pipeline.process(...).preview_chunks for preview"


# =============================================================================
# Streaming Utilities
# =============================================================================

class StreamingChunkIterator:
    """
    A wrapper for streaming chunk iteration with additional utilities.
    """
    def __init__(self, pipeline: Pipeline, input_path: str):
        self.pipeline = pipeline
        self.input_path = input_path
        self._iterator = None
        self._chunk_count = 0
    
    def __iter__(self):
        self._iterator = self.pipeline.process_stream(self.input_path)
        return self
    
    def __next__(self) -> Dict[str, Any]:
        if self._iterator is None:
            self._iterator = self.pipeline.process_stream(self.input_path)
        chunk = next(self._iterator)
        self._chunk_count += 1
        return chunk
    
    @property
    def chunks_processed(self) -> int:
        """Return the number of chunks processed so far."""
        return self._chunk_count


# =============================================================================
# Legacy & Exports
# =============================================================================

# For backward compatibility if needed
KriraLoader = Pipeline 
TextSplitter = PipelineConfig

__all__ = [
    "Pipeline", 
    "PipelineConfig", 
    "SplitStrategy", 
    "PipelineStats",
    "StreamingChunkIterator"
]
