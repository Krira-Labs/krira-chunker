"""
PDF chunker for Krira_Chunker.
"""

import os
from typing import Generator, Dict, Any, Iterator

from ..config import ChunkConfig
from ..core import FastChunker, HybridBoundaryChunker, LOGGER, clean_text
from ..exceptions import (
    DependencyNotInstalledError,
    FileSizeLimitError,
    OCRRequiredError,
    ProcessingError,
)


class PDFChunker:
    """
    Class-based PDF chunker with lazy dependency loading.
    
    Example:
        >>> cfg = ChunkConfig(max_chars=2000)
        >>> chunker = PDFChunker(cfg)
        >>> for chunk in chunker.chunk_file("report.pdf"):
        ...     print(chunk["text"][:100])
    """
    
    def __init__(self, cfg: ChunkConfig = None):
        """
        Initialize PDF chunker.
        
        Args:
            cfg: Chunk configuration. Uses defaults if None.
        """
        self.cfg = cfg or ChunkConfig()
        self._chunker = None
        self._hybrid_chunker = None
    
    @property
    def chunker(self) -> FastChunker:
        """Lazy-load FastChunker."""
        if self._chunker is None:
            self._chunker = FastChunker(self.cfg)
        return self._chunker
    
    @property
    def hybrid_chunker(self) -> HybridBoundaryChunker:
        """Lazy-load HybridBoundaryChunker."""
        if self._hybrid_chunker is None:
            self._hybrid_chunker = HybridBoundaryChunker(self.cfg)
        return self._hybrid_chunker
    
    def _get_pypdf(self):
        """Lazy import pypdf."""
        try:
            import pypdf
            return pypdf
        except ImportError:
            raise DependencyNotInstalledError("pypdf", "pdf", "PDF processing")
    
    def chunk_file(
        self,
        file_path: str,
        raise_on_ocr_needed: bool = False,
    ) -> Iterator[Dict[str, Any]]:
        """
        Chunk a PDF file.
        
        Args:
            file_path: Path to PDF file.
            raise_on_ocr_needed: If True, raise OCRRequiredError for scanned PDFs.
            
        Yields:
            Chunk dictionaries.
            
        Raises:
            DependencyNotInstalledError: If pypdf is not installed.
            FileSizeLimitError: If file exceeds size limit.
            OCRRequiredError: If PDF appears scanned and raise_on_ocr_needed=True.
        """
        pypdf = self._get_pypdf()
        cfg = self.cfg
        
        # Security: check file size
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            if size > cfg.security_max_file_bytes:
                raise FileSizeLimitError(file_path, size, cfg.security_max_file_bytes)
        
        base_meta = {
            "source": os.path.basename(file_path),
            "source_path": os.path.abspath(file_path),
            "source_type": "pdf",
        }
        
        try:
            reader = pypdf.PdfReader(file_path)
        except Exception as e:
            LOGGER.error("Error reading PDF %s: %s", file_path, e)
            raise ProcessingError(f"Failed to read PDF: {e}", {"path": file_path})
        
        total_pages = len(reader.pages)
        chunk_index = 0
        extracted_chars = 0
        
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            
            text = clean_text(text)
            extracted_chars += len(text)
            
            if not text:
                continue
            
            meta = dict(base_meta)
            meta["page"] = i + 1
            meta["total_pages"] = total_pages
            
            # Use hybrid chunker if configured
            if cfg.chunk_strategy == "hybrid":
                for ch in self.hybrid_chunker.chunk_text(
                    text=text,
                    base_meta=meta,
                    locator=f"pdf|page={i+1}",
                    start_chunk_index=chunk_index,
                ):
                    chunk_index = ch["metadata"]["chunk_index"] + 1
                    yield ch
            else:
                for ch in self.chunker.chunk_text(
                    text=text,
                    base_meta=meta,
                    mode="prose",
                    locator=f"pdf|page={i+1}",
                    joiner=" ",
                    start_chunk_index=chunk_index,
                ):
                    chunk_index = ch["metadata"]["chunk_index"] + 1
                    yield ch
        
        # Check for scanned PDF
        if total_pages > 0:
            avg = extracted_chars / total_pages
            if avg < cfg.pdf_min_chars_per_page:
                msg = (
                    f"PDF likely scanned (avg {avg:.1f} chars/page). "
                    "OCR not integrated yet."
                )
                LOGGER.warning("%s source=%s", msg, os.path.basename(file_path))
                if raise_on_ocr_needed:
                    raise OCRRequiredError(file_path, avg)


# Backward compatibility function
def iter_chunks_from_pdf(
    file_path: str,
    cfg: ChunkConfig = None
) -> Generator[Dict[str, Any], None, None]:
    """
    Iterate over chunks from a PDF file.
    
    Args:
        file_path: Path to PDF file.
        cfg: Chunk configuration.
        
    Yields:
        Chunk dictionaries.
    """
    chunker = PDFChunker(cfg)
    yield from chunker.chunk_file(file_path)
