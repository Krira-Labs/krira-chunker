"""
DOCX chunker for Krira_Chunker.
"""

import os
import zipfile
import xml.etree.ElementTree as ET
from typing import Generator, Dict, Any, List, Optional, Iterator

from ..config import ChunkConfig
from ..core import FastChunker, HybridBoundaryChunker, LOGGER
from ..exceptions import FileSizeLimitError, ProcessingError, ZipSlipError


class DOCXChunker:
    """
    Class-based DOCX chunker.
    
    Extracts paragraphs and headings from Word documents,
    preserving document structure.
    
    Example:
        >>> cfg = ChunkConfig(max_chars=2000)
        >>> chunker = DOCXChunker(cfg)
        >>> for chunk in chunker.chunk_file("document.docx"):
        ...     print(chunk["text"][:100])
    """
    
    def __init__(self, cfg: ChunkConfig = None):
        """
        Initialize DOCX chunker.
        
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
    
    def _validate_zip_member(self, archive_path: str, member_name: str) -> None:
        """Validate zip member for zip-slip attacks."""
        # Normalize path
        member_name = member_name.replace("\\", "/")
        
        if member_name.startswith("/") or ".." in member_name:
            raise ZipSlipError(archive_path, member_name)
        
        if len(member_name) > 1 and member_name[1] == ":":
            raise ZipSlipError(archive_path, member_name)
    
    def chunk_file(self, file_path: str) -> Iterator[Dict[str, Any]]:
        """
        Chunk a DOCX file.
        
        Args:
            file_path: Path to DOCX file.
            
        Yields:
            Chunk dictionaries with paragraph range metadata.
            
        Raises:
            FileSizeLimitError: If file exceeds size limit.
            ProcessingError: If DOCX is invalid.
        """
        cfg = self.cfg
        
        if not os.path.exists(file_path):
            raise ProcessingError(f"File not found: {file_path}", {"path": file_path})
        
        # Security: check file size
        size = os.path.getsize(file_path)
        if size > cfg.security_max_file_bytes:
            raise FileSizeLimitError(file_path, size, cfg.security_max_file_bytes)
        
        if not zipfile.is_zipfile(file_path):
            raise ProcessingError(f"Invalid DOCX format: {file_path}", {"path": file_path})
        
        base_meta = {
            "source": os.path.basename(file_path),
            "source_path": os.path.abspath(file_path),
            "source_type": "docx",
        }
        
        try:
            with zipfile.ZipFile(file_path) as z:
                # Validate member path
                self._validate_zip_member(file_path, "word/document.xml")
                xml_content = z.read("word/document.xml")
        except KeyError:
            raise ProcessingError("Missing document.xml in DOCX", {"path": file_path})
        except Exception as e:
            LOGGER.error("Error reading DOCX %s: %s", file_path, e)
            raise ProcessingError(f"Failed to read DOCX: {e}", {"path": file_path})
        
        try:
            root = ET.fromstring(xml_content)
        except Exception as e:
            LOGGER.error("DOCX XML Parse Error: %s", e)
            raise ProcessingError(f"Failed to parse DOCX XML: {e}", {"path": file_path})
        
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        
        units: List[str] = []
        para_ids: List[int] = []
        current_heading: Optional[str] = None
        
        para_index = 0
        for paragraph in root.findall(".//w:p", ns):
            para_index += 1
            
            style_val = None
            ppr = paragraph.find("./w:pPr", ns)
            if ppr is not None:
                ps = ppr.find("./w:pStyle", ns)
                if ps is not None:
                    style_val = ps.attrib.get(f"{{{ns['w']}}}val")
            
            texts = [node.text for node in paragraph.findall(".//w:t", ns) if node.text]
            full_text = "".join(texts).strip()
            if not full_text:
                continue
            
            # Track headings
            if style_val and str(style_val).lower().startswith("heading"):
                current_heading = full_text
            
            # Prefix with heading for context
            if current_heading and current_heading != full_text:
                full_text = f"{current_heading}\n{full_text}"
            
            units.append(full_text)
            para_ids.append(para_index)
        
        if not units:
            return
        
        chunk_index = 0
        for ch in self.chunker.chunk_units(
            units=units,
            base_meta=base_meta,
            joiner="\n",
            locator="docx",
            range_key="paragraph",
            range_vals=para_ids,
            start_chunk_index=chunk_index,
        ):
            chunk_index = ch["metadata"]["chunk_index"] + 1
            yield ch


# Backward compatibility function
def iter_chunks_from_docx(
    file_path: str,
    cfg: ChunkConfig = None
) -> Generator[Dict[str, Any], None, None]:
    """
    Iterate over chunks from a DOCX file.
    
    Args:
        file_path: Path to DOCX file.
        cfg: Chunk configuration.
        
    Yields:
        Chunk dictionaries.
    """
    chunker = DOCXChunker(cfg)
    yield from chunker.chunk_file(file_path)
