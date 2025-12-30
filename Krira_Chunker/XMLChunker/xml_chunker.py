"""
XML chunker for Krira_Chunker.
"""

import os
import xml.etree.ElementTree as ET
from typing import Generator, Dict, Any, List, Tuple, Iterator

from ..config import ChunkConfig
from ..core import FastChunker, HybridBoundaryChunker, LOGGER
from ..exceptions import FileSizeLimitError, ProcessingError


def _xml_path_from_stack(stack: List[Tuple[str, int]]) -> str:
    """Build XPath-like string from element stack."""
    parts = []
    for tag, idx in stack:
        if idx > 0:
            parts.append(f"{tag}[{idx}]")
        else:
            parts.append(tag)
    return "/" + "/".join(parts)


class XMLChunker:
    """
    Class-based XML chunker with streaming parsing.
    
    Uses iterparse for memory-efficient processing of large XML files.
    
    Example:
        >>> cfg = ChunkConfig(max_chars=2000)
        >>> chunker = XMLChunker(cfg)
        >>> for chunk in chunker.chunk_file("data.xml"):
        ...     print(chunk["text"][:100])
    """
    
    def __init__(self, cfg: ChunkConfig = None):
        """
        Initialize XML chunker.
        
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
    
    def chunk_file(self, file_path: str) -> Iterator[Dict[str, Any]]:
        """
        Chunk an XML file.
        
        Uses streaming parsing to handle large files efficiently.
        
        Args:
            file_path: Path to XML file.
            
        Yields:
            Chunk dictionaries with xml_path metadata.
            
        Raises:
            FileSizeLimitError: If file exceeds size limit.
            ProcessingError: If XML parsing fails.
        """
        cfg = self.cfg
        
        if not os.path.exists(file_path):
            raise ProcessingError(f"File not found: {file_path}", {"path": file_path})
        
        # Security: check file size
        size = os.path.getsize(file_path)
        if size > cfg.security_max_file_bytes:
            raise FileSizeLimitError(file_path, size, cfg.security_max_file_bytes)
        
        base_meta = {
            "source": os.path.basename(file_path),
            "source_path": os.path.abspath(file_path),
            "source_type": "xml",
        }
        
        # Streaming parse
        stack: List[Tuple[str, int]] = []
        counters_stack: List[Dict[str, int]] = []
        
        chunk_index = 0
        
        try:
            context = ET.iterparse(file_path, events=("start", "end"))
            
            for event, elem in context:
                tag = elem.tag
                
                if event == "start":
                    if not stack:
                        idx = 0
                    else:
                        parent_counters = counters_stack[-1]
                        idx = parent_counters.get(tag, 0)
                        parent_counters[tag] = idx + 1
                    stack.append((tag, idx))
                    counters_stack.append({})
                    continue
                
                # event == "end"
                xml_path = _xml_path_from_stack(stack)
                
                # Process text content
                for kind, txt in (("text", elem.text), ("tail", elem.tail)):
                    if not txt:
                        continue
                    text = txt.strip()
                    if not text:
                        continue
                    
                    meta = dict(base_meta)
                    meta["xml_path"] = xml_path
                    meta["xml_text_kind"] = kind
                    
                    # Use hybrid chunker if configured
                    if cfg.chunk_strategy == "hybrid":
                        for ch in self.hybrid_chunker.chunk_text(
                            text=text,
                            base_meta=meta,
                            locator=f"xml|{xml_path}|{kind}",
                            start_chunk_index=chunk_index,
                        ):
                            chunk_index = ch["metadata"]["chunk_index"] + 1
                            yield ch
                    else:
                        for ch in self.chunker.chunk_text(
                            text=text,
                            base_meta=meta,
                            mode="prose",
                            locator=f"xml|{xml_path}|{kind}",
                            joiner=" ",
                            start_chunk_index=chunk_index,
                        ):
                            chunk_index = ch["metadata"]["chunk_index"] + 1
                            yield ch
                
                # Clear element to free memory
                elem.clear()
                
                # Pop stacks
                if stack:
                    stack.pop()
                if counters_stack:
                    counters_stack.pop()
                    
        except ET.ParseError as e:
            LOGGER.error("XML Parse Error in %s: %s", file_path, e)
            raise ProcessingError(f"Failed to parse XML: {e}", {"path": file_path})
        except Exception as e:
            LOGGER.error("XML Error in %s: %s", file_path, e)
            raise ProcessingError(f"Error processing XML: {e}", {"path": file_path})


# Backward compatibility function
def iter_chunks_from_xml(
    file_path: str,
    cfg: ChunkConfig = None
) -> Generator[Dict[str, Any], None, None]:
    """
    Iterate over chunks from an XML file.
    
    Args:
        file_path: Path to XML file.
        cfg: Chunk configuration.
        
    Yields:
        Chunk dictionaries.
    """
    chunker = XMLChunker(cfg)
    yield from chunker.chunk_file(file_path)
