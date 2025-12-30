"""
URL chunker for Krira_Chunker with SSRF protection.
"""

import time
import socket
import ipaddress
import urllib.parse
from typing import Generator, Dict, Any, Iterator, Optional, List
from datetime import datetime

from ..config import ChunkConfig
from ..core import FastChunker, HybridBoundaryChunker, LOGGER, clean_text
from ..exceptions import (
    DependencyNotInstalledError,
    SSRFError,
    ContentTypeDeniedError,
    FileSizeLimitError,
    ProcessingError,
)


def _is_private_target(hostname: str) -> bool:
    """
    Check if hostname resolves to private/internal IP.
    
    Args:
        hostname: Hostname to check.
        
    Returns:
        True if hostname resolves to private IP.
    """
    # Quick checks for common local names
    hostname_lower = hostname.lower()
    if hostname_lower in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        return True
    if hostname_lower.endswith((".local", ".localhost", ".internal")):
        return True
    
    # Resolve and check IP ranges
    try:
        infos = socket.getaddrinfo(hostname, None)
    except Exception:
        # Conservative: treat resolution failure as unsafe
        return True
    
    for info in infos:
        ip_str = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except Exception:
            continue
        if (ip.is_private or ip.is_loopback or ip.is_link_local or 
            ip.is_reserved or ip.is_multicast):
            return True
    
    return False


class URLChunker:
    """
    Class-based URL chunker with SSRF protection.
    
    Example:
        >>> cfg = ChunkConfig(max_chars=2000, url_allow_private=False)
        >>> chunker = URLChunker(cfg)
        >>> for chunk in chunker.chunk_url("https://example.com"):
        ...     print(chunk["text"][:100])
    """
    
    def __init__(self, cfg: ChunkConfig = None, allow_private: bool = None):
        """
        Initialize URL chunker.
        
        Args:
            cfg: Chunk configuration. Uses defaults if None.
            allow_private: Override for url_allow_private config.
        """
        self.cfg = cfg or ChunkConfig()
        self._allow_private = (
            allow_private if allow_private is not None 
            else self.cfg.url_allow_private
        )
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
    
    def _get_requests(self):
        """Lazy import requests."""
        try:
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            return requests, HTTPAdapter, Retry
        except ImportError:
            raise DependencyNotInstalledError("requests", "url", "URL fetching")
    
    def _get_bs4(self):
        """Lazy import BeautifulSoup."""
        try:
            from bs4 import BeautifulSoup
            return BeautifulSoup
        except ImportError:
            raise DependencyNotInstalledError("beautifulsoup4", "url", "HTML parsing")
    
    def _get_trafilatura(self):
        """Lazy import trafilatura (optional)."""
        try:
            import trafilatura
            return trafilatura
        except ImportError:
            return None
    
    def _validate_url(self, url: str) -> tuple[str, str]:
        """
        Validate URL for security.
        
        Returns:
            Tuple of (scheme, hostname).
            
        Raises:
            SSRFError: If URL is unsafe.
        """
        parsed = urllib.parse.urlparse(url)
        
        if parsed.scheme not in ("http", "https"):
            raise SSRFError(url, f"Invalid scheme: {parsed.scheme}")
        
        hostname = parsed.hostname
        if not hostname:
            raise SSRFError(url, "No hostname in URL")
        
        if parsed.username or parsed.password:
            raise SSRFError(url, "URLs with embedded credentials not allowed")
        
        if not self._allow_private and _is_private_target(hostname):
            raise SSRFError(url, f"Private/internal network blocked: {hostname}")
        
        return parsed.scheme, hostname
    
    def _fetch_url(self, url: str) -> tuple[Optional[str], int, str]:
        """
        Fetch URL content with security checks.
        
        Returns:
            Tuple of (html_content, status_code, final_url).
        """
        requests, HTTPAdapter, Retry = self._get_requests()
        cfg = self.cfg
        
        # Initial validation
        self._validate_url(url)
        
        sess = requests.Session()
        retries = Retry(
            total=cfg.url_retries,
            backoff_factor=cfg.url_backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "HEAD"),
            raise_on_status=False,
        )
        sess.mount("http://", HTTPAdapter(max_retries=retries))
        sess.mount("https://", HTTPAdapter(max_retries=retries))
        
        headers = {"User-Agent": cfg.http_user_agent}
        
        try:
            # Stream download with size limit
            r = sess.get(
                url,
                headers=headers,
                timeout=cfg.http_timeout_s,
                stream=True,
                allow_redirects=True,
            )
            
            # Check final URL after redirects for SSRF
            if r.url != url:
                parsed_final = urllib.parse.urlparse(r.url)
                if not self._allow_private and _is_private_target(parsed_final.hostname or ""):
                    raise SSRFError(r.url, "Redirect to private network blocked")
            
            final_url = r.url
            status_code = r.status_code
            
            if status_code >= 400:
                return None, status_code, final_url
            
            # Check content type
            content_type = r.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if content_type and cfg.url_content_type_allowlist:
                if content_type not in cfg.url_content_type_allowlist:
                    raise ContentTypeDeniedError(url, content_type, cfg.url_content_type_allowlist)
            
            # Download with size limit
            chunks = []
            total = 0
            for b in r.iter_content(chunk_size=64 * 1024):
                if not b:
                    continue
                total += len(b)
                if total > cfg.url_max_bytes:
                    LOGGER.warning("URL content truncated at %d bytes: %s", cfg.url_max_bytes, url)
                    break
                chunks.append(b)
            
            raw = b"".join(chunks)
            try:
                html = raw.decode(r.encoding or "utf-8", errors="ignore")
            except Exception:
                html = raw.decode("utf-8", errors="ignore")
            
            return html, status_code, final_url
            
        except SSRFError:
            raise
        except ContentTypeDeniedError:
            raise
        except Exception as e:
            LOGGER.error("Error fetching URL %s: %s", url, e)
            return None, 0, url
        finally:
            try:
                sess.close()
            except Exception:
                pass
    
    def _extract_content(self, html: str, url: str) -> tuple[str, List[str], str]:
        """
        Extract main content from HTML.
        
        Returns:
            Tuple of (title, blocks, method_used).
        """
        BeautifulSoup = self._get_bs4()
        
        # Try trafilatura first (if available)
        trafilatura = self._get_trafilatura()
        if trafilatura:
            try:
                extracted = trafilatura.extract(html, include_comments=False)
                if extracted and len(extracted) > 100:
                    soup = BeautifulSoup(html, "html.parser")
                    title = soup.title.string.strip() if soup.title and soup.title.string else url
                    # Split into paragraphs
                    blocks = [p.strip() for p in extracted.split("\n\n") if p.strip()]
                    return title, blocks, "trafilatura"
            except Exception:
                pass
        
        # Fallback to BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else url
        
        # Find main content container
        is_wiki = "wikipedia.org" in url.lower()
        if is_wiki:
            container = soup.select_one("#mw-content-text") or soup.select_one("article") or soup.body
        else:
            container = (
                soup.select_one("article") or 
                soup.select_one("main") or 
                soup.select_one("#content") or
                soup.body
            )
        
        if container is None:
            container = soup
        
        # Remove noise
        for sel in ["script", "style", "noscript", "nav", "footer", "aside", "header"]:
            for node in container.select(sel):
                node.extract()
        
        if is_wiki:
            # Extra cleanup for Wikipedia
            for sel in [".mw-editsection", ".toc", "#toc", ".infobox", ".navbox",
                       ".reflist", ".reference", "#catlinks"]:
                for node in container.select(sel):
                    node.extract()
        
        # Extract blocks
        blocks: List[str] = []
        for tag in container.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]):
            t = tag.get_text(" ", strip=True)
            if t and len(t) >= self.cfg.min_chars:
                blocks.append(t)
        
        if not blocks:
            # Fallback to all text
            text = container.get_text(separator=" ", strip=True)
            if text:
                blocks = [text]
        
        return title, blocks, "beautifulsoup"
    
    def chunk_url(self, url: str) -> Iterator[Dict[str, Any]]:
        """
        Chunk content from a URL.
        
        Args:
            url: URL to fetch and chunk.
            
        Yields:
            Chunk dictionaries.
            
        Raises:
            DependencyNotInstalledError: If requests/bs4 not installed.
            SSRFError: If URL targets private network.
            ContentTypeDeniedError: If content type not allowed.
        """
        cfg = self.cfg
        
        # Rate limiting
        if cfg.url_rate_limit_s > 0:
            time.sleep(cfg.url_rate_limit_s)
        
        html, status_code, final_url = self._fetch_url(url)
        if not html:
            return
        
        title, blocks, method = self._extract_content(html, final_url)
        
        if not blocks:
            LOGGER.warning("No content extracted from URL: %s", url)
            return
        
        base_meta = {
            "source": url,
            "source_path": url,
            "source_type": "url",
            "title": title,
            "url": final_url,
            "http_status": status_code,
            "fetched_at": datetime.utcnow().isoformat() + "Z",
            "extraction_method": method,
        }
        
        chunk_index = 0
        
        # Use hybrid chunker for full text if strategy is hybrid
        if cfg.chunk_strategy == "hybrid" and len(blocks) == 1:
            for ch in self.hybrid_chunker.chunk_text(
                text=blocks[0],
                base_meta=base_meta,
                locator="url",
                start_chunk_index=chunk_index,
            ):
                chunk_index = ch["metadata"]["chunk_index"] + 1
                yield ch
        else:
            for ch in self.chunker.chunk_units(
                units=blocks,
                base_meta=base_meta,
                joiner="\n",
                locator="url",
                start_chunk_index=chunk_index,
            ):
                chunk_index = ch["metadata"]["chunk_index"] + 1
                yield ch


# Backward compatibility function
def iter_chunks_from_url(
    url: str,
    cfg: ChunkConfig = None
) -> Generator[Dict[str, Any], None, None]:
    """
    Iterate over chunks from a URL.
    
    Args:
        url: URL to fetch and chunk.
        cfg: Chunk configuration.
        
    Yields:
        Chunk dictionaries.
    """
    chunker = URLChunker(cfg)
    yield from chunker.chunk_url(url)
