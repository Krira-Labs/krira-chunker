"""
Tests for URL security - SSRF protection.
"""

import pytest
from unittest.mock import patch, MagicMock

from Krira_Chunker import ChunkConfig
from Krira_Chunker.exceptions import SSRFError, ContentTypeDeniedError, DependencyNotInstalledError


class TestSSRFProtection:
    """Test SSRF protection in URL chunker."""
    
    def test_localhost_blocked(self):
        """Test that localhost is blocked by default."""
        try:
            from Krira_Chunker.URLChunker import URLChunker
        except DependencyNotInstalledError:
            pytest.skip("requests not installed")
        
        cfg = ChunkConfig(url_allow_private=False)
        chunker = URLChunker(cfg)
        
        with pytest.raises(SSRFError) as exc_info:
            chunker._validate_url("http://localhost/admin")
        
        assert "Private/internal network blocked" in str(exc_info.value) or "localhost" in str(exc_info.value).lower()
    
    def test_127_0_0_1_blocked(self):
        """Test that 127.0.0.1 is blocked."""
        try:
            from Krira_Chunker.URLChunker import URLChunker
        except DependencyNotInstalledError:
            pytest.skip("requests not installed")
        
        cfg = ChunkConfig(url_allow_private=False)
        chunker = URLChunker(cfg)
        
        with pytest.raises(SSRFError):
            chunker._validate_url("http://127.0.0.1:8080/api")
    
    def test_local_domain_blocked(self):
        """Test that .local domains are blocked."""
        try:
            from Krira_Chunker.URLChunker import URLChunker
        except DependencyNotInstalledError:
            pytest.skip("requests not installed")
        
        cfg = ChunkConfig(url_allow_private=False)
        chunker = URLChunker(cfg)
        
        with pytest.raises(SSRFError):
            chunker._validate_url("http://myserver.local/data")
    
    def test_private_allowed_when_configured(self):
        """Test that private networks work when explicitly allowed."""
        try:
            from Krira_Chunker.URLChunker import URLChunker
        except DependencyNotInstalledError:
            pytest.skip("requests not installed")
        
        cfg = ChunkConfig(url_allow_private=True)
        chunker = URLChunker(cfg, allow_private=True)
        
        # Should not raise
        scheme, hostname = chunker._validate_url("http://localhost/test")
        assert scheme == "http"
        assert hostname == "localhost"
    
    def test_invalid_scheme_blocked(self):
        """Test that non-http(s) schemes are blocked."""
        try:
            from Krira_Chunker.URLChunker import URLChunker
        except DependencyNotInstalledError:
            pytest.skip("requests not installed")
        
        cfg = ChunkConfig()
        chunker = URLChunker(cfg)
        
        with pytest.raises(SSRFError) as exc_info:
            chunker._validate_url("file:///etc/passwd")
        
        assert "Invalid scheme" in str(exc_info.value)
    
    def test_ftp_scheme_blocked(self):
        """Test that ftp scheme is blocked."""
        try:
            from Krira_Chunker.URLChunker import URLChunker
        except DependencyNotInstalledError:
            pytest.skip("requests not installed")
        
        cfg = ChunkConfig()
        chunker = URLChunker(cfg)
        
        with pytest.raises(SSRFError):
            chunker._validate_url("ftp://example.com/file.txt")
    
    def test_credentials_in_url_blocked(self):
        """Test that URLs with embedded credentials are blocked."""
        try:
            from Krira_Chunker.URLChunker import URLChunker
        except DependencyNotInstalledError:
            pytest.skip("requests not installed")
        
        cfg = ChunkConfig()
        chunker = URLChunker(cfg)
        
        with pytest.raises(SSRFError) as exc_info:
            chunker._validate_url("http://user:pass@example.com/secret")
        
        assert "credentials" in str(exc_info.value).lower()
    
    def test_public_url_allowed(self):
        """Test that public URLs are allowed."""
        try:
            from Krira_Chunker.URLChunker import URLChunker
        except DependencyNotInstalledError:
            pytest.skip("requests not installed")
        
        cfg = ChunkConfig(url_allow_private=False)
        chunker = URLChunker(cfg)
        
        # Should not raise for public URLs
        scheme, hostname = chunker._validate_url("https://www.example.com/page")
        assert scheme == "https"
        assert hostname == "www.example.com"


class TestPrivateIPDetection:
    """Test private IP detection."""
    
    def test_is_private_target(self):
        """Test private IP detection function."""
        from Krira_Chunker.URLChunker.url_chunker import _is_private_target
        
        # These should be detected as private
        assert _is_private_target("localhost") == True
        assert _is_private_target("127.0.0.1") == True
        
        # .local domains
        assert _is_private_target("myserver.local") == True
        assert _is_private_target("test.localhost") == True


class TestContentTypeValidation:
    """Test content type validation."""
    
    def test_custom_content_type_allowlist(self):
        """Test that custom content type allowlist is respected."""
        cfg = ChunkConfig(
            url_content_type_allowlist=("text/html", "application/json")
        )
        assert "text/html" in cfg.url_content_type_allowlist
        assert "application/json" in cfg.url_content_type_allowlist
        assert "application/pdf" not in cfg.url_content_type_allowlist
