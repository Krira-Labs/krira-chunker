"""
Custom exceptions for Krira_Chunker.
"""


class KriraChunkerError(Exception):
    """Base exception for all Krira_Chunker errors."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigError(KriraChunkerError):
    """Raised when configuration is invalid."""
    pass


class DependencyNotInstalledError(KriraChunkerError):
    """Raised when an optional dependency is required but not installed."""
    
    def __init__(self, package: str, extra: str, feature: str):
        message = (
            f"The '{package}' package is required for {feature}. "
            f"Install it with: pip install krira-chunker[{extra}]"
        )
        super().__init__(message, {"package": package, "extra": extra})
        self.package = package
        self.extra = extra


class UnsupportedFormatError(KriraChunkerError):
    """Raised when file format is not supported."""
    
    def __init__(self, path: str, extension: str = None):
        ext_info = f" (extension: {extension})" if extension else ""
        message = f"Unsupported file format: {path}{ext_info}"
        super().__init__(message, {"path": path, "extension": extension})


class SecurityViolationError(KriraChunkerError):
    """Base class for security-related errors."""
    pass


class SSRFError(SecurityViolationError):
    """Raised when SSRF attack is detected."""
    
    def __init__(self, url: str, reason: str):
        message = f"SSRF protection blocked request to '{url}': {reason}"
        super().__init__(message, {"url": url, "reason": reason})


class FileSizeLimitError(SecurityViolationError):
    """Raised when file exceeds size limit."""
    
    def __init__(self, path: str, size: int, limit: int):
        message = f"File '{path}' size ({size:,} bytes) exceeds limit ({limit:,} bytes)"
        super().__init__(message, {"path": path, "size": size, "limit": limit})


class ContentTypeDeniedError(SecurityViolationError):
    """Raised when content type is not allowed."""
    
    def __init__(self, url: str, content_type: str, allowed: tuple):
        message = f"Content-Type '{content_type}' not allowed. Allowed: {allowed}"
        super().__init__(message, {"url": url, "content_type": content_type, "allowed": allowed})


class ZipSlipError(SecurityViolationError):
    """Raised when zip-slip attack is detected."""
    
    def __init__(self, archive: str, member: str):
        message = f"Potential zip-slip attack detected in '{archive}': member '{member}'"
        super().__init__(message, {"archive": archive, "member": member})


class ProcessingError(KriraChunkerError):
    """Raised when document processing fails."""
    pass


class OCRRequiredError(ProcessingError):
    """Raised when OCR is required but not available."""
    
    def __init__(self, source: str, avg_chars_per_page: float):
        message = (
            f"PDF '{source}' appears to be scanned (avg {avg_chars_per_page:.1f} chars/page). "
            "OCR is required but not integrated. Consider using an OCR tool first."
        )
        super().__init__(message, {"source": source, "avg_chars_per_page": avg_chars_per_page})


class EmptyDocumentError(ProcessingError):
    """Raised when document contains no extractable content."""
    
    def __init__(self, source: str):
        message = f"Document '{source}' contains no extractable text content"
        super().__init__(message, {"source": source})
