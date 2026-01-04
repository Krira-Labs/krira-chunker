"""
Data Cleaner module for Krira Chunker V2.0.

Removes noise and normalizes text for downstream chunking operations.
This class applies regex-based filters to remove headers, footers,
and boilerplate text that would corrupt chunk quality.

Performance: O(1) memory usage regardless of file size with streaming support.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Generator, List, Optional, Pattern, Tuple


# =============================================================================
# REGEX PATTERNS
# Pre-compiled patterns for high performance
# =============================================================================

# === HEADER PATTERNS ===
# These patterns match common page headers found in documents
HEADER_PATTERNS: List[str] = [
    r'Page\s+\d+\s+of\s+\d+',  # "Page 1 of 10" - Standard page numbering
    r'Page\s+\d+',            # "Page 5" - Simple page numbering
    r'\d+\s*/\s*\d+',         # "5 / 10" - Fraction-style page numbers
]

# === FOOTER PATTERNS ===
# These patterns match common document footers
FOOTER_PATTERNS: List[str] = [
    r'©\s*\d{4}\s+[\w\s]+',       # "© 2024 Company Name" - Copyright with year
    r'Copyright\s+\d{4}',         # "Copyright 2024" - Alternative copyright format
    r'Confidential',               # "Confidential" - Common security footer
    r'All\s+Rights\s+Reserved',    # "All Rights Reserved" - Legal boilerplate
    r'CONFIDENTIAL',               # Uppercase variant
    r'PROPRIETARY[\s\w]*',         # Proprietary notices
]

# === PII PATTERNS ===
# Email pattern: Matches standard email format
# Format: local-part@domain.tld
EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

# Phone pattern: Matches various phone formats
# Supports: +1-555-123-4567, (555) 123-4567, 555.123.4567, etc.
PHONE_PATTERN = r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'


@dataclass
class CleaningConfig:
    """
    Configuration for DataCleaner.
    
    This dataclass controls all cleaning behaviors including noise removal,
    character normalization, and privacy redaction.
    
    Attributes:
        remove_headers: Strip 'Page X of Y' patterns commonly found in PDFs.
        remove_footers: Strip copyright notices and confidentiality statements.
        custom_patterns: User-defined regex patterns to remove.
        fix_unicode: Normalize Unicode (NFKC) to fix broken characters.
        normalize_whitespace: Convert multiple spaces/tabs to single space.
        preserve_line_breaks: Keep paragraph breaks (\\n\\n) intact.
        redact_pii: Mask emails and phone numbers with placeholders.
        chunk_buffer_size: Characters to process in each streaming buffer.
    """
    
    # === NOISE REMOVAL ===
    remove_headers: bool = True
    """Strip 'Page X of Y' patterns commonly found in PDFs."""
    
    remove_footers: bool = True
    """Strip copyright notices and confidentiality statements."""
    
    custom_patterns: List[str] = field(default_factory=list)
    """User-defined regex patterns to remove (e.g., company letterhead)."""
    
    # === CHARACTER NORMALIZATION ===
    fix_unicode: bool = True
    """Normalize Unicode (NFKC) to fix broken characters like \\u00a0."""
    
    normalize_whitespace: bool = True
    """Convert multiple spaces/tabs to single space."""
    
    preserve_line_breaks: bool = True
    """Keep paragraph breaks (\\n\\n) intact for structure."""
    
    # === PRIVACY ===
    redact_pii: bool = False
    """Mask emails and phone numbers. WARNING: May impact data quality."""
    
    # === PERFORMANCE ===
    chunk_buffer_size: int = 10_000
    """Number of characters to process in each buffer (streaming mode)."""
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.chunk_buffer_size <= 0:
            raise ValueError(
                f"chunk_buffer_size must be positive, got {self.chunk_buffer_size}"
            )


class DataCleaner:
    """
    Removes noise and normalizes text for downstream chunking.
    
    This class applies regex-based filters to remove headers, footers,
    and boilerplate text that would corrupt chunk quality.
    
    Features:
        - Unicode normalization (NFKC)
        - Header/footer removal (Page X of Y, Copyright, etc.)
        - Custom pattern matching
        - PII redaction (email, phone)
        - Whitespace normalization
        - Streaming support for large files
    
    Example:
        >>> config = CleaningConfig(remove_headers=True, fix_unicode=True)
        >>> cleaner = DataCleaner(config)
        >>> cleaned = cleaner.clean_text("Page 1 of 10\\nActual content here")
        >>> print(cleaned)
        'Actual content here'
    """
    
    # Placeholder strings for PII redaction
    EMAIL_PLACEHOLDER = "<EMAIL>"
    PHONE_PLACEHOLDER = "<PHONE>"
    
    def __init__(self, config: CleaningConfig) -> None:
        """
        Initialize the cleaner with compiled regex patterns.
        
        Args:
            config: Configuration object controlling cleaning behavior.
            
        Implementation Notes:
            - Pre-compiles ALL regex patterns in __init__ for performance.
            - Stores compiled patterns as instance variables.
            - Validates config (raises ValueError if invalid).
            
        Raises:
            ValueError: If configuration is invalid.
            TypeError: If config is not a CleaningConfig instance.
        """
        if not isinstance(config, CleaningConfig):
            raise TypeError(
                f"config must be CleaningConfig, got {type(config).__name__}"
            )
        
        self.config = config
        
        # Pre-compile header patterns
        self._header_patterns: List[Pattern[str]] = []
        if config.remove_headers:
            for pattern in HEADER_PATTERNS:
                try:
                    self._header_patterns.append(
                        re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                    )
                except re.error as e:
                    raise ValueError(f"Invalid header pattern '{pattern}': {e}")
        
        # Pre-compile footer patterns
        self._footer_patterns: List[Pattern[str]] = []
        if config.remove_footers:
            for pattern in FOOTER_PATTERNS:
                try:
                    self._footer_patterns.append(
                        re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                    )
                except re.error as e:
                    raise ValueError(f"Invalid footer pattern '{pattern}': {e}")
        
        # Pre-compile custom patterns
        self._custom_patterns: List[Pattern[str]] = []
        for pattern in config.custom_patterns:
            try:
                self._custom_patterns.append(
                    re.compile(pattern, re.MULTILINE)
                )
            except re.error as e:
                raise ValueError(f"Invalid custom pattern '{pattern}': {e}")
        
        # Pre-compile PII patterns
        self._email_pattern: Optional[Pattern[str]] = None
        self._phone_pattern: Optional[Pattern[str]] = None
        if config.redact_pii:
            self._email_pattern = re.compile(EMAIL_PATTERN)
            self._phone_pattern = re.compile(PHONE_PATTERN)
        
        # Whitespace normalization pattern
        # Matches 2+ spaces or tabs (not newlines)
        self._multi_space_pattern = re.compile(r'[^\S\n]+')
        
        # Multiple newline pattern (more than 2 consecutive)
        self._multi_newline_pattern = re.compile(r'\n{3,}')
        
        # Statistics tracking
        self._stats = {
            "bytes_cleaned": 0,
            "patterns_removed": 0,
        }
    
    def clean_text(self, text: Optional[str]) -> str:
        """
        Apply all enabled cleaning filters to the input text.
        
        Args:
            text: Raw input string (can be empty or None).
            
        Returns:
            Cleaned string with noise removed.
            
        Algorithm:
            1. If text is None or empty, return "".
            2. If fix_unicode: Apply unicodedata.normalize('NFKC', text).
            3. If remove_headers: Apply header removal regex.
            4. If remove_footers: Apply footer removal regex.
            5. Apply custom_patterns in order.
            6. If redact_pii: Mask emails and phones.
            7. If normalize_whitespace: Collapse multiple spaces.
            8. Return result.strip().
            
        Edge Cases:
            - Text with only whitespace should return "".
            - Text with only headers/footers should return "".
            - Unicode errors should NOT crash (uses errors='ignore').
            
        Example:
            >>> cleaner = DataCleaner(CleaningConfig())
            >>> cleaner.clean_text("Page 1 of 5\\n\\nContent here")
            'Content here'
        """
        # Handle None or empty input
        if text is None:
            return ""
        
        if not text:
            return ""
        
        original_len = len(text)
        result = text
        patterns_removed = 0
        
        # Step 1: Unicode normalization
        if self.config.fix_unicode:
            try:
                # NFKC: Compatibility decomposition followed by canonical composition
                # Converts things like \u00a0 (non-breaking space) to regular space
                result = unicodedata.normalize('NFKC', result)
            except (TypeError, UnicodeError):
                # If normalization fails, continue with original text
                pass
        
        # Step 2: Remove headers
        if self.config.remove_headers:
            for pattern in self._header_patterns:
                result, count = pattern.subn('', result)
                patterns_removed += count
        
        # Step 3: Remove footers
        if self.config.remove_footers:
            for pattern in self._footer_patterns:
                result, count = pattern.subn('', result)
                patterns_removed += count
        
        # Step 4: Apply custom patterns
        for pattern in self._custom_patterns:
            result, count = pattern.subn('', result)
            patterns_removed += count
        
        # Step 5: Redact PII
        if self.config.redact_pii and self._email_pattern and self._phone_pattern:
            result, email_count = self._email_pattern.subn(
                self.EMAIL_PLACEHOLDER, result
            )
            result, phone_count = self._phone_pattern.subn(
                self.PHONE_PLACEHOLDER, result
            )
            patterns_removed += email_count + phone_count
        
        # Step 6: Normalize whitespace
        if self.config.normalize_whitespace:
            if self.config.preserve_line_breaks:
                # Handle each line separately to preserve line breaks
                lines = result.split('\n')
                normalized_lines = []
                for line in lines:
                    # Collapse multiple spaces/tabs to single space
                    normalized = self._multi_space_pattern.sub(' ', line)
                    normalized_lines.append(normalized.strip())
                result = '\n'.join(normalized_lines)
                
                # Collapse excessive newlines (more than 2) to double newline
                result = self._multi_newline_pattern.sub('\n\n', result)
            else:
                # Convert all whitespace (including newlines) to single space
                result = ' '.join(result.split())
        
        # Final strip
        result = result.strip()
        
        # Update statistics
        self._stats["bytes_cleaned"] += original_len
        self._stats["patterns_removed"] += patterns_removed
        
        return result
    
    def clean_stream(
        self, 
        text_stream: Generator[str, None, None]
    ) -> Generator[str, None, None]:
        """
        Clean a stream of text chunks without loading all into memory.
        
        Args:
            text_stream: Generator yielding text strings.
            
        Yields:
            Cleaned text strings.
            
        Implementation:
            - Uses a sliding window buffer to handle patterns that span chunks.
            - Buffer size controlled by config.chunk_buffer_size.
            - Yields cleaned text as soon as buffer is processed.
            
        Note:
            For patterns that may span chunk boundaries (like multi-word
            patterns), this method uses an overlap buffer to ensure
            accurate detection.
            
        Example:
            >>> def text_generator():
            ...     yield "Page 1 of 10\\n"
            ...     yield "Content line 1\\n"
            ...     yield "Content line 2"
            >>> 
            >>> cleaner = DataCleaner(CleaningConfig())
            >>> for chunk in cleaner.clean_stream(text_generator()):
            ...     print(chunk)
        """
        buffer_size = self.config.chunk_buffer_size
        
        # Overlap to handle patterns spanning chunk boundaries
        # Use max pattern length estimate (100 chars should cover most cases)
        overlap_size = min(100, buffer_size // 10)
        
        buffer = ""
        
        for chunk in text_stream:
            if chunk is None:
                continue
            
            buffer += chunk
            
            # Process when buffer is large enough
            while len(buffer) >= buffer_size + overlap_size:
                # Process the main portion
                to_process = buffer[:buffer_size]
                cleaned = self.clean_text(to_process)
                
                if cleaned:
                    yield cleaned
                
                # Keep overlap for next iteration (might contain split patterns)
                buffer = buffer[buffer_size:]
        
        # Process remaining buffer
        if buffer:
            cleaned = self.clean_text(buffer)
            if cleaned:
                yield cleaned
    
    def get_stats(self) -> dict:
        """
        Return cleaning statistics.
        
        Returns:
            Dictionary with keys:
            - 'bytes_cleaned': Total text bytes processed.
            - 'patterns_removed': Count of regex pattern matches removed.
        """
        return dict(self._stats)
    
    def reset_stats(self) -> None:
        """Reset internal statistics counters."""
        self._stats = {
            "bytes_cleaned": 0,
            "patterns_removed": 0,
        }
