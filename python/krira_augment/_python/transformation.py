"""
Data Transformer module for Krira Chunker V2.0.

Standardizes diverse input formats into Markdown for optimal chunking.
Converts HTML tables, JSON objects, and raw CSV into clean Markdown format.

Performance: O(1) memory usage with streaming support for large files.
"""

import csv
import json
from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Dict, Generator, List, Literal, Optional, Union


@dataclass
class TransformConfig:
    """
    Configuration for DataTransformer.
    
    This dataclass controls transformation behaviors including output format,
    table handling, and JSON flattening depth.
    
    Attributes:
        output_format: Target format for transformation ("markdown" or "plain_text").
        preserve_tables: Convert tables to Markdown format instead of flattening.
        max_table_columns: Maximum columns to preserve in tables.
        json_indent: Add indentation when flattening JSON.
        max_json_depth: Maximum nesting depth to preserve.
    """
    
    # === TARGET FORMAT ===
    output_format: Literal["markdown", "plain_text"] = "markdown"
    """Target format for transformation."""
    
    # === TABLE HANDLING ===
    preserve_tables: bool = True
    """Convert tables to Markdown table format instead of flattening."""
    
    max_table_columns: int = 10
    """Maximum columns to preserve. Wider tables are summarized."""
    
    # === JSON HANDLING ===
    json_indent: bool = True
    """Add indentation when flattening JSON."""
    
    max_json_depth: int = 3
    """Maximum nesting depth to preserve. Deeper objects are truncated."""
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.output_format not in ("markdown", "plain_text"):
            raise ValueError(
                f"output_format must be 'markdown' or 'plain_text', "
                f"got '{self.output_format}'"
            )
        if self.max_table_columns <= 0:
            raise ValueError(
                f"max_table_columns must be positive, got {self.max_table_columns}"
            )
        if self.max_json_depth <= 0:
            raise ValueError(
                f"max_json_depth must be positive, got {self.max_json_depth}"
            )


class DataTransformer:
    """
    Standardizes diverse input formats into Markdown.
    
    Converts HTML tables, JSON objects, and raw CSV into clean
    Markdown format for optimal chunking.
    
    Features:
        - CSV to Markdown table conversion
        - JSON to Markdown list flattening
        - Nested object support
        - Configurable column limits
        - Plain text fallback
    
    Example:
        >>> config = TransformConfig(output_format="markdown")
        >>> transformer = DataTransformer(config)
        >>> md = transformer.csv_to_markdown("Name,Age\\nAlice,30")
        >>> print(md)
        | Name | Age |
        |------|-----|
        | Alice | 30 |
    """
    
    def __init__(self, config: TransformConfig) -> None:
        """
        Initialize the transformer with configuration.
        
        Args:
            config: Configuration object controlling transformation behavior.
            
        Raises:
            TypeError: If config is not a TransformConfig instance.
        """
        if not isinstance(config, TransformConfig):
            raise TypeError(
                f"config must be TransformConfig, got {type(config).__name__}"
            )
        
        self.config = config
        
        # Statistics tracking
        self._stats = {
            "tables_transformed": 0,
            "json_objects_transformed": 0,
            "rows_processed": 0,
        }
    
    def csv_to_markdown(
        self, 
        csv_text: str, 
        has_header: bool = True
    ) -> str:
        """
        Convert CSV text to Markdown table format.
        
        Args:
            csv_text: Raw CSV string.
            has_header: Whether first row is a header.
            
        Returns:
            Markdown table string.
            
        Example Input:
            "Name,Age\\nAlice,30\\nBob,25"
            
        Example Output:
            | Name | Age |
            |------|-----|
            | Alice | 30 |
            | Bob | 25 |
            
        Edge Cases:
            - Empty CSV returns "".
            - Cells with commas are handled correctly (using csv.reader).
            - Cells with line breaks are stripped.
            - If column count varies, pads with empty cells.
        """
        if not csv_text or not csv_text.strip():
            return ""
        
        try:
            reader = csv.reader(StringIO(csv_text))
            rows = list(reader)
        except csv.Error:
            # If parsing fails, return original text
            return csv_text
        
        if not rows:
            return ""
        
        # Track statistics
        self._stats["tables_transformed"] += 1
        self._stats["rows_processed"] += len(rows)
        
        # Determine max columns (for normalization)
        max_cols = max(len(row) for row in rows) if rows else 0
        
        if max_cols == 0:
            return ""
        
        # Apply column limit
        effective_cols = min(max_cols, self.config.max_table_columns)
        truncated = max_cols > self.config.max_table_columns
        
        # Normalize rows (pad with empty cells if needed)
        normalized_rows = []
        for row in rows:
            # Clean cell contents (strip newlines within cells)
            cleaned_row = [
                str(cell).replace('\n', ' ').replace('\r', '').strip()
                for cell in row[:effective_cols]
            ]
            # Pad if needed
            while len(cleaned_row) < effective_cols:
                cleaned_row.append("")
            normalized_rows.append(cleaned_row)
        
        if not normalized_rows:
            return ""
        
        if self.config.output_format == "plain_text":
            return self._csv_to_plain_text(normalized_rows, has_header, truncated)
        
        # Build Markdown table
        result_lines = []
        
        if has_header:
            # First row is header
            header_row = normalized_rows[0]
            data_rows = normalized_rows[1:]
            
            # Generate header names if first row is empty
            if not any(cell.strip() for cell in header_row):
                header_row = [f"Column_{i+1}" for i in range(effective_cols)]
        else:
            # Generate column headers
            header_row = [f"Column_{i+1}" for i in range(effective_cols)]
            data_rows = normalized_rows
        
        # Add note if truncated
        if truncated:
            result_lines.append(
                f"*Note: Table truncated from {max_cols} to "
                f"{effective_cols} columns*\n"
            )
        
        # Calculate column widths for alignment
        col_widths = []
        for i in range(effective_cols):
            max_width = len(header_row[i]) if i < len(header_row) else 0
            for row in data_rows:
                if i < len(row):
                    max_width = max(max_width, len(row[i]))
            col_widths.append(max(max_width, 3))  # Minimum width of 3
        
        # Build header line
        header_cells = [
            f" {header_row[i].ljust(col_widths[i])} " 
            for i in range(effective_cols)
        ]
        result_lines.append("|" + "|".join(header_cells) + "|")
        
        # Build separator line
        separator_cells = [
            "-" * (col_widths[i] + 2)
            for i in range(effective_cols)
        ]
        result_lines.append("|" + "|".join(separator_cells) + "|")
        
        # Build data rows
        for row in data_rows:
            data_cells = [
                f" {row[i].ljust(col_widths[i]) if i < len(row) else ''.ljust(col_widths[i])} "
                for i in range(effective_cols)
            ]
            result_lines.append("|" + "|".join(data_cells) + "|")
        
        return "\n".join(result_lines)
    
    def _csv_to_plain_text(
        self, 
        rows: List[List[str]], 
        has_header: bool,
        truncated: bool
    ) -> str:
        """Convert normalized CSV rows to plain text format."""
        if not rows:
            return ""
        
        result_lines = []
        
        if truncated:
            result_lines.append("[Table truncated]")
        
        if has_header and rows:
            header = rows[0]
            data_rows = rows[1:]
        else:
            header = [f"Column_{i+1}" for i in range(len(rows[0]))]
            data_rows = rows
        
        for row in data_rows:
            parts = []
            for i, cell in enumerate(row):
                if cell.strip():
                    col_name = header[i] if i < len(header) else f"Column_{i+1}"
                    parts.append(f"{col_name}: {cell}")
            if parts:
                result_lines.append(" | ".join(parts))
        
        return "\n".join(result_lines)
    
    def json_to_markdown(self, json_text: str) -> str:
        """
        Flatten JSON object into Markdown list.
        
        Args:
            json_text: JSON string (object or array).
            
        Returns:
            Markdown formatted text.
            
        Example Input:
            {"user": "Alice", "age": 30, "city": "NYC"}
            
        Example Output:
            - **user**: Alice
            - **age**: 30
            - **city**: NYC
            
        Edge Cases:
            - Nested objects indent sub-bullets.
            - Arrays are numbered lists.
            - Null values display as "None".
            - Invalid JSON returns original text with warning comment.
        """
        if not json_text or not json_text.strip():
            return ""
        
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as e:
            # Return original text with warning
            return f"<!-- Invalid JSON: {e} -->\n{json_text}"
        
        # Track statistics
        self._stats["json_objects_transformed"] += 1
        
        if self.config.output_format == "plain_text":
            return self._json_to_plain_text(data)
        
        return self._format_json_value(data, depth=0)
    
    def _format_json_value(
        self, 
        value: Any, 
        depth: int = 0,
        is_array_item: bool = False
    ) -> str:
        """
        Recursively format a JSON value as Markdown.
        
        Args:
            value: The JSON value to format.
            depth: Current nesting depth.
            is_array_item: Whether this is an array item.
            
        Returns:
            Markdown formatted string.
        """
        indent = "  " * depth if self.config.json_indent else ""
        
        # Handle depth limit
        if depth >= self.config.max_json_depth:
            if isinstance(value, (dict, list)):
                return f"{indent}[...truncated...]"
        
        # Handle None/null
        if value is None:
            return "None"
        
        # Handle primitives
        if isinstance(value, bool):
            return str(value).lower()
        
        if isinstance(value, (int, float)):
            return str(value)
        
        if isinstance(value, str):
            return value
        
        # Handle arrays
        if isinstance(value, list):
            if not value:
                return "[]"
            
            lines = []
            for i, item in enumerate(value, 1):
                if isinstance(item, dict):
                    # Nested object in array
                    formatted = self._format_json_value(item, depth + 1)
                    lines.append(f"{indent}{i}. ")
                    # Indent the formatted content
                    for line in formatted.split('\n'):
                        lines.append(f"{indent}   {line}")
                elif isinstance(item, list):
                    # Nested array
                    formatted = self._format_json_value(item, depth + 1)
                    lines.append(f"{indent}{i}. {formatted}")
                else:
                    # Primitive value
                    formatted = self._format_json_value(item, depth)
                    lines.append(f"{indent}{i}. {formatted}")
            
            return "\n".join(lines)
        
        # Handle objects
        if isinstance(value, dict):
            if not value:
                return "{}"
            
            lines = []
            for key, val in value.items():
                if isinstance(val, dict):
                    # Nested object
                    lines.append(f"{indent}- **{key}**:")
                    formatted = self._format_json_value(val, depth + 1)
                    for line in formatted.split('\n'):
                        lines.append(f"  {line}")
                elif isinstance(val, list):
                    # Nested array
                    lines.append(f"{indent}- **{key}**:")
                    formatted = self._format_json_value(val, depth + 1)
                    for line in formatted.split('\n'):
                        lines.append(f"  {line}")
                else:
                    # Primitive value
                    formatted = self._format_json_value(val, depth)
                    lines.append(f"{indent}- **{key}**: {formatted}")
            
            return "\n".join(lines)
        
        # Fallback for unknown types
        return str(value)
    
    def _json_to_plain_text(self, data: Any, depth: int = 0) -> str:
        """Convert JSON data to plain text format."""
        indent = "  " * depth if self.config.json_indent else ""
        
        if depth >= self.config.max_json_depth:
            if isinstance(data, (dict, list)):
                return f"{indent}[...truncated...]"
        
        if data is None:
            return "None"
        
        if isinstance(data, bool):
            return str(data).lower()
        
        if isinstance(data, (int, float, str)):
            return str(data)
        
        if isinstance(data, list):
            if not data:
                return "[]"
            lines = []
            for i, item in enumerate(data, 1):
                formatted = self._json_to_plain_text(item, depth + 1)
                lines.append(f"{indent}{i}. {formatted}")
            return "\n".join(lines)
        
        if isinstance(data, dict):
            if not data:
                return "{}"
            lines = []
            for key, val in data.items():
                formatted = self._json_to_plain_text(val, depth + 1)
                if '\n' in formatted:
                    lines.append(f"{indent}{key}:")
                    lines.append(formatted)
                else:
                    lines.append(f"{indent}{key}: {formatted}")
            return "\n".join(lines)
        
        return str(data)
    
    def transform_row(
        self, 
        row: Dict[str, Any],
        format_as: Literal["markdown", "plain_text", "auto"] = "auto"
    ) -> str:
        """
        Transform a single row (dict) to formatted text.
        
        Args:
            row: Dictionary representing a data row.
            format_as: Output format override.
            
        Returns:
            Formatted text representation of the row.
        """
        if not row:
            return ""
        
        output_format = format_as if format_as != "auto" else self.config.output_format
        
        if output_format == "markdown":
            parts = []
            for key, value in row.items():
                if value is not None and str(value).strip():
                    parts.append(f"**{key}**: {value}")
            return " | ".join(parts)
        else:
            parts = []
            for key, value in row.items():
                if value is not None and str(value).strip():
                    parts.append(f"{key}: {value}")
            return " | ".join(parts)
    
    def transform_rows(
        self,
        rows: Generator[Dict[str, Any], None, None]
    ) -> Generator[str, None, None]:
        """
        Transform a stream of rows to formatted text.
        
        Args:
            rows: Generator yielding row dictionaries.
            
        Yields:
            Formatted text for each row.
        """
        for row in rows:
            transformed = self.transform_row(row)
            if transformed:
                yield transformed
    
    def excel_row_to_text(
        self, 
        headers: List[str], 
        row_values: List[Any]
    ) -> str:
        """
        Convert an Excel row to formatted text.
        
        Args:
            headers: Column headers.
            row_values: Row values (same length as headers).
            
        Returns:
            Formatted text representation.
        """
        if not headers or not row_values:
            return ""
        
        parts = []
        for header, value in zip(headers, row_values):
            if value is not None:
                str_value = str(value).strip()
                if str_value:
                    if self.config.output_format == "markdown":
                        parts.append(f"**{header}**: {str_value}")
                    else:
                        parts.append(f"{header}: {str_value}")
        
        return " | ".join(parts)
    
    def get_stats(self) -> Dict[str, int]:
        """
        Return transformation statistics.
        
        Returns:
            Dictionary with transformation counts.
        """
        return dict(self._stats)
    
    def reset_stats(self) -> None:
        """Reset internal statistics counters."""
        self._stats = {
            "tables_transformed": 0,
            "json_objects_transformed": 0,
            "rows_processed": 0,
        }
