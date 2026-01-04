//! Data Transformer module for Krira Augment
//!
//! Standardizes diverse input formats into Markdown.
//! Converts CSV data and JSON objects into clean Markdown format
//! for optimal chunking.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;
use std::collections::HashMap;

use crate::errors::KriraError;

// =============================================================================
// TransformConfig
// =============================================================================

/// Configuration for DataTransformer.
///
/// This class controls transformation behaviors including output format,
/// table handling, and JSON flattening depth.
#[pyclass]
#[derive(Clone, Debug)]
pub struct TransformConfig {
    /// Target format for transformation ("markdown" or "plain_text")
    #[pyo3(get, set)]
    pub output_format: String,

    /// Convert tables to Markdown table format instead of flattening
    #[pyo3(get, set)]
    pub preserve_tables: bool,

    /// Maximum columns to preserve. Wider tables are summarized
    #[pyo3(get, set)]
    pub max_table_columns: usize,

    /// Add indentation when flattening JSON
    #[pyo3(get, set)]
    pub json_indent: bool,

    /// Maximum nesting depth to preserve. Deeper objects are truncated
    #[pyo3(get, set)]
    pub max_json_depth: usize,
}

#[pymethods]
impl TransformConfig {
    /// Create a new TransformConfig with default values
    #[new]
    #[pyo3(signature = (
        output_format = "markdown",
        preserve_tables = true,
        max_table_columns = 10,
        json_indent = true,
        max_json_depth = 3
    ))]
    pub fn new(
        output_format: &str,
        preserve_tables: bool,
        max_table_columns: usize,
        json_indent: bool,
        max_json_depth: usize,
    ) -> PyResult<Self> {
        if output_format != "markdown" && output_format != "plain_text" {
            return Err(PyErr::from(KriraError::ConfigError(format!(
                "output_format must be 'markdown' or 'plain_text', got '{}'",
                output_format
            ))));
        }

        if max_table_columns == 0 {
            return Err(PyErr::from(KriraError::ConfigError(
                "max_table_columns must be positive".to_string(),
            )));
        }

        if max_json_depth == 0 {
            return Err(PyErr::from(KriraError::ConfigError(
                "max_json_depth must be positive".to_string(),
            )));
        }

        Ok(Self {
            output_format: output_format.to_string(),
            preserve_tables,
            max_table_columns,
            json_indent,
            max_json_depth,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "TransformConfig(output_format='{}', preserve_tables={}, max_table_columns={})",
            self.output_format, self.preserve_tables, self.max_table_columns
        )
    }
}

impl Default for TransformConfig {
    fn default() -> Self {
        Self {
            output_format: "markdown".to_string(),
            preserve_tables: true,
            max_table_columns: 10,
            json_indent: true,
            max_json_depth: 3,
        }
    }
}

// =============================================================================
// DataTransformer
// =============================================================================

/// Standardizes diverse input formats into Markdown.
///
/// Converts CSV data and JSON objects into clean Markdown format
/// for optimal chunking.
#[pyclass]
pub struct DataTransformer {
    config: TransformConfig,
    tables_transformed: usize,
    json_objects_transformed: usize,
    rows_processed: usize,
}

#[pymethods]
impl DataTransformer {
    /// Initialize the transformer with configuration
    #[new]
    pub fn new(config: TransformConfig) -> PyResult<Self> {
        Ok(Self {
            config,
            tables_transformed: 0,
            json_objects_transformed: 0,
            rows_processed: 0,
        })
    }

    /// Convert CSV text to Markdown table format
    ///
    /// # Arguments
    /// * `csv_text` - Raw CSV string
    /// * `has_header` - Whether first row is a header
    ///
    /// # Returns
    /// Markdown table string
    #[pyo3(signature = (csv_text, has_header = true))]
    pub fn csv_to_markdown(&mut self, csv_text: &str, has_header: bool) -> PyResult<String> {
        if csv_text.trim().is_empty() {
            return Ok(String::new());
        }

        // Parse CSV
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .flexible(true)
            .from_reader(csv_text.as_bytes());

        let rows: Vec<Vec<String>> = reader
            .records()
            .filter_map(|r| r.ok())
            .map(|r| r.iter().map(|s| s.replace('\n', " ").trim().to_string()).collect())
            .collect();

        if rows.is_empty() {
            return Ok(String::new());
        }

        self.tables_transformed += 1;
        self.rows_processed += rows.len();

        // Determine max columns (for normalization)
        let max_cols = rows.iter().map(|r| r.len()).max().unwrap_or(0);
        if max_cols == 0 {
            return Ok(String::new());
        }

        // Apply column limit
        let effective_cols = std::cmp::min(max_cols, self.config.max_table_columns);
        let truncated = max_cols > self.config.max_table_columns;

        // Normalize rows (pad with empty cells if needed)
        let normalized_rows: Vec<Vec<String>> = rows
            .iter()
            .map(|row| {
                let mut new_row: Vec<String> = row.iter().take(effective_cols).cloned().collect();
                while new_row.len() < effective_cols {
                    new_row.push(String::new());
                }
                new_row
            })
            .collect();

        if self.config.output_format == "plain_text" {
            return Ok(self.csv_to_plain_text(&normalized_rows, has_header, truncated));
        }

        // Build Markdown table
        let mut lines: Vec<String> = Vec::new();

        let (header_row, data_rows) = if has_header && !normalized_rows.is_empty() {
            let header = &normalized_rows[0];
            let data = &normalized_rows[1..];
            (header.clone(), data.to_vec())
        } else {
            let header: Vec<String> = (0..effective_cols)
                .map(|i| format!("Column_{}", i + 1))
                .collect();
            (header, normalized_rows)
        };

        // Add truncation note
        if truncated {
            lines.push(format!(
                "*Note: Table truncated from {} to {} columns*\n",
                max_cols, effective_cols
            ));
        }

        // Calculate column widths
        let col_widths: Vec<usize> = (0..effective_cols)
            .map(|i| {
                let header_width = header_row.get(i).map(|h| h.len()).unwrap_or(0);
                let data_width = data_rows
                    .iter()
                    .map(|row| row.get(i).map(|c| c.len()).unwrap_or(0))
                    .max()
                    .unwrap_or(0);
                std::cmp::max(std::cmp::max(header_width, data_width), 3)
            })
            .collect();

        // Build header line
        let header_cells: Vec<String> = header_row
            .iter()
            .enumerate()
            .map(|(i, h)| format!(" {:width$} ", h, width = col_widths[i]))
            .collect();
        lines.push(format!("|{}|", header_cells.join("|")));

        // Build separator line
        let separator_cells: Vec<String> = col_widths
            .iter()
            .map(|w| "-".repeat(w + 2))
            .collect();
        lines.push(format!("|{}|", separator_cells.join("|")));

        // Build data rows
        for row in &data_rows {
            let data_cells: Vec<String> = row
                .iter()
                .enumerate()
                .map(|(i, cell)| {
                    let width = col_widths.get(i).copied().unwrap_or(3);
                    format!(" {:width$} ", cell, width = width)
                })
                .collect();
            lines.push(format!("|{}|", data_cells.join("|")));
        }

        Ok(lines.join("\n"))
    }

    /// Flatten JSON object into Markdown list
    ///
    /// # Arguments
    /// * `json_text` - JSON string (object or array)
    ///
    /// # Returns
    /// Markdown formatted text
    pub fn json_to_markdown(&mut self, json_text: &str) -> PyResult<String> {
        if json_text.trim().is_empty() {
            return Ok(String::new());
        }

        let data: Value = match serde_json::from_str(json_text) {
            Ok(v) => v,
            Err(e) => {
                return Ok(format!("<!-- Invalid JSON: {} -->\n{}", e, json_text));
            }
        };

        self.json_objects_transformed += 1;

        if self.config.output_format == "plain_text" {
            Ok(self.json_to_plain_text(&data, 0))
        } else {
            Ok(self.format_json_value(&data, 0))
        }
    }

    /// Transform a single row (dict) to formatted text
    pub fn transform_row(&self, py: Python, row: &PyDict) -> PyResult<String> {
        if row.is_empty() {
            return Ok(String::new());
        }

        let mut parts: Vec<String> = Vec::new();

        for (key, value) in row.iter() {
            let key_str: String = key.extract()?;
            let value_str: String = value.str()?.to_string();

            if !value_str.is_empty() && value_str != "None" {
                if self.config.output_format == "markdown" {
                    parts.push(format!("**{}**: {}", key_str, value_str));
                } else {
                    parts.push(format!("{}: {}", key_str, value_str));
                }
            }
        }

        Ok(parts.join(" | "))
    }

    /// Convert Excel row to formatted text
    pub fn excel_row_to_text(&self, headers: Vec<String>, values: Vec<String>) -> String {
        if headers.is_empty() || values.is_empty() {
            return String::new();
        }

        let parts: Vec<String> = headers
            .iter()
            .zip(values.iter())
            .filter(|(_, v)| !v.trim().is_empty())
            .map(|(h, v)| {
                if self.config.output_format == "markdown" {
                    format!("**{}**: {}", h, v.trim())
                } else {
                    format!("{}: {}", h, v.trim())
                }
            })
            .collect();

        parts.join(" | ")
    }

    /// Return transformation statistics
    pub fn get_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("tables_transformed".to_string(), self.tables_transformed);
        stats.insert(
            "json_objects_transformed".to_string(),
            self.json_objects_transformed,
        );
        stats.insert("rows_processed".to_string(), self.rows_processed);
        stats
    }

    /// Reset internal statistics counters
    pub fn reset_stats(&mut self) {
        self.tables_transformed = 0;
        self.json_objects_transformed = 0;
        self.rows_processed = 0;
    }

    fn __repr__(&self) -> String {
        format!(
            "DataTransformer(tables_transformed={}, json_objects_transformed={})",
            self.tables_transformed, self.json_objects_transformed
        )
    }
}

impl DataTransformer {
    /// Convert normalized CSV rows to plain text format
    fn csv_to_plain_text(
        &self,
        rows: &[Vec<String>],
        has_header: bool,
        truncated: bool,
    ) -> String {
        if rows.is_empty() {
            return String::new();
        }

        let mut lines: Vec<String> = Vec::new();

        if truncated {
            lines.push("[Table truncated]".to_string());
        }

        let (header, data_rows) = if has_header && !rows.is_empty() {
            (rows[0].clone(), &rows[1..])
        } else {
            let h: Vec<String> = (0..rows[0].len())
                .map(|i| format!("Column_{}", i + 1))
                .collect();
            (h, rows)
        };

        for row in data_rows {
            let parts: Vec<String> = row
                .iter()
                .enumerate()
                .filter(|(_, cell)| !cell.trim().is_empty())
                .map(|(i, cell)| {
                    let col_name = header.get(i).cloned().unwrap_or_else(|| format!("Column_{}", i + 1));
                    format!("{}: {}", col_name, cell)
                })
                .collect();

            if !parts.is_empty() {
                lines.push(parts.join(" | "));
            }
        }

        lines.join("\n")
    }

    /// Format JSON value as Markdown recursively
    fn format_json_value(&self, value: &Value, depth: usize) -> String {
        let indent = if self.config.json_indent {
            "  ".repeat(depth)
        } else {
            String::new()
        };

        // Handle depth limit
        if depth >= self.config.max_json_depth {
            if value.is_object() || value.is_array() {
                return format!("{}[...truncated...]", indent);
            }
        }

        match value {
            Value::Null => "None".to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Number(n) => n.to_string(),
            Value::String(s) => s.clone(),
            Value::Array(arr) => {
                if arr.is_empty() {
                    return "[]".to_string();
                }

                let lines: Vec<String> = arr
                    .iter()
                    .enumerate()
                    .map(|(i, item)| {
                        let formatted = self.format_json_value(item, depth + 1);
                        if item.is_object() || item.is_array() {
                            let sub_lines: Vec<String> = formatted
                                .lines()
                                .map(|l| format!("{}   {}", indent, l))
                                .collect();
                            format!("{}{}. \n{}", indent, i + 1, sub_lines.join("\n"))
                        } else {
                            format!("{}{}. {}", indent, i + 1, formatted)
                        }
                    })
                    .collect();

                lines.join("\n")
            }
            Value::Object(obj) => {
                if obj.is_empty() {
                    return "{}".to_string();
                }

                let lines: Vec<String> = obj
                    .iter()
                    .map(|(key, val)| {
                        if val.is_object() || val.is_array() {
                            let formatted = self.format_json_value(val, depth + 1);
                            let sub_lines: Vec<String> = formatted
                                .lines()
                                .map(|l| format!("  {}", l))
                                .collect();
                            format!("{}- **{}**:\n{}", indent, key, sub_lines.join("\n"))
                        } else {
                            let formatted = self.format_json_value(val, depth);
                            format!("{}- **{}**: {}", indent, key, formatted)
                        }
                    })
                    .collect();

                lines.join("\n")
            }
        }
    }

    /// Format JSON value as plain text
    fn json_to_plain_text(&self, value: &Value, depth: usize) -> String {
        let indent = if self.config.json_indent {
            "  ".repeat(depth)
        } else {
            String::new()
        };

        if depth >= self.config.max_json_depth {
            if value.is_object() || value.is_array() {
                return format!("{}[...truncated...]", indent);
            }
        }

        match value {
            Value::Null => "None".to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Number(n) => n.to_string(),
            Value::String(s) => s.clone(),
            Value::Array(arr) => {
                if arr.is_empty() {
                    return "[]".to_string();
                }
                arr.iter()
                    .enumerate()
                    .map(|(i, item)| {
                        let formatted = self.json_to_plain_text(item, depth + 1);
                        format!("{}{}. {}", indent, i + 1, formatted)
                    })
                    .collect::<Vec<String>>()
                    .join("\n")
            }
            Value::Object(obj) => {
                if obj.is_empty() {
                    return "{}".to_string();
                }
                obj.iter()
                    .map(|(key, val)| {
                        let formatted = self.json_to_plain_text(val, depth + 1);
                        if formatted.contains('\n') {
                            format!("{}{}:\n{}", indent, key, formatted)
                        } else {
                            format!("{}{}: {}", indent, key, formatted)
                        }
                    })
                    .collect::<Vec<String>>()
                    .join("\n")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csv_to_markdown() {
        let config = TransformConfig::default();
        let mut transformer = DataTransformer::new(config).unwrap();

        let result = transformer.csv_to_markdown("Name,Age\nAlice,30\nBob,25", true).unwrap();
        assert!(result.contains("| Name"));
        assert!(result.contains("| Alice"));
        assert!(result.contains("| 30"));
    }

    #[test]
    fn test_json_to_markdown() {
        let config = TransformConfig::default();
        let mut transformer = DataTransformer::new(config).unwrap();

        let result = transformer
            .json_to_markdown(r#"{"name": "Alice", "age": 30}"#)
            .unwrap();
        assert!(result.contains("**name**"));
        assert!(result.contains("Alice"));
    }

    #[test]
    fn test_empty_input() {
        let config = TransformConfig::default();
        let mut transformer = DataTransformer::new(config).unwrap();

        assert_eq!(transformer.csv_to_markdown("", true).unwrap(), "");
        assert_eq!(transformer.json_to_markdown("").unwrap(), "");
    }
}
