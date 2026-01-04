//! Pipeline Orchestrator for Krira Augment
//!
//! Orchestrates the Clean -> Transform -> Chunk workflow.
//! This is the main entry point for users processing CSV and XLSX files.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use crate::chunker::{FastChunker, HybridBoundaryChunker, clean_text, stable_id};
use crate::cleaning::{CleaningConfig, DataCleaner};
use crate::config::{ChunkConfig, ChunkStrategy};
use crate::errors::KriraError;
use crate::transformation::{DataTransformer, TransformConfig};

// =============================================================================
// PipelineConfig
// =============================================================================

/// Master configuration for the full Clean -> Transform -> Chunk pipeline.
#[pyclass]
#[derive(Clone)]
pub struct PipelineConfig {
    /// Configuration for DataCleaner
    #[pyo3(get)]
    pub cleaning_config: CleaningConfig,

    /// Configuration for DataTransformer
    #[pyo3(get)]
    pub transform_config: TransformConfig,

    /// Configuration for chunking
    #[pyo3(get)]
    pub chunk_config: ChunkConfig,

    /// Number of rows to process per batch for CSV files
    #[pyo3(get, set)]
    pub csv_batch_rows: usize,

    /// Number of rows to process per batch for XLSX files
    #[pyo3(get, set)]
    pub xlsx_batch_rows: usize,

    /// Log progress status every N rows processed
    #[pyo3(get, set)]
    pub log_progress_every: usize,
}

#[pymethods]
impl PipelineConfig {
    /// Create a new PipelineConfig with default values
    #[new]
    #[pyo3(signature = (
        cleaning_config = None,
        transform_config = None,
        chunk_config = None,
        csv_batch_rows = 50000,
        xlsx_batch_rows = 25000,
        log_progress_every = 100000
    ))]
    pub fn new(
        cleaning_config: Option<CleaningConfig>,
        transform_config: Option<TransformConfig>,
        chunk_config: Option<ChunkConfig>,
        csv_batch_rows: usize,
        xlsx_batch_rows: usize,
        log_progress_every: usize,
    ) -> PyResult<Self> {
        if csv_batch_rows == 0 {
            return Err(PyErr::from(KriraError::ConfigError(
                "csv_batch_rows must be positive".to_string(),
            )));
        }

        if xlsx_batch_rows == 0 {
            return Err(PyErr::from(KriraError::ConfigError(
                "xlsx_batch_rows must be positive".to_string(),
            )));
        }

        Ok(Self {
            cleaning_config: cleaning_config.unwrap_or_default(),
            transform_config: transform_config.unwrap_or_default(),
            chunk_config: chunk_config.unwrap_or_default(),
            csv_batch_rows,
            xlsx_batch_rows,
            log_progress_every,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "PipelineConfig(csv_batch_rows={}, xlsx_batch_rows={}, chunk_strategy='{}')",
            self.csv_batch_rows, self.xlsx_batch_rows, self.chunk_config.chunk_strategy()
        )
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            cleaning_config: CleaningConfig::default(),
            transform_config: TransformConfig::default(),
            chunk_config: ChunkConfig::default(),
            csv_batch_rows: 50_000,
            xlsx_batch_rows: 25_000,
            log_progress_every: 100_000,
        }
    }
}

// =============================================================================
// KriraPipeline
// =============================================================================

/// Orchestrates Clean -> Transform -> Chunk workflow.
///
/// This is the main entry point for users. It chains DataCleaner,
/// DataTransformer, and chunking in sequence.
#[pyclass]
pub struct KriraPipeline {
    config: PipelineConfig,
    cleaner: DataCleaner,
    transformer: DataTransformer,

    // Statistics
    rows_processed: usize,
    chunks_created: usize,
    bytes_cleaned: usize,
    patterns_removed: usize,
    files_processed: usize,
}

#[pymethods]
impl KriraPipeline {
    /// Initialize all pipeline components
    #[new]
    pub fn new(config: PipelineConfig) -> PyResult<Self> {
        let cleaner = DataCleaner::new(config.cleaning_config.clone())?;
        let transformer = DataTransformer::new(config.transform_config.clone())?;

        Ok(Self {
            config,
            cleaner,
            transformer,
            rows_processed: 0,
            chunks_created: 0,
            bytes_cleaned: 0,
            patterns_removed: 0,
            files_processed: 0,
        })
    }

    /// Process a file through the full pipeline
    ///
    /// Supports CSV and XLSX files
    pub fn process_file(&mut self, py: Python, file_path: &str) -> PyResult<PyObject> {
        let path = Path::new(file_path);

        // Validate file exists
        if !path.exists() {
            return Err(PyErr::from(KriraError::FileNotFound(file_path.to_string())));
        }

        // Detect file extension
        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        self.files_processed += 1;

        match extension.as_str() {
            "csv" => self.process_csv_file(py, file_path),
            "xlsx" => Err(PyErr::from(KriraError::UnsupportedFormat(
                "XLSX support requires openpyxl. Use Python wrapper.".to_string(),
            ))),
            _ => Err(PyErr::from(KriraError::UnsupportedFormat(format!(
                "Unsupported format: .{}. Supported: .csv, .xlsx",
                extension
            )))),
        }
    }

    /// Process a CSV file
    fn process_csv_file(&mut self, py: Python, file_path: &str) -> PyResult<PyObject> {
        let path = Path::new(file_path);
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);

        let mut csv_reader = csv::ReaderBuilder::new()
            .flexible(true)
            .has_headers(true)
            .from_reader(reader);

        // Get headers
        let headers: Vec<String> = csv_reader
            .headers()?
            .iter()
            .enumerate()
            .map(|(i, h)| {
                let h = h.trim();
                if h.is_empty() {
                    format!("col_{}", i + 1)
                } else {
                    h.to_string()
                }
            })
            .collect();

        // Build source info
        let source = path.file_name().and_then(|n| n.to_str()).unwrap_or("unknown");
        let source_path = path.to_string_lossy().to_string();
        let source_type = "csv";

        let mut all_chunks: Vec<PyObject> = Vec::new();
        let mut batch_texts: Vec<String> = Vec::new();
        let mut batch_row_ids: Vec<usize> = Vec::new();
        let mut chunk_index: usize = 0;

        // Process rows
        for (row_num, result) in csv_reader.records().enumerate() {
            let record = match result {
                Ok(r) => r,
                Err(_) => continue,
            };

            // Transform row to text
            let parts: Vec<String> = record
                .iter()
                .enumerate()
                .filter(|(_, v)| !v.trim().is_empty())
                .map(|(i, v)| {
                    let header = headers.get(i).cloned().unwrap_or_else(|| format!("col_{}", i + 1));
                    if self.config.transform_config.output_format == "markdown" {
                        format!("**{}**: {}", header, v.trim())
                    } else {
                        format!("{}: {}", header, v.trim())
                    }
                })
                .collect();

            if parts.is_empty() {
                continue;
            }

            let row_text = parts.join(" | ");

            // Clean the text
            let cleaned_text = self.cleaner.clean_text(Some(&row_text));
            if cleaned_text.is_empty() {
                continue;
            }

            batch_texts.push(cleaned_text);
            batch_row_ids.push(row_num + 1);
            self.rows_processed += 1;

            // Process batch when full
            if batch_texts.len() >= self.config.csv_batch_rows {
                let chunks = self.chunk_batch(
                    py,
                    &batch_texts,
                    &batch_row_ids,
                    source,
                    &source_path,
                    source_type,
                    chunk_index,
                )?;

                for chunk in chunks {
                    chunk_index += 1;
                    self.chunks_created += 1;
                    all_chunks.push(chunk);
                }

                batch_texts.clear();
                batch_row_ids.clear();
            }
        }

        // Process remaining batch
        if !batch_texts.is_empty() {
            let chunks = self.chunk_batch(
                py,
                &batch_texts,
                &batch_row_ids,
                source,
                &source_path,
                source_type,
                chunk_index,
            )?;

            for chunk in chunks {
                self.chunks_created += 1;
                all_chunks.push(chunk);
            }
        }

        // Update stats from cleaner
        let cleaner_stats = self.cleaner.get_stats();
        self.bytes_cleaned += cleaner_stats.get("bytes_cleaned").copied().unwrap_or(0);
        self.patterns_removed += cleaner_stats.get("patterns_removed").copied().unwrap_or(0);

        let list = PyList::new(py, &all_chunks);
        Ok(list.into())
    }

    /// Process raw text through the pipeline
    pub fn process_text(&mut self, py: Python, text: &str) -> PyResult<PyObject> {
        let cleaned = self.cleaner.clean_text(Some(text));

        if cleaned.is_empty() {
            let empty = PyList::empty(py);
            return Ok(empty.into());
        }

        let chunks = self.chunk_text(py, &cleaned, "text_input", "text_input", "text", 0)?;
        Ok(chunks)
    }

    /// Chunk text using the configured chunker
    pub fn chunk_text(
        &mut self,
        py: Python,
        text: &str,
        source: &str,
        source_path: &str,
        source_type: &str,
        start_chunk_index: usize,
    ) -> PyResult<PyObject> {
        let config = self.config.chunk_config.clone();
        let config_hash = config.config_hash();
        let max_size = config.get_max_size();
        let overlap_size = config.get_overlap_size();
        let min_chars = config.min_chars;

        let text = clean_text(text);
        if text.is_empty() {
            let empty = PyList::empty(py);
            return Ok(empty.into());
        }

        let mut chunks: Vec<PyObject> = Vec::new();
        let mut chunk_index = start_chunk_index;

        // Simple chunking if text fits
        if text.len() <= max_size {
            if text.len() >= min_chars {
                let chunk = self.create_chunk(
                    py,
                    &text,
                    source,
                    source_path,
                    source_type,
                    chunk_index,
                    &config_hash,
                    "natural",
                    None,
                    None,
                )?;
                chunks.push(chunk);
            }
            let list = PyList::new(py, &chunks);
            return Ok(list.into());
        }

        // Use appropriate chunking strategy
        let mut pos = 0;
        let step = max_size.saturating_sub(overlap_size);

        while pos < text.len() {
            let end = (pos + max_size).min(text.len());

            // Try to find a natural break point
            let mut split_pos = end;
            if end < text.len() {
                // Look for newline
                if let Some(newline) = text[pos..end].rfind('\n') {
                    split_pos = pos + newline + 1;
                } else if let Some(space) = text[pos..end].rfind(' ') {
                    split_pos = pos + space + 1;
                }
            }

            let chunk_text = text[pos..split_pos].trim();
            if chunk_text.len() >= min_chars {
                let chunk = self.create_chunk(
                    py,
                    chunk_text,
                    source,
                    source_path,
                    source_type,
                    chunk_index,
                    &config_hash,
                    "natural",
                    None,
                    None,
                )?;
                chunks.push(chunk);
                chunk_index += 1;
                self.chunks_created += 1;
            }

            pos = if split_pos > pos + step {
                split_pos
            } else {
                pos + step
            };
        }

        let list = PyList::new(py, &chunks);
        Ok(list.into())
    }

    /// Return pipeline statistics
    pub fn get_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("rows_processed".to_string(), self.rows_processed);
        stats.insert("chunks_created".to_string(), self.chunks_created);
        stats.insert("bytes_cleaned".to_string(), self.bytes_cleaned);
        stats.insert("patterns_removed".to_string(), self.patterns_removed);
        stats.insert("files_processed".to_string(), self.files_processed);
        stats
    }

    /// Reset all statistics counters
    pub fn reset_stats(&mut self) {
        self.rows_processed = 0;
        self.chunks_created = 0;
        self.bytes_cleaned = 0;
        self.patterns_removed = 0;
        self.files_processed = 0;
        self.cleaner.reset_stats();
        self.transformer.reset_stats();
    }

    fn __repr__(&self) -> String {
        format!(
            "KriraPipeline(rows_processed={}, chunks_created={}, files_processed={})",
            self.rows_processed, self.chunks_created, self.files_processed
        )
    }
}

impl KriraPipeline {
    /// Chunk a batch of texts
    fn chunk_batch(
        &mut self,
        py: Python,
        texts: &[String],
        row_ids: &[usize],
        source: &str,
        source_path: &str,
        source_type: &str,
        start_chunk_index: usize,
    ) -> PyResult<Vec<PyObject>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let combined = texts.join("\n");
        let config = &self.config.chunk_config;
        let config_hash = config.config_hash();
        let max_size = config.get_max_size();
        let overlap_size = config.get_overlap_size();
        let min_chars = config.min_chars;

        let row_start = row_ids.first().copied();
        let row_end = row_ids.last().copied();

        let mut chunks: Vec<PyObject> = Vec::new();
        let mut chunk_index = start_chunk_index;

        // Simple chunking
        let step = max_size.saturating_sub(overlap_size);
        let mut pos = 0;

        while pos < combined.len() {
            let end = (pos + max_size).min(combined.len());

            // Try to find a natural break point
            let mut split_pos = end;
            if end < combined.len() {
                if let Some(newline) = combined[pos..end].rfind('\n') {
                    split_pos = pos + newline + 1;
                }
            }

            let chunk_text = combined[pos..split_pos].trim();
            if chunk_text.len() >= min_chars {
                let chunk = self.create_chunk(
                    py,
                    chunk_text,
                    source,
                    source_path,
                    source_type,
                    chunk_index,
                    &config_hash,
                    "natural",
                    row_start,
                    row_end,
                )?;
                chunks.push(chunk);
                chunk_index += 1;
            }

            pos = if split_pos > pos + step {
                split_pos
            } else {
                pos + step
            };
        }

        Ok(chunks)
    }

    /// Create a chunk dictionary
    #[allow(clippy::too_many_arguments)]
    fn create_chunk(
        &self,
        py: Python,
        content: &str,
        source: &str,
        source_path: &str,
        source_type: &str,
        chunk_index: usize,
        config_hash: &str,
        boundary_type: &str,
        row_start: Option<usize>,
        row_end: Option<usize>,
    ) -> PyResult<PyObject> {
        let content = clean_text(content);
        let chunk_id = stable_id(source, source_path, chunk_index, &content);

        let metadata = PyDict::new(py);
        metadata.set_item("source", source)?;
        metadata.set_item("source_path", source_path)?;
        metadata.set_item("source_type", source_type)?;
        metadata.set_item("chunk_index", chunk_index)?;
        metadata.set_item("config_hash", config_hash)?;
        metadata.set_item("boundary_type", boundary_type)?;

        if let Some(start) = row_start {
            metadata.set_item("row_start", start)?;
        }
        if let Some(end) = row_end {
            metadata.set_item("row_end", end)?;
        }

        let chunk = PyDict::new(py);
        chunk.set_item("id", chunk_id)?;
        chunk.set_item("text", content)?;
        chunk.set_item("metadata", metadata)?;

        Ok(chunk.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert_eq!(config.csv_batch_rows, 50_000);
        assert_eq!(config.xlsx_batch_rows, 25_000);
    }
}
