//! Configuration for Krira Chunker
//!
//! Contains ChunkConfig with all chunking parameters.

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::errors::KriraError;

/// Chunking strategy enum
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChunkStrategy {
    Fixed,
    Sentence,
    Markdown,
    Hybrid,
    LLM, // Added LLM strategy
}

impl Default for ChunkStrategy {
    fn default() -> Self {
        ChunkStrategy::Hybrid
    }
}

impl std::fmt::Display for ChunkStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChunkStrategy::Fixed => write!(f, "fixed"),
            ChunkStrategy::Sentence => write!(f, "sentence"),
            ChunkStrategy::Markdown => write!(f, "markdown"),
            ChunkStrategy::Hybrid => write!(f, "hybrid"),
            ChunkStrategy::LLM => write!(f, "llm"),
        }
    }
}

/// Configuration for the chunking process.
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChunkConfig {
    // Chunk sizing
    #[pyo3(get, set)]
    pub max_chars: usize,

    #[pyo3(get, set)]
    pub overlap_chars: usize,

    // Token-based control
    #[pyo3(get, set)]
    pub use_tokens: bool,

    #[pyo3(get, set)]
    pub max_tokens: usize,

    #[pyo3(get, set)]
    pub overlap_tokens: usize,

    // Filters
    #[pyo3(get, set)]
    pub min_chars: usize,

    // Strategy
    chunk_strategy_name: String,

    // Preservation flags
    #[pyo3(get, set)]
    pub preserve_code_blocks: bool,

    #[pyo3(get, set)]
    pub preserve_tables: bool,

    #[pyo3(get, set)]
    pub preserve_lists: bool,

    // Tabular data
    #[pyo3(get, set)]
    pub rows_per_chunk: Option<usize>,

    // Streaming/batching
    #[pyo3(get, set)]
    pub sink_batch_size: usize,

    #[pyo3(get, set)]
    pub csv_batch_rows: usize,

    #[pyo3(get, set)]
    pub xlsx_batch_rows: usize,

    // HTTP settings
    #[pyo3(get, set)]
    pub http_timeout_s: usize,

    #[pyo3(get, set)]
    pub url_max_bytes: usize,

    #[pyo3(get, set)]
    pub url_allow_private: bool,

    // Security
    #[pyo3(get, set)]
    pub security_max_file_bytes: usize,
}

#[pymethods]
impl ChunkConfig {
    #[new]
    #[pyo3(signature = (
        max_chars = 2200,
        overlap_chars = 250,
        use_tokens = false,
        max_tokens = 512,
        overlap_tokens = 64,
        min_chars = 30,
        chunk_strategy = "hybrid",
        preserve_code_blocks = true,
        preserve_tables = true,
        preserve_lists = true,
        rows_per_chunk = None,
        sink_batch_size = 256,
        csv_batch_rows = 50000,
        xlsx_batch_rows = 25000,
        http_timeout_s = 15,
        url_max_bytes = 8388608,
        url_allow_private = false,
        security_max_file_bytes = 50000000
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        max_chars: usize,
        overlap_chars: usize,
        use_tokens: bool,
        max_tokens: usize,
        overlap_tokens: usize,
        min_chars: usize,
        chunk_strategy: &str,
        preserve_code_blocks: bool,
        preserve_tables: bool,
        preserve_lists: bool,
        rows_per_chunk: Option<usize>,
        sink_batch_size: usize,
        csv_batch_rows: usize,
        xlsx_batch_rows: usize,
        http_timeout_s: usize,
        url_max_bytes: usize,
        url_allow_private: bool,
        security_max_file_bytes: usize,
    ) -> PyResult<Self> {
        // Validate
        if overlap_chars >= max_chars {
            return Err(PyErr::from(KriraError::ConfigError(format!(
                "overlap_chars ({}) must be less than max_chars ({})",
                overlap_chars, max_chars
            ))));
        }

        if use_tokens && overlap_tokens >= max_tokens {
            return Err(PyErr::from(KriraError::ConfigError(format!(
                "overlap_tokens ({}) must be less than max_tokens ({})",
                overlap_tokens, max_tokens
            ))));
        }

        if max_chars == 0 {
            return Err(PyErr::from(KriraError::ConfigError(
                "max_chars must be positive".to_string(),
            )));
        }

        let valid_strategies = ["fixed", "sentence", "markdown", "hybrid", "llm"];
        if !valid_strategies.contains(&chunk_strategy) {
            return Err(PyErr::from(KriraError::ConfigError(format!(
                "Invalid chunk_strategy: {}. Must be one of: {:?}",
                chunk_strategy, valid_strategies
            ))));
        }

        Ok(Self {
            max_chars,
            overlap_chars,
            use_tokens,
            max_tokens,
            overlap_tokens,
            min_chars,
            chunk_strategy_name: chunk_strategy.to_string(),
            preserve_code_blocks,
            preserve_tables,
            preserve_lists,
            rows_per_chunk,
            sink_batch_size,
            csv_batch_rows,
            xlsx_batch_rows,
            http_timeout_s,
            url_max_bytes,
            url_allow_private,
            security_max_file_bytes,
        })
    }

    #[getter]
    pub fn chunk_strategy(&self) -> String {
        self.chunk_strategy_name.clone()
    }

    #[setter]
    pub fn set_chunk_strategy(&mut self, strategy: &str) -> PyResult<()> {
        let valid = ["fixed", "sentence", "markdown", "hybrid", "llm"];
        if !valid.contains(&strategy) {
            return Err(PyErr::from(KriraError::ConfigError(format!(
                "Invalid strategy: {}",
                strategy
            ))));
        }
        self.chunk_strategy_name = strategy.to_string();
        Ok(())
    }

    pub fn config_hash(&self) -> String {
        let mut hasher = DefaultHasher::new();
        self.max_chars.hash(&mut hasher);
        self.overlap_chars.hash(&mut hasher);
        self.use_tokens.hash(&mut hasher);
        self.max_tokens.hash(&mut hasher);
        self.overlap_tokens.hash(&mut hasher);
        self.min_chars.hash(&mut hasher);
        self.chunk_strategy_name.hash(&mut hasher);
        self.preserve_code_blocks.hash(&mut hasher);
        self.preserve_tables.hash(&mut hasher);
        self.preserve_lists.hash(&mut hasher);
        format!("{:x}", hasher.finish())[..12].to_string()
    }

    pub fn get_max_size(&self) -> usize {
        if self.use_tokens { self.max_tokens } else { self.max_chars }
    }

    pub fn get_overlap_size(&self) -> usize {
        if self.use_tokens { self.overlap_tokens } else { self.overlap_chars }
    }

    fn __repr__(&self) -> String {
        format!(
            "ChunkConfig(max_chars={}, overlap_chars={}, chunk_strategy='{}')",
            self.max_chars, self.overlap_chars, self.chunk_strategy_name
        )
    }
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            max_chars: 2200,
            overlap_chars: 250,
            use_tokens: false,
            max_tokens: 512,
            overlap_tokens: 64,
            min_chars: 30,
            chunk_strategy_name: "hybrid".to_string(),
            preserve_code_blocks: true,
            preserve_tables: true,
            preserve_lists: true,
            rows_per_chunk: None,
            sink_batch_size: 256,
            csv_batch_rows: 50_000,
            xlsx_batch_rows: 25_000,
            http_timeout_s: 15,
            url_max_bytes: 8 * 1024 * 1024,
            url_allow_private: false,
            security_max_file_bytes: 50_000_000,
        }
    }
}

impl ChunkConfig {
    pub fn strategy(&self) -> ChunkStrategy {
        match self.chunk_strategy_name.as_str() {
            "fixed" => ChunkStrategy::Fixed,
            "sentence" => ChunkStrategy::Sentence,
            "markdown" => ChunkStrategy::Markdown,
            "llm" => ChunkStrategy::LLM,
            _ => ChunkStrategy::Hybrid,
        }
    }
}
