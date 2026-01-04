use pyo3::prelude::*;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum KriraError {
    #[error("IO Error: {0}")]
    IOError(#[from] std::io::Error),

    #[error("Configuration Error: {0}")]
    ConfigError(String),

    #[error("CSV Error: {0}")]
    CsvError(#[from] csv::Error),
    
    #[error("Processing Error: {0}")]
    ProcessError(String),
}

// Convert Rust errors to Python exceptions
impl From<KriraError> for PyErr {
    fn from(err: KriraError) -> PyErr {
        match err {
            KriraError::IOError(e) => pyo3::exceptions::PyIOError::new_err(e.to_string()),
            KriraError::ConfigError(e) => pyo3::exceptions::PyValueError::new_err(e),
            KriraError::CsvError(e) => pyo3::exceptions::PyValueError::new_err(e.to_string()),
            KriraError::ProcessError(e) => pyo3::exceptions::PyRuntimeError::new_err(e),
        }
    }
}
