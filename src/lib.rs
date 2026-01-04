//! Core library for Krira Augment
//! 
//! Implements high-performance parallel file processing.

use std::fs::File;
use std::io::{BufWriter, Write};


use memmap2::MmapOptions;
use pyo3::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

mod cleaning;
mod chunker;
mod errors;

use cleaning::RustCleaner;
use chunker::RustChunker;
use errors::KriraError;

// =============================================================================
// structs
// =============================================================================

#[derive(Serialize, Deserialize)]
struct PipelineConfig {
    max_chars: usize,
}

#[derive(Serialize)]
struct ChunkObj {
    text: String,
    length: usize,
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Split a large string slice into valid chunks (approx target_chunk_size) aligned to newlines.
fn split_into_chunks(text: &str, target_chunk_size: usize) -> Vec<&str> {
    let mut chunks = Vec::with_capacity((text.len() / target_chunk_size) + 1);
    let mut start = 0;
    
    while start < text.len() {
        let mut end = start + target_chunk_size;
        
        if end >= text.len() {
            end = text.len();
        } else {
            // Find next newline from 'end' to avoid cutting lines
            if let Some(pos) = text[end..].find('\n') {
                end = end + pos + 1; 
            } else {
                end = text.len();
            }
        }
        
        chunks.push(&text[start..end]);
        start = end;
    }
    chunks
}

#[pyfunction]
fn process_file_rust(py: Python, input_path: String, output_path: String, config_json: String) -> PyResult<()> {
    
    // 1. Parse Config
    let config: PipelineConfig = serde_json::from_str(&config_json)
        .map_err(|e| KriraError::ConfigError(e.to_string()))?;
        
    let chunker = RustChunker::new(config.max_chars);

    // 2. Prepare Output Writer (Thread-safe via Bounded Channel)
    // Using 128 batches to keep memory low on 8GB RAM systems
    let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<ChunkObj>>(128); 

    // 3. Release GIL for heavy lifting
    py.allow_threads(move || -> Result<(), KriraError> {
        // Spawn Writer Thread
        let writer_handle = std::thread::spawn(move || -> Result<(), KriraError> {
            let output_file = File::create(&output_path).map_err(KriraError::IOError)?;
            let mut writer = BufWriter::with_capacity(64 * 1024, output_file);
            
            while let Ok(batch) = rx.recv() {
                for res in batch {
                    if let Ok(json_line) = serde_json::to_string(&res) {
                        writeln!(writer, "{}", json_line).map_err(KriraError::IOError)?;
                    }
                }
            }
            writer.flush().map_err(KriraError::IOError)?;
            Ok(())
        });

        // Open Input File
        let file = File::open(&input_path).map_err(KriraError::IOError)?;
        
        // Mmap
        let mmap = unsafe { MmapOptions::new().map(&file).map_err(KriraError::IOError)? };
        
        // Convert to str (assumes UTF-8)
        let content = std::str::from_utf8(&mmap[..])
            .map_err(|e| KriraError::ConfigError(format!("File is not valid UTF-8: {}", e)))?;

        // 4. Split into manageable chunks (32 MB) to reduce Rayon task overhead
        let chunks = split_into_chunks(content, 32 * 1024 * 1024);

        // 5. Parallel Processing
        chunks.par_iter().for_each_with(tx, |sender, chunk| {
            let mut batch = Vec::with_capacity(100);
            
            for line in chunk.lines() {
                if line.trim().is_empty() { continue; }

                // Clean
                let cleaned = RustCleaner::clean(line);
                if cleaned.is_empty() { continue; }

                // Chunk
                let sub_chunks = chunker.chunk(&cleaned);

                for c in sub_chunks {
                     batch.push(ChunkObj {
                        length: c.len(),
                        text: c,
                    });
                }

                // Batch flushing to channel (100 items = ~50KB per batch)
                if batch.len() >= 100 {
                    let _ = sender.send(batch);
                    batch = Vec::with_capacity(100);
                }
            }
            
            if !batch.is_empty() {
                let _ = sender.send(batch);
            }
        });

        // Wait for writer to finish
        match writer_handle.join() {
            Ok(result) => result,
            Err(e) => Err(KriraError::ConfigError(format!("Writer thread panicked: {:?}", e))),
        }
    }).map_err(PyErr::from)
}

// =============================================================================
// Python Module
// =============================================================================

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_file_rust, m)?)?;
    Ok(())
}
