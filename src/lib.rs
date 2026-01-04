//! Core library for Krira Augment
//! 
<<<<<<< HEAD
//! Implements high-performance parallel file processing with streaming support.
=======
//! Implements high-performance parallel file processing.
>>>>>>> c85f1727f4e8aad48e031764a3067eb4374a5a9f

use std::fs::File;
use std::io::{BufWriter, Write};

<<<<<<< HEAD
use memmap2::MmapOptions;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::mpsc;
=======

use memmap2::MmapOptions;
use pyo3::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
>>>>>>> c85f1727f4e8aad48e031764a3067eb4374a5a9f

mod cleaning;
mod chunker;
mod errors;

use cleaning::RustCleaner;
use chunker::RustChunker;
use errors::KriraError;

// =============================================================================
<<<<<<< HEAD
// Structs
=======
// structs
>>>>>>> c85f1727f4e8aad48e031764a3067eb4374a5a9f
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

<<<<<<< HEAD
/// Internal chunk data structure for streaming
struct StreamChunk {
    text: String,
    char_count: usize,
}

=======
>>>>>>> c85f1727f4e8aad48e031764a3067eb4374a5a9f
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

<<<<<<< HEAD
// =============================================================================
// File-Based Processing (Original)
// =============================================================================

=======
>>>>>>> c85f1727f4e8aad48e031764a3067eb4374a5a9f
#[pyfunction]
fn process_file_rust(py: Python, input_path: String, output_path: String, config_json: String) -> PyResult<()> {
    
    // 1. Parse Config
    let config: PipelineConfig = serde_json::from_str(&config_json)
        .map_err(|e| KriraError::ConfigError(e.to_string()))?;
        
    let chunker = RustChunker::new(config.max_chars);

    // 2. Prepare Output Writer (Thread-safe via Bounded Channel)
    // Using 128 batches to keep memory low on 8GB RAM systems
<<<<<<< HEAD
    let (tx, rx) = mpsc::sync_channel::<Vec<ChunkObj>>(128); 
=======
    let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<ChunkObj>>(128); 
>>>>>>> c85f1727f4e8aad48e031764a3067eb4374a5a9f

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
<<<<<<< HEAD
// Streaming Mode: ChunkIterator
// =============================================================================

use std::sync::Mutex;

/// A Python iterator that yields chunks one by one without writing to disk.
/// 
/// This provides O(1) memory usage regardless of input file size by using
/// a bounded channel with backpressure.
#[pyclass]
pub struct ChunkIterator {
    /// Receiver for chunks from the background thread (wrapped in Mutex for thread safety)
    receiver: Mutex<Option<mpsc::Receiver<StreamChunk>>>,
    /// Source file path for metadata
    source_path: String,
    /// Current chunk index
    chunk_index: Mutex<usize>,
    /// Flag indicating if iteration is complete
    finished: Mutex<bool>,
}

#[pymethods]
impl ChunkIterator {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(slf: PyRef<Self>) -> Option<PyObject> {
        // Check if finished
        {
            let finished = slf.finished.lock().unwrap();
            if *finished {
                return None;
            }
        }

        // Get the receiver and try to receive
        let chunk_result = {
            let receiver_guard = slf.receiver.lock().unwrap();
            if let Some(ref receiver) = *receiver_guard {
                receiver.recv().ok()
            } else {
                None
            }
        };

        match chunk_result {
            Some(chunk) => {
                let current_index = {
                    let mut index = slf.chunk_index.lock().unwrap();
                    let idx = *index;
                    *index += 1;
                    idx
                };

                Python::with_gil(|py| {
                    // Create metadata dict
                    let metadata = PyDict::new(py);
                    let _ = metadata.set_item("source", &slf.source_path);
                    let _ = metadata.set_item("chunk_index", current_index);
                    let _ = metadata.set_item("char_count", chunk.char_count);

                    // Create main chunk dict
                    let chunk_dict = PyDict::new(py);
                    let _ = chunk_dict.set_item("text", chunk.text);
                    let _ = chunk_dict.set_item("metadata", metadata);

                    Some(chunk_dict.unbind().into())
                })
            }
            None => {
                let mut finished = slf.finished.lock().unwrap();
                *finished = true;
                None
            }
        }
    }
}

/// Start streaming chunks from a file.
/// 
/// This function spawns a background thread that processes the file and sends
/// chunks through a bounded channel. The Python side can iterate over the
/// ChunkIterator to receive chunks one by one.
/// 
/// # Arguments
/// * `input_path` - Path to the input file
/// * `max_chars` - Maximum characters per chunk
/// * `overlap_chars` - Character overlap between chunks (reserved for future use)
/// 
/// # Returns
/// A ChunkIterator that yields chunks as Python dictionaries with 'text' and 'metadata' keys.
#[pyfunction]
fn process_stream(
    py: Python,
    input_path: String,
    max_chars: usize,
    _overlap_chars: usize,  // Reserved for future overlap implementation
) -> PyResult<Py<ChunkIterator>> {
    use std::path::Path;
    
    // Validate file exists
    if !Path::new(&input_path).exists() {
        return Err(PyErr::from(KriraError::FileNotFound(input_path)));
    }

    // Create bounded channel with backpressure (100 chunks max in buffer)
    // This limits memory usage to approximately 100 * chunk_size bytes
    let (sender, receiver) = mpsc::sync_channel::<StreamChunk>(100);
    let input_path_clone = input_path.clone();

    // Spawn background processing thread
    std::thread::spawn(move || {
        let result: Result<(), KriraError> = (|| {
            let chunker = RustChunker::new(max_chars);
            
            // Open and mmap file for zero-copy access
            let file = File::open(&input_path_clone).map_err(KriraError::IOError)?;
            let mmap = unsafe { MmapOptions::new().map(&file).map_err(KriraError::IOError)? };
            
            // Convert to UTF-8
            let content = std::str::from_utf8(&mmap[..])
                .map_err(|e| KriraError::ConfigError(format!("File is not valid UTF-8: {}", e)))?;

            // Split into 32MB segments for efficient processing
            let segments = split_into_chunks(content, 32 * 1024 * 1024);

            // Process segments sequentially to maintain order
            // (Parallelism is used within the file-based mode, 
            // but streaming requires ordered output)
            for segment in segments {
                for line in segment.lines() {
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }

                    // Clean the line
                    let cleaned = RustCleaner::clean(line);
                    if cleaned.is_empty() {
                        continue;
                    }

                    // Chunk the cleaned text
                    let sub_chunks = chunker.chunk(&cleaned);

                    // Send each chunk through the channel
                    for chunk_text in sub_chunks {
                        let stream_chunk = StreamChunk {
                            char_count: chunk_text.len(),
                            text: chunk_text,
                        };

                        // If receiver is dropped (iteration stopped), stop processing
                        if sender.send(stream_chunk).is_err() {
                            return Ok(());
                        }
                    }
                }
            }

            Ok(())
        })();

        if let Err(e) = result {
            eprintln!("Streaming error: {}", e);
        }
    });

    // Create and return the iterator
    Py::new(py, ChunkIterator {
        receiver: Mutex::new(Some(receiver)),
        source_path: input_path,
        chunk_index: Mutex::new(0),
        finished: Mutex::new(false),
    })
}

// =============================================================================
=======
>>>>>>> c85f1727f4e8aad48e031764a3067eb4374a5a9f
// Python Module
// =============================================================================

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_file_rust, m)?)?;
<<<<<<< HEAD
    m.add_function(wrap_pyfunction!(process_stream, m)?)?;
    m.add_class::<ChunkIterator>()?;
=======
>>>>>>> c85f1727f4e8aad48e031764a3067eb4374a5a9f
    Ok(())
}
