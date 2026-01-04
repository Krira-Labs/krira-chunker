//! Core Chunking Logic

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RustChunker {
    pub max_chars: usize,
}

impl RustChunker {
    pub fn new(max_chars: usize) -> Self {
        Self { max_chars }
    }

    /// Hybrid chunking: Paragraphs -> Sentences -> Accumulate
    pub fn chunk(&self, text: &str) -> Vec<String> {
        let mut chunks = Vec::new();
        let mut buffer = String::with_capacity(self.max_chars);

        // Split by paragraphs
        let paragraphs: Vec<&str> = text.split("\n\n").collect();

        for para in paragraphs {
            let para_len = para.len();

            // If paragraph fits in current buffer
            if buffer.len() + para_len + 2 <= self.max_chars {
                if !buffer.is_empty() {
                    buffer.push_str("\n\n");
                }
                buffer.push_str(para);
            } else {
                // If buffer has content, flush it
                if !buffer.is_empty() {
                    chunks.push(buffer.clone());
                    buffer.clear();
                }

                // If paragraph itself is too large, split by sentences
                if para_len > self.max_chars {
                    // Simple sentence split by period (can use regex if strict)
                    // For speed, strict char split is faster
                    let sentences: Vec<&str> = para.split(". ").collect();
                    
                    for sent in sentences {
                        let sent_len = sent.len();
                         if buffer.len() + sent_len + 2 <= self.max_chars {
                             if !buffer.is_empty() { buffer.push_str(". "); }
                             buffer.push_str(sent);
                         } else {
                             if !buffer.is_empty() {
                                 chunks.push(buffer.clone());
                                 buffer.clear();
                             }
                             // If sentence still too big (unlikely but possible)
                             if sent_len > self.max_chars {
                                // Hard split
                                let mut start = 0;
                                while start < sent_len {
                                    let end = (start + self.max_chars).min(sent_len);
                                    chunks.push(sent[start..end].to_string());
                                    start += self.max_chars;
                                }
                             } else {
                                 buffer.push_str(sent);
                             }
                         }
                    }
                } else {
                    buffer.push_str(para);
                }
            }
        }

        if !buffer.is_empty() {
            chunks.push(buffer);
        }

        chunks
    }
}
