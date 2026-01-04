//! Data Cleaning Module
//! 
//! High-performance text cleaning using pre-compiled regexes.

use lazy_static::lazy_static;
use regex::Regex;


lazy_static! {
    static ref HEADER_RE: Regex = Regex::new(r"(?i)Page \d+ of \d+").unwrap();
    static ref FOOTER_RE: Regex = Regex::new(r"(?i)© \d{4}").unwrap();
    static ref MULTI_WS_RE: Regex = Regex::new(r"[ \t]+").unwrap();
}



pub struct RustCleaner;

impl RustCleaner {
    /// Clean text using zero-copy (cow) operations where possible, 
    /// but Regex::replace returns Cow<str> so we often allocate new strings.
    pub fn clean(text: &str) -> String {
        // 1. Remove Headers
        let t1 = HEADER_RE.replace_all(text, "");
        
        // 2. Remove Footers
        let t2 = FOOTER_RE.replace_all(&t1, "");
        
        // 3. Normalize Whitespace (simple version)
        let t3 = MULTI_WS_RE.replace_all(&t2, " ");
        
        // 4. Unicode normalization (NFC)
        // Using standard iterator for efficiency if needed, but string allocation is dominating anyway.
        // For strict O(1) implies streaming, but regex replace on full string is O(N).
        
        t3.trim().to_string()
    }
}
