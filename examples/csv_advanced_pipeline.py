#!/usr/bin/env python3
"""
Advanced CSV Pipeline Example for Krira Augment.

This example demonstrates the full Clean -> Transform -> Chunk pipeline
for processing CSV files with data cleaning, transformation, and chunking.

Features demonstrated:
- Header/footer removal
- Unicode normalization
- PII redaction (optional)
- Markdown transformation
- Streaming chunking
- Pipeline statistics
"""

import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krira_augment import (
    KriraPipeline,
    PipelineConfig,
    CleaningConfig,
    TransformConfig,
)

# Try to import ChunkConfig from Krira_Chunker
try:
    from Krira_Chunker import ChunkConfig
except ImportError:
    ChunkConfig = None
    print("Note: Krira_Chunker not available, using fallback chunking")


def main():
    """Run the advanced CSV pipeline example."""
    
    print("=" * 60)
    print("Krira Augment V2.0 - Advanced CSV Pipeline Example")
    print("=" * 60)
    print()
    
    # ==========================================================================
    # Step 1: Configure the Pipeline
    # ==========================================================================
    
    print("Step 1: Configuring pipeline...")
    
    # Cleaning configuration
    cleaning_config = CleaningConfig(
        # Noise removal
        remove_headers=True,       # Remove "Page X of Y" patterns
        remove_footers=True,       # Remove copyright notices
        custom_patterns=[          # Custom patterns to remove
            r"INTERNAL USE ONLY",
            r"DRAFT",
        ],
        
        # Normalization
        fix_unicode=True,          # Fix broken Unicode characters
        normalize_whitespace=True, # Collapse multiple spaces
        preserve_line_breaks=True, # Keep paragraph structure
        
        # Privacy (disabled for this example - enable for production)
        redact_pii=False,          # Set to True to mask emails/phones
        
        # Performance
        chunk_buffer_size=10_000,  # Buffer size for streaming
    )
    
    # Transformation configuration
    transform_config = TransformConfig(
        output_format="markdown",  # Output as markdown
        preserve_tables=True,      # Keep table structure
        max_table_columns=10,      # Limit columns for very wide tables
        json_indent=True,          # Indent JSON content
        max_json_depth=3,          # Limit JSON nesting
    )
    
    # Optional: Chunk configuration (if Krira_Chunker is available)
    chunk_config = None
    if ChunkConfig:
        chunk_config = ChunkConfig(
            max_chars=2000,        # Maximum characters per chunk
            overlap_chars=200,     # Overlap between chunks
            use_tokens=False,      # Use character count (faster)
            chunk_strategy="hybrid",  # Boundary-aware chunking
        )
    
    # Master pipeline configuration
    pipeline_config = PipelineConfig(
        cleaning_config=cleaning_config,
        transform_config=transform_config,
        chunk_config=chunk_config,
        csv_batch_rows=50_000,     # Process 50k rows at a time
        log_progress_every=10_000, # Log progress every 10k rows
    )
    
    print("  ✓ Configuration complete")
    print()
    
    # ==========================================================================
    # Step 2: Initialize Pipeline
    # ==========================================================================
    
    print("Step 2: Initializing pipeline...")
    
    pipeline = KriraPipeline(pipeline_config)
    
    print("  ✓ Pipeline initialized")
    print()
    
    # ==========================================================================
    # Step 3: Create Sample Data (for demonstration)
    # ==========================================================================
    
    print("Step 3: Creating sample data...")
    
    # Create a sample CSV file for demonstration
    sample_csv_content = """ID,Name,Email,Phone,Department,Notes
1,Alice Smith,alice.smith@company.com,555-123-4567,Engineering,"Page 1 of 10 - Project Lead"
2,Bob Johnson,bob.j@company.com,555-234-5678,Marketing,"Handles social media campaigns"
3,Charlie Brown,charlie@example.org,555-345-6789,Sales,"Top performer Q3 2024"
4,Diana Ross,diana.r@company.com,555-456-7890,HR,"Employee onboarding specialist"
5,Edward Chen,e.chen@company.com,555-567-8901,Engineering,"Backend developer - Python & Go"
6,Fiona Williams,fiona@company.com,555-678-9012,Finance,"Budget analysis and reporting"
7,George Miller,g.miller@example.com,555-789-0123,Engineering,"Senior DevOps engineer"
8,Helen Taylor,helen.t@company.com,555-890-1234,Marketing,"Content strategy & SEO"
9,Ivan Petrov,ivan@company.com,555-901-2345,Sales,"Enterprise accounts"
10,Julia Roberts,j.roberts@company.com,555-012-3456,Legal,"Contract negotiations"
Confidential - © 2024 Acme Corporation
All Rights Reserved - INTERNAL USE ONLY"""
    
    # Save to temporary file
    sample_file = "sample_financial_data.csv"
    with open(sample_file, "w", encoding="utf-8") as f:
        f.write(sample_csv_content)
    
    print(f"  ✓ Created sample file: {sample_file}")
    print()
    
    # ==========================================================================
    # Step 4: Process File Through Pipeline
    # ==========================================================================
    
    print("Step 4: Processing file through pipeline...")
    print("-" * 60)
    
    start_time = time.time()
    chunk_count = 0
    
    try:
        for chunk in pipeline.process_file(sample_file):
            chunk_count += 1
            
            # Display first 3 chunks
            if chunk_count <= 3:
                print(f"\n--- Chunk {chunk_count} ---")
                text_preview = chunk["text"][:300]
                if len(chunk["text"]) > 300:
                    text_preview += "..."
                print(f"Text:\n{text_preview}")
                print(f"\nMetadata: {chunk['metadata']}")
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return 1
    
    elapsed = time.time() - start_time
    
    print()
    print("-" * 60)
    
    # ==========================================================================
    # Step 5: Display Statistics
    # ==========================================================================
    
    print()
    print("Step 5: Pipeline Statistics")
    print("=" * 60)
    
    stats = pipeline.get_stats()
    
    print(f"  Files Processed:    {stats['files_processed']:,}")
    print(f"  Rows Processed:     {stats['rows_processed']:,}")
    print(f"  Chunks Created:     {stats['chunks_created']:,}")
    print(f"  Bytes Cleaned:      {stats['bytes_cleaned']:,}")
    print(f"  Patterns Removed:   {stats['patterns_removed']:,}")
    print()
    print(f"  Processing Time:    {elapsed:.2f}s")
    
    if stats['rows_processed'] > 0:
        rows_per_sec = stats['rows_processed'] / elapsed
        print(f"  Speed:              {rows_per_sec:,.0f} rows/sec")
    
    print()
    print("=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    
    # Cleanup
    if os.path.exists(sample_file):
        os.remove(sample_file)
        print(f"\n  ✓ Cleaned up temporary file: {sample_file}")
    
    return 0


def example_with_pii_redaction():
    """Example demonstrating PII redaction."""
    
    print("\n" + "=" * 60)
    print("PII Redaction Example")
    print("=" * 60)
    
    config = PipelineConfig(
        cleaning_config=CleaningConfig(
            redact_pii=True,  # Enable PII redaction
        )
    )
    
    pipeline = KriraPipeline(config)
    
    # Process text with PII
    text = """
    Contact Information:
    - Email: john.doe@example.com
    - Phone: 555-123-4567
    - Alternate: +1-800-555-0199
    
    Please reach out for more details.
    """
    
    print("\nOriginal Text:")
    print(text)
    
    print("\nAfter PII Redaction:")
    for chunk in pipeline.process_text(text):
        print(chunk["text"])


def example_custom_patterns():
    """Example demonstrating custom pattern removal."""
    
    print("\n" + "=" * 60)
    print("Custom Pattern Removal Example")
    print("=" * 60)
    
    config = PipelineConfig(
        cleaning_config=CleaningConfig(
            custom_patterns=[
                r"CLASSIFIED",
                r"TOP SECRET",
                r"REF-\d{6}",  # Reference numbers like REF-123456
            ]
        )
    )
    
    pipeline = KriraPipeline(config)
    
    text = """
    Document: REF-987654
    Status: CLASSIFIED
    
    This is the actual content of the document.
    
    Classification: TOP SECRET
    """
    
    print("\nOriginal Text:")
    print(text)
    
    print("\nAfter Custom Pattern Removal:")
    for chunk in pipeline.process_text(text):
        print(chunk["text"])


if __name__ == "__main__":
    exit_code = main()
    
    # Run additional examples
    example_with_pii_redaction()
    example_custom_patterns()
    
    sys.exit(exit_code)
