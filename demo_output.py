"""Demo script to showcase Krira Augment output."""
from krira_augment import Pipeline

# Initialize
pipeline = Pipeline()

# Process test file
print("Processing test_data/test.csv...")
stats = pipeline.process("test_data/test.csv")

# Print the beautiful formatted output
print(stats)
