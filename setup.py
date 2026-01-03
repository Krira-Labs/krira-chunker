"""
Setup configuration for Krira_Chunker.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8", errors="ignore") as f:
    try:
        long_description = f.read()
    except FileNotFoundError:
        long_description = "Production-grade document chunking library for RAG applications."

setup(
    name="krira-chunker",
    version="0.2.11",
    author="Krira Labs",
    author_email="kriralabs@gmail.com",
    description="Production-grade document chunking library for RAG applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kriralabs/krira-chunker",
    license="MIT",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Text Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    
    # No required dependencies for core import
    install_requires=[],
    
    # Optional dependencies via extras
    extras_require={
        # Individual format support
        "pdf": ["pypdf>=4.0.0"],
        "url": [
            "requests>=2.28.0",
            "beautifulsoup4>=4.12.0",
            "trafilatura>=1.6.0",
        ],
        "csv": ["polars>=0.20.0"],
        "xlsx": ["openpyxl>=3.1.0"],
        "json": ["ijson>=3.2.0"],
        "tokens": ["tiktoken>=0.5.0"],
        
        # Testing
        "test": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "hypothesis>=6.0.0",
        ],
        
        # Benchmarking
        "bench": [
            "psutil>=5.9.0",
            "rich>=13.0.0",
        ],
        
        # All dependencies
        "all": [
            "pypdf>=4.0.0",
            "requests>=2.28.0",
            "beautifulsoup4>=4.12.0",
            "trafilatura>=1.6.0",
            "polars>=0.20.0",
            "openpyxl>=3.1.0",
            "ijson>=3.2.0",
            "tiktoken>=0.5.0",
            "psutil>=5.9.0",
            "rich>=13.0.0",
            "pytest>=7.0.0",
        ],
    },
    
    # Entry points
    entry_points={
        "console_scripts": [
            "krira-bench=Krira_Chunker.bench:main",
        ],
    },
)
