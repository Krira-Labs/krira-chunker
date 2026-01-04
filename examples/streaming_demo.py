"""
Streaming Mode Example: OpenAI + Pinecone

This example demonstrates how to use Krira Augment's streaming mode
to process chunks in real-time without creating intermediate files.
"""

from krira_augment.krira_chunker import Pipeline, PipelineConfig

# Uncomment these imports when you have the dependencies:
# import openai
# from pinecone import Pinecone


def main():
    """
    Streaming pipeline example.
    
    This processes chunks one-by-one without writing to disk,
    providing maximum efficiency for real-time RAG pipelines.
    """
    
    # === Configuration ===
    # Replace with your actual API keys
    OPENAI_API_KEY = "sk-..."        # https://platform.openai.com/api-keys
    PINECONE_API_KEY = "pcone-..."   # https://app.pinecone.io/
    PINECONE_INDEX_NAME = "my-rag"
    
    # Create a sample CSV file for testing
    import tempfile
    import os
    
    sample_data = """id,text,category
1,The quick brown fox jumps over the lazy dog.,animals
2,Machine learning is a subset of artificial intelligence.,tech
3,Paris is the capital of France and known for the Eiffel Tower.,geography
4,Python is a popular programming language for data science.,tech
5,The Great Wall of China is one of the world's wonders.,history
"""
    
    # Create temp file
    fd, temp_path = tempfile.mkstemp(suffix=".csv", prefix="krira_demo_")
    os.close(fd)
    with open(temp_path, 'w', encoding='utf-8') as f:
        f.write(sample_data)
    
    print(f"📁 Created sample file: {temp_path}")
    
    try:
        # === Initialize Pipeline ===
        config = PipelineConfig(
            chunk_size=512,     # Max characters per chunk
            chunk_overlap=50,   # Overlap between chunks
        )
        pipeline = Pipeline(config=config)
        
        # === Stream Mode Demo ===
        print("\n🚀 Starting Streaming Mode...")
        print("=" * 60)
        
        chunk_count = 0
        for chunk in pipeline.process_stream(temp_path):
            chunk_count += 1
            
            # In production, you would:
            # 1. Create embeddings
            # 2. Store in vector database
            #
            # Example (uncomment when you have OpenAI/Pinecone):
            # 
            # client = openai.OpenAI(...)
            # 
            # response = client.embeddings.create(
            #     input=chunk["text"],
            #     model="text-embedding-3-small"
            # )
            # embedding = response.data[0].embedding
            # pc = Pinecone(api_key=PINECONE_API_KEY)
            # index = pc.Index(PINECONE_INDEX_NAME)
            # index.upsert(vectors=[(
            #     f"chunk_{chunk_count}",
            #     embedding,
            #     chunk["metadata"]
            # )])
            
            # For demo, just print the chunks
            text_preview = chunk["text"][:100] + "..." if len(chunk["text"]) > 100 else chunk["text"]
            print(f"\n[Chunk {chunk_count}]")
            print(f"  Text: {text_preview}")
            print(f"  Metadata: {chunk['metadata']}")
        
        print("\n" + "=" * 60)
        print(f"✅ Streaming Complete!")
        print(f"   Total chunks processed: {chunk_count}")
        print(f"   No intermediate file created!")
        
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
            print(f"\n🧹 Cleaned up temp file")


def demo_with_local_embedding():
    """
    Demo using local SentenceTransformers + ChromaDB (FREE, no API keys).
    """
    try:
        from sentence_transformers import SentenceTransformer
        import chromadb
    except ImportError:
        print("❌ This demo requires: pip install sentence-transformers chromadb")
        return
    
    # Create sample file
    import tempfile
    import os
    
    sample_text = """The quick brown fox jumps over the lazy dog.
Machine learning is transforming how we process data.
Vector databases enable semantic search capabilities.
Krira Augment provides high-performance text chunking.
RAG pipelines combine retrieval with generation."""
    
    fd, temp_path = tempfile.mkstemp(suffix=".txt", prefix="krira_local_demo_")
    os.close(fd)
    with open(temp_path, 'w', encoding='utf-8') as f:
        f.write(sample_text)
    
    print(f"📁 Created sample file: {temp_path}")
    
    try:
        # Initialize embedding model and vector store
        print("\n🔧 Loading SentenceTransformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        client = chromadb.Client()
        collection = client.get_or_create_collection("demo_chunks")
        
        # Initialize Krira pipeline
        config = PipelineConfig(chunk_size=200, chunk_overlap=20)
        pipeline = Pipeline(config=config)
        
        print("\n🚀 Streaming and embedding locally...")
        print("=" * 60)
        
        chunk_count = 0
        for chunk in pipeline.process_stream(temp_path):
            chunk_count += 1
            
            # Embed locally
            embedding = model.encode(chunk["text"])
            
            # Store in ChromaDB
            # Store in ChromaDB
            meta = chunk.get("metadata")
            collection.add(
                ids=[f"chunk_{chunk_count}"],
                embeddings=[embedding.tolist()],
                metadatas=[meta] if meta else None,
                documents=[chunk["text"]]
            )
            
            print(f"  ✓ Chunk {chunk_count}: {chunk['text'][:50]}...")
        
        print("\n" + "=" * 60)
        print(f"✅ Complete! {chunk_count} chunks embedded and stored.")
        
        # Demo: Query the collection
        print("\n🔍 Demo Query: 'data processing'")
        results = collection.query(
            query_embeddings=[model.encode("data processing").tolist()],
            n_results=2
        )
        print(f"   Top results: {results['documents']}")
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Krira Augment - Streaming Mode Demo")
    print("=" * 60)
    
    # Basic demo (no external dependencies)
    main()
    
    # Uncomment to run local embedding demo:
    # demo_with_local_embedding()
