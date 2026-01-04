"""
Streaming Integration Examples for Krira Augment

This script contains all streaming integration examples.
Each function demonstrates how to use Krira Augment's streaming mode
with a different embedding provider and vector store combination.

Streaming mode processes chunks in real-time without creating intermediate files,
providing maximum efficiency for production RAG pipelines.

Run with: python examples/streaming_integrations.py
"""

from krira_augment.krira_chunker import Pipeline, PipelineConfig


def openai_pinecone_streaming():
    """
    OpenAI + Pinecone Integration (Streaming Mode)
    
    Prerequisites:
        pip install openai pinecone-client
    """
    # API Keys
    OPENAI_API_KEY = "sk-..."        # https://platform.openai.com/api-keys  
    PINECONE_API_KEY = "pcone-..."   # https://app.pinecone.io/
    PINECONE_INDEX_NAME = "my-rag"
    
    # Initialize (uncomment when ready)
    # import openai
    # from pinecone import Pinecone
    # client = openai.OpenAI(api_key=OPENAI_API_KEY)
    # pc = Pinecone(api_key=PINECONE_API_KEY)
    # index = pc.Index(PINECONE_INDEX_NAME)
    
    # Configure pipeline
    config = PipelineConfig(chunk_size=512, chunk_overlap=50)
    pipeline = Pipeline(config=config)
    
    # Stream and embed (no file created)
    # chunk_count = 0
    # for chunk in pipeline.process_stream("data.csv"):
    #     chunk_count += 1
    #     
    #     # Embed
    #     response = client.embeddings.create(
    #         input=chunk["text"],
    #         model="text-embedding-3-small"
    #     )
    #     embedding = response.data[0].embedding
    #     
    #     # Store immediately
    #     index.upsert(vectors=[(f"chunk_{chunk_count}", embedding, chunk["metadata"])])
    #     
    #     if chunk_count % 100 == 0:
    #         print(f"Processed {chunk_count} chunks...")
    # 
    # print(f"Done! Embedded {chunk_count} chunks. No intermediate file created.")
    pass


def openai_qdrant_streaming():
    """
    OpenAI + Qdrant Integration (Streaming Mode)
    
    Prerequisites:
        pip install openai qdrant-client
    """
    # from qdrant_client import QdrantClient
    # from qdrant_client.models import PointStruct
    
    # client = openai.OpenAI(api_key="sk-...")
    # qdrant = QdrantClient(url="https://xyz.qdrant.io", api_key="qdrant-...")
    
    # config = PipelineConfig(chunk_size=512, chunk_overlap=50)
    # pipeline = Pipeline(config=config)
    
    # chunk_count = 0
    # for chunk in pipeline.process_stream("data.csv"):
    #     chunk_count += 1
    #     
    #     response = client.embeddings.create(input=chunk["text"], model="text-embedding-3-small")
    #     embedding = response.data[0].embedding
    #     
    #     qdrant.upsert(
    #         collection_name="my-chunks",
    #         points=[PointStruct(id=chunk_count, vector=embedding, payload=chunk["metadata"])]
    #     )
    #     
    #     if chunk_count % 100 == 0:
    #         print(f"Processed {chunk_count} chunks...")
    pass


def openai_weaviate_streaming():
    """
    OpenAI + Weaviate Integration (Streaming Mode)
    
    Prerequisites:
        pip install openai weaviate-client
    """
    # import openai
    # import weaviate
    # import weaviate.classes as wvc
    
    # client_w = weaviate.connect_to_wcs(
    #     cluster_url="https://xyz.weaviate.network",
    #     auth_credentials=weaviate.auth.AuthApiKey("weaviate-...")
    # )
    # client_o = openai.OpenAI(api_key="sk-...")
    # collection = client_w.collections.get("Chunk")
    
    # config = PipelineConfig(chunk_size=512, chunk_overlap=50)
    # pipeline = Pipeline(config=config)
    
    # chunk_count = 0
    # for chunk in pipeline.process_stream("data.csv"):
    #     chunk_count += 1
    #     
    #     response = client_o.embeddings.create(input=chunk["text"], model="text-embedding-3-small")
    #     embedding = response.data[0].embedding
    #     
    #     collection.data.insert(
    #         properties={"text": chunk["text"], "metadata": str(chunk["metadata"])},
    #         vector=embedding
    #     )
    #     
    #     if chunk_count % 100 == 0:
    #         print(f"Processed {chunk_count} chunks...")
    pass


def cohere_pinecone_streaming():
    """
    Cohere + Pinecone Integration (Streaming Mode)
    
    Prerequisites:
        pip install cohere pinecone-client
    """
    # import cohere
    # from pinecone import Pinecone
    
    # co = cohere.Client("co-...")
    # pc = Pinecone(api_key="pcone-...")
    # index = pc.Index("my-rag")
    
    # config = PipelineConfig(chunk_size=512, chunk_overlap=50)
    # pipeline = Pipeline(config=config)
    
    # chunk_count = 0
    # for chunk in pipeline.process_stream("data.csv"):
    #     chunk_count += 1
    #     
    #     response = co.embed(texts=[chunk["text"]], model="embed-english-v3.0")
    #     embedding = response.embeddings[0]
    #     
    #     index.upsert(vectors=[(f"chunk_{chunk_count}", embedding, chunk["metadata"])])
    #     
    #     if chunk_count % 100 == 0:
    #         print(f"Processed {chunk_count} chunks...")
    pass


def cohere_qdrant_streaming():
    """
    Cohere + Qdrant Integration (Streaming Mode)
    
    Prerequisites:
        pip install cohere qdrant-client
    """
    # import cohere
    # from qdrant_client import QdrantClient
    # from qdrant_client.models import PointStruct
    
    # co = cohere.Client("co-...")
    # qdrant = QdrantClient(url="https://xyz.qdrant.io", api_key="qdrant-...")
    
    # config = PipelineConfig(chunk_size=512, chunk_overlap=50)
    # pipeline = Pipeline(config=config)
    
    # chunk_count = 0
    # for chunk in pipeline.process_stream("data.csv"):
    #     chunk_count += 1
    #     
    #     response = co.embed(texts=[chunk["text"]], model="embed-english-v3.0")
    #     embedding = response.embeddings[0]
    #     
    #     qdrant.upsert(
    #         collection_name="my-chunks",
    #         points=[PointStruct(id=chunk_count, vector=embedding, payload=chunk["metadata"])]
    #     )
    #     
    #     if chunk_count % 100 == 0:
    #         print(f"Processed {chunk_count} chunks...")
    pass


def sentence_transformers_chromadb_streaming():
    """
    Sentence Transformers + ChromaDB Integration (Streaming Mode, FREE)
    
    Prerequisites:
        pip install sentence-transformers chromadb
    """
    try:
        from sentence_transformers import SentenceTransformer
        import chromadb
    except ImportError:
        print("Please install: pip install sentence-transformers chromadb")
        return
    
    print("Loading SentenceTransformer model (this may take a moment)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.Client()
    collection = client.get_or_create_collection("my_chunks_streaming")
    
    # Create test file
    import tempfile
    import os
    
    test_content = """The quick brown fox jumps over the lazy dog.
Machine learning is transforming how we process data.
Vector databases enable semantic search capabilities.
Krira Augment provides high-performance text chunking.
RAG pipelines combine retrieval with generation."""
    
    fd, temp_path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    with open(temp_path, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    # Configure and stream
    config = PipelineConfig(chunk_size=512, chunk_overlap=50)
    pipeline = Pipeline(config=config)
    
    chunk_count = 0
    for chunk in pipeline.process_stream(temp_path):
        chunk_count += 1
        
        # Embed locally (free, runs on your machine)
        embedding = model.encode(chunk["text"])
        
        # Store locally
        meta = chunk.get("metadata")
        collection.add(
            ids=[f"chunk_{chunk_count}"],
            embeddings=[embedding.tolist()],
            metadatas=[meta] if meta else None,
            documents=[chunk["text"]]
        )
        
        print(f"  ✓ Processed chunk {chunk_count}: {chunk['text'][:40]}...")
    
    print(f"\n✅ Done! {chunk_count} chunks embedded. All local, no API costs.")
    print("   No intermediate file was created!")
    
    # Demo query
    query = "data processing"
    query_embedding = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )
    
    print(f"\n🔍 Query: '{query}'")
    print(f"   Top results: {results['documents']}")
    
    # Cleanup
    os.unlink(temp_path)


def huggingface_faiss_streaming():
    """
    Hugging Face + FAISS Integration (Streaming Mode, FREE)
    
    Prerequisites:
        pip install transformers torch faiss-cpu
    """
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        import faiss
        import numpy as np
    except ImportError:
        print("Please install: pip install transformers torch faiss-cpu")
        return
    
    print("Loading Hugging Face model (this may take a moment)...")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model_hf = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    index = faiss.IndexFlatL2(384)
    
    # Create test file
    import tempfile
    import os
    
    test_content = """The quick brown fox jumps over the lazy dog.
Machine learning is transforming how we process data.
Vector databases enable semantic search capabilities.
Krira Augment provides high-performance text chunking.
RAG pipelines combine retrieval with generation."""
    
    fd, temp_path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    with open(temp_path, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    # Configure and stream
    config = PipelineConfig(chunk_size=512, chunk_overlap=50)
    pipeline = Pipeline(config=config)
    
    chunk_count = 0
    embeddings_batch = []
    BATCH_SIZE = 100
    
    import torch.nn.functional as F

    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    for chunk in pipeline.process_stream(temp_path):
        chunk_count += 1
        
        # Embed locally
        inputs = tokenizer(chunk["text"], return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model_hf(**inputs)
            embedding = mean_pooling(outputs, inputs['attention_mask'])
            embedding = F.normalize(embedding, p=2, dim=1)
            embedding = embedding.squeeze().numpy()
        
        embeddings_batch.append(embedding)
        
        # Add to FAISS in batches
        if len(embeddings_batch) >= BATCH_SIZE:
            embeddings_array = np.array(embeddings_batch).astype('float32')
            index.add(embeddings_array)
            embeddings_batch = []
            print(f"  Added batch to FAISS, total: {chunk_count} chunks")
    
    # Add remaining embeddings
    if embeddings_batch:
        embeddings_array = np.array(embeddings_batch).astype('float32')
        index.add(embeddings_array)
    
    print(f"\n✅ Done! {chunk_count} chunks embedded in FAISS index.")
    print(f"   Index contains {index.ntotal} vectors")
    print("   No intermediate file was created!")
    
    # Cleanup
    os.unlink(temp_path)


def production_error_handling_streaming():
    """
    Production-Ready Streaming with Error Handling
    
    This example shows how to implement proper error handling
    for production RAG pipelines.
    """
    import time
    
    # Simulate API calls with mock functions
    def mock_embed(text):
        """Simulated embedding function."""
        if "error" in text.lower():
            raise Exception("Mock API error")
        return [0.1] * 384  # Mock 384-dim embedding
    
    def mock_store(chunk_id, embedding, metadata):
        """Simulated vector store."""
        pass
    
    # Create test file
    import tempfile
    import os
    
    test_content = """Line 1: Normal text.
Line 2: More normal text.
Line 3: Even more text.
Line 4: Final text."""
    
    fd, temp_path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    with open(temp_path, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    config = PipelineConfig(chunk_size=512, chunk_overlap=50)
    pipeline = Pipeline(config=config)
    
    chunk_count = 0
    error_count = 0
    
    for chunk in pipeline.process_stream(temp_path):
        chunk_count += 1
        
        try:
            # Embed
            embedding = mock_embed(chunk["text"])
            
            # Store
            mock_store(f"chunk_{chunk_count}", embedding, chunk["metadata"])
            
        except Exception as e:
            error_count += 1
            print(f"  ⚠️  Error on chunk {chunk_count}: {e}")
            
            # Retry logic
            if "rate_limit" in str(e).lower():
                print("  Rate limited, waiting 60 seconds...")
                time.sleep(60)
        
        if chunk_count % 100 == 0:
            print(f"  Processed {chunk_count} chunks, {error_count} errors")
    
    print(f"\n✅ Done! {chunk_count} chunks processed, {error_count} errors")
    
    # Cleanup
    os.unlink(temp_path)


if __name__ == "__main__":
    print("=" * 60)
    print("Krira Augment - Streaming Integration Examples")
    print("=" * 60)
    print()
    
    # Run the free examples
    print("1. Sentence Transformers + ChromaDB (FREE)")
    print("-" * 60)
    sentence_transformers_chromadb_streaming()
    
    print()
    print("2. Production Error Handling Demo")
    print("-" * 60)
    production_error_handling_streaming()
    
    print()
    print("=" * 60)
    print("✅ All demos completed!")
    print("=" * 60)
