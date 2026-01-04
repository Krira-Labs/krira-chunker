"""
File-Based Integration Examples for Krira Augment

This script contains all integration examples for file-based chunking mode.
Each function demonstrates how to use Krira Augment with a different
embedding provider and vector store combination.

Run with: python examples/file_based_integrations.py
"""

from krira_augment.krira_chunker import Pipeline, PipelineConfig, SplitStrategy
import json


def openai_pinecone_example():
    """
    OpenAI + Pinecone Integration
    
    Prerequisites:
        pip install openai pinecone-client
    """
    # import openai
    # from pinecone import Pinecone
    
    # API Keys
    OPENAI_API_KEY = "sk-..."        # https://platform.openai.com/api-keys
    PINECONE_API_KEY = "pcone-..."   # https://app.pinecone.io/
    PINECONE_INDEX_NAME = "my-rag"
    
    # Step 1: Chunk the file
    config = PipelineConfig(chunk_size=512, chunk_overlap=50)
    pipeline = Pipeline(config=config)
    result = pipeline.process("sample.csv", output_path="chunks.jsonl")
    
    print(f"Chunks Created: {result.chunks_created}")
    print(f"Execution Time: {result.execution_time:.2f}s")
    
    # Step 2: Embed and store (uncomment when ready)
    # client = openai.OpenAI(api_key=OPENAI_API_KEY)
    # pc = Pinecone(api_key=PINECONE_API_KEY)
    # index = pc.Index(PINECONE_INDEX_NAME)
    # 
    # with open("chunks.jsonl", "r") as f:
    #     for line_num, line in enumerate(f, 1):
    #         chunk = json.loads(line)
    #         
    #         response = client.embeddings.create(
    #             input=chunk["text"],
    #             model="text-embedding-3-small"
    #         )
    #         embedding = response.data[0].embedding
    #         
    #         index.upsert(vectors=[(f"chunk_{line_num}", embedding, chunk.get("metadata", {}))])
    #         
    #         if line_num % 100 == 0:
    #             print(f"Processed {line_num} chunks...")
    # 
    # print("Done! All chunks embedded and stored in Pinecone.")


def openai_qdrant_example():
    """
    OpenAI + Qdrant Integration
    
    Prerequisites:
        pip install openai qdrant-client
    """
    # import openai
    # from qdrant_client import QdrantClient
    # from qdrant_client.models import PointStruct
    
    # client = openai.OpenAI(api_key="sk-...")
    # qdrant = QdrantClient(url="https://xyz.qdrant.io", api_key="qdrant-...")
    
    # with open("chunks.jsonl", "r") as f:
    #     for line_num, line in enumerate(f, 1):
    #         chunk = json.loads(line)
    #         response = client.embeddings.create(input=chunk["text"], model="text-embedding-3-small")
    #         embedding = response.data[0].embedding
    #         qdrant.upsert(
    #             collection_name="my-chunks", 
    #             points=[PointStruct(id=line_num, vector=embedding, payload=chunk.get("metadata", {}))]
    #         )
    #         
    #         if line_num % 100 == 0:
    #             print(f"Processed {line_num} chunks...")
    pass


def openai_weaviate_example():
    """
    OpenAI + Weaviate Integration
    
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
    
    # with open("chunks.jsonl", "r") as f:
    #     for line_num, line in enumerate(f, 1):
    #         chunk = json.loads(line)
    #         response = client_o.embeddings.create(input=chunk["text"], model="text-embedding-3-small")
    #         embedding = response.data[0].embedding
    #         
    #         collection.data.insert(
    #             properties={"text": chunk["text"], "metadata": str(chunk.get("metadata", {}))},
    #             vector=embedding
    #         )
    #         
    #         if line_num % 100 == 0:
    #             print(f"Processed {line_num} chunks...")
    pass


def cohere_pinecone_example():
    """
    Cohere + Pinecone Integration
    
    Prerequisites:
        pip install cohere pinecone-client
    """
    # import cohere
    # from pinecone import Pinecone
    
    # co = cohere.Client("co-...")
    # pc = Pinecone(api_key="pcone-...")
    # index = pc.Index("my-rag")
    
    # with open("chunks.jsonl", "r") as f:
    #     for line_num, line in enumerate(f, 1):
    #         chunk = json.loads(line)
    #         response = co.embed(texts=[chunk["text"]], model="embed-english-v3.0")
    #         embedding = response.embeddings[0]
    #         index.upsert(vectors=[(f"chunk_{line_num}", embedding, chunk.get("metadata", {}))])
    #         
    #         if line_num % 100 == 0:
    #             print(f"Processed {line_num} chunks...")
    pass


def cohere_qdrant_example():
    """
    Cohere + Qdrant Integration
    
    Prerequisites:
        pip install cohere qdrant-client
    """
    # import cohere
    # from qdrant_client import QdrantClient
    # from qdrant_client.models import PointStruct
    
    # co = cohere.Client("co-...")
    # qdrant = QdrantClient(url="https://xyz.qdrant.io", api_key="qdrant-...")
    
    # with open("chunks.jsonl", "r") as f:
    #     for line_num, line in enumerate(f, 1):
    #         chunk = json.loads(line)
    #         response = co.embed(texts=[chunk["text"]], model="embed-english-v3.0")
    #         embedding = response.embeddings[0]
    #         qdrant.upsert(
    #             collection_name="my-chunks",
    #             points=[PointStruct(id=line_num, vector=embedding, payload=chunk.get("metadata", {}))]
    #         )
    #         
    #         if line_num % 100 == 0:
    #             print(f"Processed {line_num} chunks...")
    pass


def sentence_transformers_chromadb_example():
    """
    Sentence Transformers + ChromaDB Integration (FREE - No API Keys!)
    
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
    collection = client.get_or_create_collection("my_chunks")
    
    # First, create chunks
    config = PipelineConfig(chunk_size=512, chunk_overlap=50)
    pipeline = Pipeline(config=config)
    
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
    
    result = pipeline.process(temp_path, output_path="demo_chunks.jsonl")
    print(f"Created {result.chunks_created} chunks")
    
    # Embed and store
    with open("demo_chunks.jsonl", "r") as f:
        for line_num, line in enumerate(f, 1):
            chunk = json.loads(line)
            embedding = model.encode(chunk["text"])
            collection.add(
                ids=[f"chunk_{line_num}"],
                embeddings=[embedding.tolist()],
                metadatas=[{"text": chunk["text"][:100]}],  # Store preview
                documents=[chunk["text"]]
            )
    
    print(f"Embedded {line_num} chunks in ChromaDB")
    
    # Demo query
    query = "data processing"
    query_embedding = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )
    
    print(f"\nQuery: '{query}'")
    print(f"Top results: {results['documents']}")
    
    # Cleanup
    os.unlink(temp_path)
    os.unlink("demo_chunks.jsonl")


def huggingface_faiss_example():
    """
    Hugging Face + FAISS Integration (FREE - No API Keys!)
    
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
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    index = faiss.IndexFlatL2(384)  # 384 is the embedding dimension
    
    # First, create chunks
    config = PipelineConfig(chunk_size=512, chunk_overlap=50)
    pipeline = Pipeline(config=config)
    
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
    
    result = pipeline.process(temp_path, output_path="demo_chunks.jsonl")
    print(f"Created {result.chunks_created} chunks")
    
    # Embed and store
    import torch.nn.functional as F

    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    embeddings_list = []
    with open("demo_chunks.jsonl", "r") as f:
        for line_num, line in enumerate(f, 1):
            chunk = json.loads(line)
            inputs = tokenizer(chunk["text"], return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                embedding = mean_pooling(outputs, inputs['attention_mask'])
                embedding = F.normalize(embedding, p=2, dim=1)
                embedding = embedding.squeeze().numpy()
            embeddings_list.append(embedding)
    
    embeddings_array = np.array(embeddings_list).astype('float32')
    index.add(embeddings_array)
    
    print(f"Added {len(embeddings_list)} embeddings to FAISS index")
    
    # Cleanup
    os.unlink(temp_path)
    os.unlink("demo_chunks.jsonl")


if __name__ == "__main__":
    print("=" * 60)
    print("Krira Augment - File-Based Integration Examples")
    print("=" * 60)
    print()
    
    # Run the free examples
    print("Running: Sentence Transformers + ChromaDB (FREE)")
    print("-" * 60)
    sentence_transformers_chromadb_example()
    
    print()
    print("=" * 60)
    print("✅ Demo completed!")
    print("=" * 60)
