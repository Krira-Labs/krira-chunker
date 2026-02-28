# Krira Augment — Krira Chunker (Beta)

**High-Performance Rust Chunking Engine for RAG Pipelines**

Process gigabytes of text in seconds. **40x faster than LangChain** with **O(1) memory usage**.

> ⚠️ **Beta Software** — Actively developed. APIs may change. We welcome bug reports and feedback.

---

## Installation

```bash
pip install krira-augment
```

---

## Quick Usage

```python
from krira_augment.krira_chunker import Pipeline, PipelineConfig, SplitStrategy

config = PipelineConfig(
    chunk_size=512,
    strategy=SplitStrategy.SMART,
    clean_html=True,
    clean_unicode=True,
)

pipeline = Pipeline(config=config)

result = pipeline.process("sample.csv", output_path="output.jsonl")

print(result)
print(f"Chunks Created: {result.chunks_created}")
print(f"Execution Time: {result.execution_time:.2f}s")
print(f"Throughput: {result.mb_per_second:.2f} MB/s")
print(f"Preview: {result.preview_chunks[:3]}")
```

---

## Performance Benchmark

Processing **42.4 million chunks** in **113.79 seconds** (47.51 MB/s).

```text
============================================================
✅ KRIRA AUGMENT - Processing Complete
============================================================
📊 Chunks Created:  42,448,765
⏱️  Execution Time:  113.79 seconds
🚀 Throughput:      47.51 MB/s
📁 Output File:     output.jsonl
============================================================

📝 Preview (Top 3 Chunks):
------------------------------------------------------------
[1] event_time,event_type,product_id,category_id,category_code,brand,price,user_id,user_session
[2] 2019-10-01 00:00:00 UTC,view,44600062,2103807459595387724,,shiseido,35.79,541312140,72d76fde-8bb3-4e00-8c23-a032dfed738c
[3] 2019-10-01 00:00:00 UTC,view,3900821,2053013552326770905,appliances.environment.water_heater...
```

---

## Krira-Chunker Architecture

<img width="2251" height="1624" alt="diagram-export-1-5-2026-10_24_14-PM" src="https://github.com/user-attachments/assets/2247c233-95df-4984-b091-6720bcd45372" />

---

## How Krira-Chunker Works

<img width="1600" height="893" alt="image" src="https://github.com/user-attachments/assets/76ce6157-7e76-423d-9da0-a78d8bf70979" />

---

## Chunking Strategies

Krira Chunker supports three strategies:

- **Fixed** — Splits by exact character/token count. Predictable but ignores semantic boundaries. Best for uniform data like CSVs.
- **Structured** — Respects document structure such as headings, paragraphs, and sections. Best for PDFs and Word documents.
- **Smart (Hybrid)** — Combines both: structure-aware splitting with configurable size limits. Recommended for most use cases.

---

## Supported Formats

| Format | Extension | Method |
|--------|-----------|--------|
| **CSV** | `.csv` | Direct processing |
| **Text** | `.txt` | Direct processing |
| **JSONL** | `.jsonl` | Direct processing |
| **JSON** | `.json` | Auto-flattening |
| **PDF** | `.pdf` | pdfplumber extraction |
| **Word** | `.docx` | python-docx extraction |
| **Excel** | `.xlsx` | openpyxl extraction |
| **XML** | `.xml` | ElementTree parsing |
| **URLs** | `http://` | BeautifulSoup scraping |

### PDF Support — Known Limitations

| PDF Type | Supported |
|----------|-----------|
| ✅ Text-based PDFs | Yes |
| ✅ Mixed content PDFs | Yes |
| ⚠️ Multi-column layouts | Partial |
| 🔄 Scanned / image-based PDFs | Coming soon (OCR roadmap) |
| ❌ Password protected PDFs | Not supported |

If you encounter unexpected output from a specific PDF, please open an issue with the file — we actively fix these cases.

---

## Complete Example: Local (ChromaDB) — FREE

No API keys required. Runs entirely on your machine.

```bash
pip install sentence-transformers chromadb
```

```python
from krira_augment.krira_chunker import Pipeline, PipelineConfig
from sentence_transformers import SentenceTransformer
import chromadb
import json

# Step 1: Chunk the file (Rust Core)
config = PipelineConfig(chunk_size=512, chunk_overlap=50)
pipeline = Pipeline(config=config)
result = pipeline.process("sample.csv", output_path="chunks.jsonl")

print(f"Chunks Created: {result.chunks_created}")
print(f"Execution Time: {result.execution_time:.2f}s")
print(f"Throughput: {result.mb_per_second:.2f} MB/s")

# Step 2: Embed and store (Local)
model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.Client()
collection = client.get_or_create_collection("my_rag_db")

with open("chunks.jsonl", "r") as f:
    for line_num, line in enumerate(f, 1):
        chunk = json.loads(line)
        embedding = model.encode(chunk["text"])
        meta = chunk.get("metadata")
        collection.add(
            ids=[f"chunk_{line_num}"],
            embeddings=[embedding.tolist()],
            metadatas=[meta] if meta else None,
            documents=[chunk["text"]]
        )
        if line_num % 100 == 0:
            print(f"Processed {line_num} chunks...")

print("Done! All chunks stored in ChromaDB.")
```

---

## Cloud Integrations (OpenAI, Pinecone, Cohere)

### OpenAI + Pinecone

```bash
pip install openai pinecone-client
```

```python
from openai import OpenAI
from pinecone import Pinecone

OPENAI_API_KEY = "sk-..."
PINECONE_API_KEY = "pcone-..."
PINECONE_INDEX_NAME = "my-rag"

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

with open("chunks.jsonl", "r") as f:
    for line_num, line in enumerate(f, 1):
        chunk = json.loads(line)
        response = client.embeddings.create(
            input=chunk["text"],
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
        index.upsert(vectors=[(f"chunk_{line_num}", embedding, chunk.get("metadata", {}))])
        if line_num % 100 == 0:
            print(f"Processed {line_num} chunks...")
```

### OpenAI + Qdrant

```bash
pip install openai qdrant-client
```

```python
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

client = OpenAI(api_key="sk-...")
qdrant = QdrantClient(url="https://xyz.qdrant.io", api_key="qdrant-...")

with open("chunks.jsonl", "r") as f:
    for line_num, line in enumerate(f, 1):
        chunk = json.loads(line)
        response = client.embeddings.create(input=chunk["text"], model="text-embedding-3-small")
        embedding = response.data[0].embedding
        qdrant.upsert(
            collection_name="my-chunks",
            points=[PointStruct(id=line_num, vector=embedding, payload=chunk.get("metadata", {}))]
        )
        if line_num % 100 == 0:
            print(f"Processed {line_num} chunks...")
```

### OpenAI + Weaviate

```bash
pip install openai weaviate-client
```

```python
import weaviate
import weaviate.classes as wvc
from openai import OpenAI

client_w = weaviate.connect_to_wcs(
    cluster_url="https://xyz.weaviate.network",
    auth_credentials=weaviate.auth.AuthApiKey("weaviate-...")
)
client_o = OpenAI(api_key="sk-...")
collection = client_w.collections.get("Chunk")

with open("chunks.jsonl", "r") as f:
    for line_num, line in enumerate(f, 1):
        chunk = json.loads(line)
        response = client_o.embeddings.create(input=chunk["text"], model="text-embedding-3-small")
        embedding = response.data[0].embedding
        collection.data.insert(
            properties={"text": chunk["text"], "metadata": str(chunk.get("metadata", {}))},
            vector=embedding
        )
        if line_num % 100 == 0:
            print(f"Processed {line_num} chunks...")
```

### Cohere + Pinecone

```bash
pip install cohere pinecone-client
```

```python
import cohere
from pinecone import Pinecone

co = cohere.Client("co-...")
pc = Pinecone(api_key="pcone-...")
index = pc.Index("my-rag")

with open("chunks.jsonl", "r") as f:
    for line_num, line in enumerate(f, 1):
        chunk = json.loads(line)
        response = co.embed(texts=[chunk["text"]], model="embed-english-v3.0")
        embedding = response.embeddings[0]
        index.upsert(vectors=[(f"chunk_{line_num}", embedding, chunk.get("metadata", {}))])
        if line_num % 100 == 0:
            print(f"Processed {line_num} chunks...")
```

### Hugging Face + FAISS (FREE)

```bash
pip install transformers torch faiss-cpu
```

```python
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import faiss
import numpy as np
import json

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
index = faiss.IndexFlatL2(384)

batch_embeddings = []
BATCH_SIZE = 64

with open("chunks.jsonl", "r") as f:
    for line_num, line in enumerate(f, 1):
        chunk = json.loads(line)
        encoded_input = tokenizer(chunk["text"], padding=True, truncation=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        batch_embeddings.append(sentence_embeddings.squeeze().numpy())
        if len(batch_embeddings) >= BATCH_SIZE:
            index.add(np.vstack(batch_embeddings).astype('float32'))
            batch_embeddings = []
        if line_num % 100 == 0:
            print(f"Processed {line_num} chunks...")

if batch_embeddings:
    index.add(np.vstack(batch_embeddings).astype('float32'))

faiss.write_index(index, "my_vectors.index")
print("Done! Vectors saved to my_vectors.index")
```

---

## Streaming Mode (No Files)

Process chunks without saving to disk — maximum efficiency for real-time pipelines.

### OpenAI + Pinecone (Streaming)

```python
from krira_augment.krira_chunker import Pipeline, PipelineConfig
from openai import OpenAI
from pinecone import Pinecone

client = OpenAI(api_key="sk-...")
pc = Pinecone(api_key="pcone-...")
index = pc.Index("my-rag")

config = PipelineConfig(chunk_size=512, chunk_overlap=50)
pipeline = Pipeline(config=config)

chunk_count = 0
for chunk in pipeline.process_stream("data.csv"):
    chunk_count += 1
    response = client.embeddings.create(input=chunk["text"], model="text-embedding-3-small")
    embedding = response.data[0].embedding
    index.upsert(vectors=[(f"chunk_{chunk_count}", embedding, chunk["metadata"])])
    if chunk_count % 100 == 0:
        print(f"Processed {chunk_count} chunks...")

print(f"Done! Embedded {chunk_count} chunks.")
```

---

## Streaming vs File-Based

| Feature | File-Based | Streaming |
|---------|------------|-----------|
| **Disk I/O** | Creates chunks.jsonl | None |
| **Memory Usage** | O(1) constant | O(1) constant |
| **Speed** | Chunking + Embedding | Overlapped (faster) |
| **Use Case** | Large files, batch processing | Real-time, no storage |
| **Flexibility** | Can re-process chunks | Single pass only |

**Use Streaming when** you want maximum speed, no disk writes, and don't need to inspect chunks later.

**Use File-Based when** you want to debug output, re-process with different embeddings, or share chunks with your team.

---

## Error Handling

```python
from krira_augment.krira_chunker import Pipeline, PipelineConfig
from openai import OpenAI
from pinecone import Pinecone
import time

client = OpenAI(api_key="sk-...")
pc = Pinecone(api_key="pcone-...")
index = pc.Index("my-rag")

config = PipelineConfig(chunk_size=512, chunk_overlap=50)
pipeline = Pipeline(config=config)

chunk_count = 0
error_count = 0

for chunk in pipeline.process_stream("data.csv"):
    chunk_count += 1
    try:
        response = client.embeddings.create(input=chunk["text"], model="text-embedding-3-small")
        embedding = response.data[0].embedding
        index.upsert(vectors=[(f"chunk_{chunk_count}", embedding, chunk["metadata"])])
    except Exception as e:
        error_count += 1
        print(f"Error on chunk {chunk_count}: {e}")
        if "rate_limit" in str(e).lower():
            print("Rate limited, waiting 60 seconds...")
            time.sleep(60)
    if chunk_count % 100 == 0:
        print(f"Processed {chunk_count} chunks, {error_count} errors")

print(f"Done! {chunk_count} chunks processed, {error_count} errors")
```

---

## Provider Comparison

| Embedding | Vector Store | Cost | API Keys | Streaming |
|-----------|--------------|------|----------|-----------|
| OpenAI | Pinecone | Paid | 2 | ✅ |
| OpenAI | Qdrant | Paid | 2 | ✅ |
| OpenAI | Weaviate | Paid | 2 | ✅ |
| Cohere | Pinecone | Paid | 2 | ✅ |
| Cohere | Qdrant | Paid | 2 | ✅ |
| SentenceTransformers | ChromaDB | **FREE** | 0 | ✅ |
| Hugging Face | FAISS | **FREE** | 0 | ✅ |

---

## API Keys Setup

- **OpenAI:** https://platform.openai.com/api-keys
- **Cohere:** https://dashboard.cohere.com/api-keys
- **Pinecone:** https://app.pinecone.io/
- **Qdrant:** https://cloud.qdrant.io/
- **Weaviate:** https://console.weaviate.cloud/

---

## Development

```bash
# Clone the repo
git clone https://github.com/Krira-Labs/krira-chunker

# Install Maturin
pip install maturin

# Build and install locally
maturin develop
```

---

## Contributing & Feedback

Found a bug? Have a feature request? We actively respond to issues.

- 🐛 [Open an Issue](https://github.com/Krira-Labs/krira-chunker/issues)
- 💬 [Start a Discussion](https://github.com/Krira-Labs/krira-chunker/discussions)
- 📧 Reach us at: kriralabs.com

If a specific file format produces unexpected output, please share a sample in the issue — we'll fix it.

---

*Built by [Krira Labs](https://kriralabs.com) — Building the nervous system for the Intelligence Age.*
