# Weaviate Vector Database Implementation

[Weaviate](https://weaviate.io/) is an open-source, AI-first vector database designed for scale and enterprise workloads. It supports dense vectors, BM25 keyword search, and multi-modal embeddings with integrated ML models.

---

## 1. Overview & Architecture

- **License**: BSD 3-Clause (Open Source)
- **Indexing Engines**: HNSW graph (hierarchical navigable small world), Dynamic/Flat index, inverted index for hybrid search
- **Communication Protocol**: gRPC (high-throughput data plane) + REST/GraphQL (control plane)
- **Key Characteristics**:
  - **Modern v4 Python SDK**: Full type safety, collections management, and dynamic batching.
  - **Hybrid Search**: Native combination of sparse BM25 keyword scoring and dense vector similarity.
  - **Modular Vectorization**: Can plug directly into OpenAI, Cohere, HuggingFace, or custom local models.

---

## 2. Prerequisites & Setup

### Running with Docker

From the project root:
```bash
docker compose up -d weaviate
```

- **REST API**: `http://localhost:8080/v1/meta`
- **gRPC API**: `localhost:50051`

---

## 3. Installation & Running

1. Activate your virtual environment and install requirements:
   ```bash
   pip install -r weaviate/requirements.txt
   ```

2. Run the demonstration script:
   ```bash
   python weaviate/main.py
   ```

---

## 4. Code Flow Explained

1. **Client Connection**: Uses `weaviate.connect_to_local(host="localhost", port=8080, grpc_port=50051)`.
2. **Collection Creation**: Configures `client.collections.create(...)` with explicit property typings (`doc_id`, `title`, `category`, `tags`, `content`) and HNSW Cosine distance.
3. **Dynamic Batching**: Ingests objects using `articles.batch.dynamic()` which automatically tunes batch sizes for network conditions.
4. **Vector Similarity Search**: Queries via `articles.query.near_vector(near_vector=..., limit=3)` returning metadata with exact distance metrics.
5. **Property Filtering**: Employs `Filter.by_property("category").equal("culinary")` alongside vector retrieval.
