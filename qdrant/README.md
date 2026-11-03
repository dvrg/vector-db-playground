# Qdrant Vector Database Implementation

[Qdrant](https://qdrant.tech/) is an open-source, high-performance vector similarity search engine and vector database written in Rust. It provides fast search with extended filtering capabilities.

---

## 1. Overview & Architecture

- **License**: Apache 2.0
- **Indexing Engine**: Custom HNSW implementation in Rust with payload-based filtering
- **Primary Storage**: Memory-mapped files (mmap) or on-disk RocksDB
- **Key Characteristics**:
  - **Payload-Aware HNSW**: Qdrant modifies the HNSW traversal to navigate filtered graphs directly, avoiding post-filtering bottlenecks.
  - **Scalar & Product Quantization**: Highly efficient memory compression for massive vector collections.
  - **Interactive Web Dashboard**: Built-in visual dashboard for inspecting collections, points, and query execution.
  - **Dual API**: Full REST (port 6333) and high-throughput gRPC (port 6334) endpoints.

---

## 2. Prerequisites & Setup

### Running with Docker

From the project root:
```bash
docker compose up -d qdrant
```

- **REST API & Dashboard**: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)
- **gRPC API**: `localhost:6334`

### In-Memory Fallback

If Docker is not running, `qdrant/main.py` automatically initializes an in-memory instance (`QdrantClient(":memory:")`), allowing local testing without infrastructure setup.

---

## 3. Installation & Running

1. Activate your virtual environment and install requirements:
   ```bash
   pip install -r qdrant/requirements.txt
   ```

2. Run the demonstration script:
   ```bash
   python qdrant/main.py
   ```

---

## 4. Code Flow Explained

1. **Client Initialization**: Connects to `QdrantClient(host="localhost", port=6333)`.
2. **Collection Creation**: Uses `client.recreate_collection` with `models.VectorParams(size=8, distance=Distance.COSINE)`.
3. **Point Upsert**: Maps JSON records into `models.PointStruct` items carrying vector coordinates and payload dictionaries.
4. **Vector Search**: Calls `client.search(...)` retrieving the top-3 nearest neighbors by cosine similarity.
5. **Payload Filtering**: Applies `models.Filter(must=[models.FieldCondition(key="category", match=models.MatchValue(value="culinary"))])` during the graph traversal.
