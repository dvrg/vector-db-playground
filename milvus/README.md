# Milvus Vector Database Implementation

[Milvus](https://milvus.io/) is an open-source, highly scalable vector database built for massive enterprise similarity search and AI applications, capable of handling billions of vectors.

---

## 1. Overview & Architecture

- **License**: Apache 2.0
- **Indexing Engines**: HNSW, IVF_FLAT, IVF_SQ8, SCaNN, DiskANN
- **Storage Subsystems**: Object storage (MinIO / S3) + Metadata store (etcd) + Log broker (Pulsar / Kafka / RocksDB)
- **Key Characteristics**:
  - **Disaggregated Storage & Compute**: Query nodes, data nodes, and index nodes can scale horizontally and independently.
  - **Milvus Lite**: Lightweight embedded Python version of Milvus that runs directly without containers for quick prototyping.
  - **Unified MilvusClient**: The modern Python SDK (`pymilvus>=2.4.0`) offers a simplified, user-friendly API for collections, inserts, and hybrid vector search.

---

## 2. Prerequisites & Setup

### Option A: Running Distributed / Standalone with Docker Compose

From the project root:
```bash
docker compose up -d etcd minio milvus
```

- **Milvus gRPC API**: `localhost:19530`
- **Milvus Metrics**: `localhost:9091`
- **MinIO Console**: `http://localhost:9001` (Credentials: `minioadmin` / `minioadmin`)

### Option B: Milvus Lite (In-Process Fallback)

If Docker is not running, `milvus/main.py` automatically initializes Milvus Lite using a local SQLite-like backend (`milvus/milvus_demo.db`), requiring zero infrastructure setup.

---

## 3. Installation & Running

1. Activate your virtual environment and install requirements:
   ```bash
   pip install -r milvus/requirements.txt
   ```

2. Run the demonstration script:
   ```bash
   python milvus/main.py
   ```

---

## 4. Code Flow Explained

1. **Client Initialization**: Connects to `MilvusClient(uri="http://localhost:19530")` or falls back to `MilvusClient(uri="milvus_demo.db")`.
2. **Collection Lifecycle**: Calls `client.drop_collection("articles")` and `client.create_collection(...)` configured with `dimension=8` and `metric_type="COSINE"`.
3. **Data Ingestion**: Inserts dictionary records containing vector embeddings and scalar attributes (title, category, tags, content).
4. **Vector Search**: Invokes `client.search(...)` retrieving the top-3 nearest matches.
5. **Boolean Filtering**: Applies the expression string `filter='category == "culinary"'` during the query pass.
