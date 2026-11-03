# Chroma Vector Database Implementation

[Chroma](https://www.trychroma.com/) is an AI-native open-source embedding database designed to simplify building LLM-powered applications, memory systems, and semantic search.

---

## 1. Overview & Architecture

- **License**: Apache 2.0
- **Indexing Engine**: HNSW (via `hnswlib` or native Rust backend in modern versions)
- **Primary Storage**: SQLite / ClickHouse / Embedded files
- **Key Characteristics**:
  - **Embedded or Client-Server**: Chroma can run completely in-process (Python native, no server needed) or as a client-server architecture over HTTP.
  - **Developer-Friendly**: Minimal configuration needed to store documents, metadata, and embeddings together.
  - **Rich Filtering**: Supports metadata queries with MongoDB-like filter syntax (`$eq`, `$in`, `$and`, `$or`).

---

## 2. Prerequisites & Setup

### Option A: Using Docker (Recommended for Server Mode)

From the project root:
```bash
docker compose up -d chroma
```
The Chroma server will be available at `http://localhost:8000`. You can test connection with:
```bash
curl http://localhost:8000/api/v1/heartbeat
```

### Option B: Embedded In-Memory / Local Mode (No Docker Needed)

`chroma/main.py` detects whether the Docker HTTP server is running. If not running, it automatically falls back to an embedded `PersistentClient` saving to `chroma/chroma_data/`.

---

## 3. Installation & Running

1. Activate your virtual environment and install requirements:
   ```bash
   pip install -r chroma/requirements.txt
   ```

2. Run the demonstration script:
   ```bash
   python chroma/main.py
   ```

---

## 4. Code Flow Explained

The `main.py` script follows our standardized vector playground lifecycle:

1. **Client Initialization**: Attempts connection to `chromadb.HttpClient(host="localhost", port=8000)`. Falls back gracefully to `chromadb.PersistentClient()`.
2. **Collection Creation**: Drops existing `articles` collection for reproducibility and recreates it with `{"hnsw:space": "cosine"}`.
3. **Data Ingestion**: Loads sample vectors from `dataset/sample_embeddings.json` and inserts documents, embeddings, and metadata via `collection.add(...)`.
4. **Vector Similarity Search**: Executes an approximate nearest neighbor (ANN) search with `n_results=3` using the query vector.
5. **Metadata Filtering**: Queries with a boolean constraint `where={"category": "culinary"}` to ensure only relevant domain results are returned.
