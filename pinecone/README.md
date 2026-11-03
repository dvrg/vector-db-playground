# Pinecone Vector Database Implementation

[Pinecone](https://www.pinecone.io/) is a fully managed, cloud-native vector database designed for high availability, low-latency search, and effortless scalability without managing infrastructure.

---

## 1. Overview & Architecture

- **License**: Commercial / Proprietary (Cloud SaaS)
- **Deployment Model**: Fully Managed Serverless or Dedicated Pods (AWS, GCP, Azure)
- **Indexing Engine**: Proprietary graph and tree-based approximate nearest neighbor algorithm
- **Key Characteristics**:
  - **Zero Infrastructure Ops**: No Kubernetes clusters, sharding, or storage volumes to maintain.
  - **Serverless Architecture**: Scales automatically based on read and write workloads with pay-per-request pricing.
  - **Metadata Filtering**: Supports flexible filtering syntax (`$eq`, `$in`, `$gt`, `$gte`, `$and`, etc.) combined with vector queries.
  - **Namespaces**: Logical partitioning within an index for multi-tenant isolation.

---

## 2. Prerequisites & Setup

Because Pinecone is entirely cloud-based, it is not run inside local Docker Compose.

1. Sign up for a free tier account at [app.pinecone.io](https://app.pinecone.io).
2. Generate an API Key from the Pinecone dashboard.
3. Add the key to your `.env` file at the root of this repository:
   ```env
   PINECONE_API_KEY=your_actual_api_key_here
   PINECONE_INDEX_NAME=sample-articles
   PINECONE_CLOUD=aws
   PINECONE_REGION=us-east-1
   ```

*Note: If no API key is provided, `pinecone/main.py` automatically runs in demonstration simulation mode, showing the exact API calls and computing identical cosine similarity values.*

---

## 3. Installation & Running

1. Activate your virtual environment and install requirements:
   ```bash
   pip install -r pinecone/requirements.txt
   ```

2. Run the demonstration script:
   ```bash
   python pinecone/main.py
   ```

---

## 4. Code Flow Explained

1. **Client Initialization**: Initializes the client using `Pinecone(api_key=...)`.
2. **Serverless Index Lifecycle**: Checks `pc.list_indexes()`, creates a fresh index with `ServerlessSpec(cloud="aws", region="us-east-1")`, and connects to `pc.Index(INDEX_NAME)`.
3. **Data Upsert**: Formats records into `{"id": ..., "values": [...], "metadata": {...}}` and calls `index.upsert(...)`.
4. **Vector Search**: Calls `index.query(vector=..., top_k=3, include_metadata=True)`.
5. **Metadata Filtering**: Passes `filter={"category": {"$eq": "culinary"}}` directly into `index.query(...)`.
