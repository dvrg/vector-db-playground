"""
Milvus Vector Database Implementation
Demonstrating:
  1. Connection / Client Initialization (Milvus Standalone or Milvus Lite Fallback)
  2. Collection Creation & Data Ingestion (Unified MilvusClient API)
  3. Vector Similarity Search (Top-K with Cosine Distance)
  4. Filtered Vector Similarity Search (Boolean Expression Filtering)
"""

import json
import os
from pathlib import Path

# Load environment variables if python-dotenv is installed
root_dir = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(root_dir / ".env")
except ImportError:
    pass

from pymilvus import MilvusClient

# ------------------------------------------------------------------------------
# Step 1: Configuration & Client Initialization
# ------------------------------------------------------------------------------

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
COLLECTION_NAME = "articles"

print("=" * 60)
print("1. Connecting to Milvus Vector Database...")
print("=" * 60)

client = None
try:
    # Try connecting to the Dockerized Milvus standalone server
    server_uri = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
    client = MilvusClient(uri=server_uri, timeout=3.0)
    # Ping collections to confirm active connection
    client.list_collections()
    print(f"✓ Connected to Milvus Standalone Server at {server_uri}")
except Exception as e:
    print(f"! Notice: Could not connect to Milvus server ({e}).")
    print("  Falling back to embedded Milvus Lite (local SQLite-like engine)...")
    local_db_path = str(Path(__file__).resolve().parent / "milvus_demo.db")
    client = MilvusClient(uri=local_db_path)
    print(f"✓ Initialized embedded Milvus Lite at: {local_db_path}")

# ------------------------------------------------------------------------------
# Step 2: Collection Management & Data Ingestion
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"2. Preparing Collection '{COLLECTION_NAME}' & Ingesting Data...")
print("=" * 60)

# Load sample dataset
dataset_path = root_dir / "dataset" / "sample_embeddings.json"
with open(dataset_path, "r", encoding="utf-8") as f:
    data = json.load(f)

vector_dimension = data["dimension"]
documents_data = data["documents"]
sample_queries = data["sample_queries"]

# Drop existing collection for clean repeatable run
if client.has_collection(collection_name=COLLECTION_NAME):
    client.drop_collection(collection_name=COLLECTION_NAME)
    print(f"  Existing collection '{COLLECTION_NAME}' dropped.")

# Create collection with Cosine similarity metric
client.create_collection(
    collection_name=COLLECTION_NAME,
    dimension=vector_dimension,
    metric_type="COSINE",
    auto_id=False
)
print(f"✓ Created collection '{COLLECTION_NAME}' (dim={vector_dimension}, metric=COSINE).")

# Prepare records
records = [
    {
        "id": doc["id"],
        "vector": doc["vector"],
        "title": doc["title"],
        "category": doc["category"],
        "tags": doc["tags"],
        "content": doc["content"]
    }
    for doc in documents_data
]

# Insert records
insert_result = client.insert(
    collection_name=COLLECTION_NAME,
    data=records
)
print(f"✓ Successfully inserted {insert_result.get('insert_count', len(records))} rows into '{COLLECTION_NAME}'.")

# ------------------------------------------------------------------------------
# Step 3: Vector Similarity Search (Top-K ANN)
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. Executing Vector Similarity Search (Unfiltered Top-3)...")
print("=" * 60)

tech_query = sample_queries["technology_query"]
query_vector = tech_query["vector"]

print(f"Query prompt : \"{tech_query['prompt']}\"")
print(f"Query vector : {query_vector[:4]}... (dim={len(query_vector)})")

search_results = client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    limit=3,
    output_fields=["title", "category", "tags", "content"]
)

print("\n--- Search Results ---")
for idx, hit in enumerate(search_results[0], start=1):
    entity = hit.get("entity", {})
    score = hit.get("distance", 0.0)  # In pymilvus with COSINE, distance represents cosine similarity
    print(f"Rank {idx}: [ID: {hit.get('id')}] {entity.get('title')}")
    print(f"  Category   : {entity.get('category')} (tags: {entity.get('tags')})")
    print(f"  Score      : {score:.4f} (Cosine Similarity)")
    print(f"  Content    : {entity.get('content', '')[:80]}...")

# ------------------------------------------------------------------------------
# Step 4: Filtered Vector Similarity Search
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. Executing Filtered Vector Similarity Search (category == 'culinary')...")
print("=" * 60)

culinary_query = sample_queries["culinary_query"]
print(f"Query prompt : \"{culinary_query['prompt']}\"")
print(f"Filter clause: category == 'culinary'")

filtered_results = client.search(
    collection_name=COLLECTION_NAME,
    data=[culinary_query["vector"]],
    limit=2,
    filter='category == "culinary"',
    output_fields=["title", "category", "tags", "content"]
)

print("\n--- Filtered Search Results ---")
for idx, hit in enumerate(filtered_results[0], start=1):
    entity = hit.get("entity", {})
    score = hit.get("distance", 0.0)
    print(f"Rank {idx}: [ID: {hit.get('id')}] {entity.get('title')}")
    print(f"  Category   : {entity.get('category')}")
    print(f"  Score      : {score:.4f} (Cosine Similarity)")
    print(f"  Content    : {entity.get('content', '')[:80]}...")

print("\n" + "=" * 60)
print("Milvus demonstration completed successfully!")
print("=" * 60)
