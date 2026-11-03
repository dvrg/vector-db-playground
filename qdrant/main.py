"""
Qdrant Vector Database Implementation
Demonstrating:
  1. Connection / Client Initialization (Docker Server or In-Memory Fallback)
  2. Collection Creation & Point Ingestion (Vectors + Payload Metadata)
  3. Vector Similarity Search (Top-K ANN with Cosine Distance)
  4. Filtered Vector Similarity Search (Payload Filtering via Filter & FieldCondition)
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

from qdrant_client import QdrantClient
from qdrant_client.http import models

# ------------------------------------------------------------------------------
# Step 1: Configuration & Client Initialization
# ------------------------------------------------------------------------------

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION_NAME = "articles"

print("=" * 60)
print("1. Connecting to Qdrant Vector Database...")
print("=" * 60)

try:
    # Try connecting to the Dockerized Qdrant instance
    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        api_key=QDRANT_API_KEY,
        timeout=3.0
    )
    # Check server status
    collections_info = client.get_collections()
    print(f"✓ Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
except Exception as e:
    print(f"! Notice: Could not connect to Qdrant server ({e}).")
    print("  Falling back to embedded in-memory Qdrant client (:memory:)...")
    client = QdrantClient(":memory:")
    print("✓ Initialized embedded in-memory Qdrant instance.")

# ------------------------------------------------------------------------------
# Step 2: Collection Management & Data Ingestion
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"2. Preparing Collection '{COLLECTION_NAME}' & Inserting Points...")
print("=" * 60)

# Load sample dataset
dataset_path = root_dir / "dataset" / "sample_embeddings.json"
with open(dataset_path, "r", encoding="utf-8") as f:
    data = json.load(f)

vector_dimension = data["dimension"]
documents_data = data["documents"]
sample_queries = data["sample_queries"]

# Recreate collection (drops if exists, creates fresh)
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=vector_dimension,
        distance=models.Distance.COSINE
    )
)
print(f"✓ Recreated collection '{COLLECTION_NAME}' (dim={vector_dimension}, metric=COSINE).")

# Prepare points with payload
points = [
    models.PointStruct(
        id=doc["id"],
        vector=doc["vector"],
        payload={
            "title": doc["title"],
            "category": doc["category"],
            "tags": doc["tags"],
            "content": doc["content"]
        }
    )
    for doc in documents_data
]

# Upsert points
client.upsert(
    collection_name=COLLECTION_NAME,
    points=points,
    wait=True
)
print(f"✓ Successfully upserted {len(points)} points into '{COLLECTION_NAME}'.")

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
    query_vector=query_vector,
    limit=3,
    with_payload=True
)

print("\n--- Search Results ---")
for idx, hit in enumerate(search_results, start=1):
    payload = hit.payload
    print(f"Rank {idx}: [ID: {hit.id}] {payload['title']}")
    print(f"  Category   : {payload['category']} (tags: {payload['tags']})")
    print(f"  Score      : {hit.score:.4f} (Cosine Similarity)")
    print(f"  Content    : {payload['content'][:80]}...")

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
    query_vector=culinary_query["vector"],
    limit=2,
    query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="category",
                match=models.MatchValue(value="culinary")
            )
        ]
    ),
    with_payload=True
)

print("\n--- Filtered Search Results ---")
for idx, hit in enumerate(filtered_results, start=1):
    payload = hit.payload
    print(f"Rank {idx}: [ID: {hit.id}] {payload['title']}")
    print(f"  Category   : {payload['category']}")
    print(f"  Score      : {hit.score:.4f} (Cosine Similarity)")
    print(f"  Content    : {payload['content'][:80]}...")

print("\n" + "=" * 60)
print("Qdrant demonstration completed successfully!")
print("=" * 60)
