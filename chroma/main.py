"""
Chroma Vector Database Implementation
Demonstrating:
  1. Connection / Client Initialization (HTTP Server or Embedded Fallback)
  2. Collection Creation & Data Ingestion (Vectors + Documents + Metadata)
  3. Vector Similarity Search (Top-K ANN with Cosine Distance)
  4. Filtered Vector Similarity Search (Metadata Filtering)
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

import chromadb
from chromadb.config import Settings

# ------------------------------------------------------------------------------
# Step 1: Configuration & Client Initialization
# ------------------------------------------------------------------------------

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = "articles"

print("=" * 60)
print("1. Connecting to Chroma Vector Database...")
print("=" * 60)

try:
    # Try connecting to the Dockerized Chroma HTTP server first
    client = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        settings=Settings(anonymized_telemetry=False)
    )
    client.heartbeat()
    print(f"✓ Connected to Chroma HTTP Server at {CHROMA_HOST}:{CHROMA_PORT}")
except Exception as e:
    print(f"! Notice: Could not connect to Chroma HTTP Server ({e}).")
    print("  Falling back to embedded local PersistentClient (no Docker required)...")
    local_data_path = str(Path(__file__).resolve().parent / "chroma_data")
    client = chromadb.PersistentClient(
        path=local_data_path,
        settings=Settings(anonymized_telemetry=False)
    )
    print(f"✓ Initialized embedded Chroma PersistentClient at: {local_data_path}")

# ------------------------------------------------------------------------------
# Step 2: Collection Management & Data Ingestion
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"2. Preparing Collection '{COLLECTION_NAME}' & Inserting Data...")
print("=" * 60)

# Reset collection for repeatable playground runs
try:
    client.delete_collection(name=COLLECTION_NAME)
    print(f"  Existing collection '{COLLECTION_NAME}' deleted.")
except Exception:
    pass

# Create collection with Cosine distance metric
collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)
print(f"✓ Created collection '{COLLECTION_NAME}' with HNSW space 'cosine'.")

# Load sample dataset
dataset_path = root_dir / "dataset" / "sample_embeddings.json"
with open(dataset_path, "r", encoding="utf-8") as f:
    data = json.load(f)

documents_data = data["documents"]
sample_queries = data["sample_queries"]

# Prepare batches for Chroma ingestion
ids = [str(doc["id"]) for doc in documents_data]
embeddings = [doc["vector"] for doc in documents_data]
documents = [doc["content"] for doc in documents_data]
metadatas = [
    {
        "title": doc["title"],
        "category": doc["category"],
        "tags": ",".join(doc["tags"])
    }
    for doc in documents_data
]

collection.add(
    ids=ids,
    embeddings=embeddings,
    metadatas=metadatas,
    documents=documents
)
print(f"✓ Successfully ingested {len(ids)} documents into '{COLLECTION_NAME}'.")

# ------------------------------------------------------------------------------
# Step 3: Vector Similarity Search (Top-K ANN)
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. Executing Vector Similarity Search (Unfiltered Top-3)...")
print("=" * 60)

query_info = sample_queries["technology_query"]
query_vector = query_info["vector"]
query_text = query_info["prompt"]

print(f"Query prompt : \"{query_text}\"")
print(f"Query vector : {query_vector[:4]}... (dim={len(query_vector)})")

results = collection.query(
    query_embeddings=[query_vector],
    n_results=3,
    include=["metadatas", "documents", "distances"]
)

print("\n--- Search Results ---")
for idx, (doc_id, metadata, text, distance) in enumerate(
    zip(
        results["ids"][0],
        results["metadatas"][0],
        results["documents"][0],
        results["distances"][0]
    ),
    start=1
):
    # Cosine distance in Chroma: similarity = 1 - distance
    similarity_score = 1.0 - distance
    print(f"Rank {idx}: [ID: {doc_id}] {metadata['title']}")
    print(f"  Category   : {metadata['category']} (tags: {metadata['tags']})")
    print(f"  Distance   : {distance:.4f} | Cosine Similarity: {similarity_score:.4f}")
    print(f"  Content    : {text[:80]}...")

# ------------------------------------------------------------------------------
# Step 4: Filtered Vector Similarity Search
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. Executing Filtered Vector Similarity Search (category == 'culinary')...")
print("=" * 60)

culinary_query = sample_queries["culinary_query"]
print(f"Query prompt : \"{culinary_query['prompt']}\"")
print(f"Filter clause: category == 'culinary'")

filtered_results = collection.query(
    query_embeddings=[culinary_query["vector"]],
    n_results=2,
    where={"category": "culinary"},
    include=["metadatas", "documents", "distances"]
)

print("\n--- Filtered Search Results ---")
for idx, (doc_id, metadata, text, distance) in enumerate(
    zip(
        filtered_results["ids"][0],
        filtered_results["metadatas"][0],
        filtered_results["documents"][0],
        filtered_results["distances"][0]
    ),
    start=1
):
    similarity_score = 1.0 - distance
    print(f"Rank {idx}: [ID: {doc_id}] {metadata['title']}")
    print(f"  Category   : {metadata['category']}")
    print(f"  Distance   : {distance:.4f} | Cosine Similarity: {similarity_score:.4f}")
    print(f"  Content    : {text[:80]}...")

print("\n" + "=" * 60)
print("Chroma demonstration completed successfully!")
print("=" * 60)
