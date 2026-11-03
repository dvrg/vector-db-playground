"""
Pinecone Vector Database Implementation
Demonstrating:
  1. Connection / Client Initialization (Cloud-Managed Pinecone SDK v5+)
  2. Serverless Index Creation & Vector Upsert (Vectors + Metadata)
  3. Vector Similarity Search (Top-K ANN with Cosine Metric)
  4. Filtered Vector Similarity Search (Metadata Filter Dictionary)

Note: Pinecone is a fully cloud-managed vector database. If no valid PINECONE_API_KEY
is detected in your .env file, this script will run a high-fidelity mock simulation
to demonstrate the exact client API and data flow without failing.
"""

import json
import os
import time
from pathlib import Path

# Load environment variables if python-dotenv is installed
root_dir = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(root_dir / ".env")
except ImportError:
    pass

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "sample-articles")
CLOUD_PROVIDER = os.getenv("PINECONE_CLOUD", "aws")
CLOUD_REGION = os.getenv("PINECONE_REGION", "us-east-1")

print("=" * 60)
print("1. Connecting to Pinecone Cloud Vector Database...")
print("=" * 60)

is_live_mode = False
pc = None
index = None

if PINECONE_API_KEY and not PINECONE_API_KEY.startswith("your_pinecone_api_key"):
    try:
        from pinecone import Pinecone, ServerlessSpec
        pc = Pinecone(api_key=PINECONE_API_KEY)
        print(f"✓ Connected to Pinecone Cloud API with user API key.")
        is_live_mode = True
    except Exception as e:
        print(f"! Failed to initialize live Pinecone client: {e}")
        print("  Switching to demonstration simulation mode...")
else:
    print("! Notice: PINECONE_API_KEY is not configured or is using placeholder.")
    print("  To run against live Pinecone Cloud:")
    print("    1. Sign up for free at https://app.pinecone.io")
    print("    2. Add PINECONE_API_KEY to your .env file")
    print("  Continuing in demonstration simulation mode...\n")

# Load sample dataset
dataset_path = root_dir / "dataset" / "sample_embeddings.json"
with open(dataset_path, "r", encoding="utf-8") as f:
    data = json.load(f)

vector_dimension = data["dimension"]
documents_data = data["documents"]
sample_queries = data["sample_queries"]

# ------------------------------------------------------------------------------
# Step 2: Index Management & Vector Upsert
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"2. Managing Index '{INDEX_NAME}' & Upserting Vectors...")
print("=" * 60)

if is_live_mode:
    from pinecone import ServerlessSpec
    # Check if index exists; if so, delete for fresh playground run
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    if INDEX_NAME in existing_indexes:
        print(f"  Existing index '{INDEX_NAME}' found. Deleting for fresh test...")
        pc.delete_index(INDEX_NAME)
        time.sleep(2)

    print(f"  Creating serverless index '{INDEX_NAME}' (dim={vector_dimension}, metric=cosine)...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=vector_dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=CLOUD_PROVIDER,
            region=CLOUD_REGION
        )
    )
    # Wait for index readiness
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(1)

    index = pc.Index(INDEX_NAME)
    print(f"✓ Index '{INDEX_NAME}' is ready.")

    # Prepare vectors for upsert
    vectors_to_upsert = [
        {
            "id": str(doc["id"]),
            "values": doc["vector"],
            "metadata": {
                "title": doc["title"],
                "category": doc["category"],
                "tags": doc["tags"],
                "content": doc["content"]
            }
        }
        for doc in documents_data
    ]

    index.upsert(vectors=vectors_to_upsert)
    print(f"✓ Upserted {len(vectors_to_upsert)} vectors into Pinecone index '{INDEX_NAME}'.")
    # Brief pause for eventual consistency in cloud index
    time.sleep(2)
else:
    print(f"✓ [Simulated] Index '{INDEX_NAME}' created (metric='cosine', dim={vector_dimension}, spec={CLOUD_PROVIDER}/{CLOUD_REGION}).")
    print(f"✓ [Simulated] Upserted {len(documents_data)} vectors with metadata payloads.")

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

if is_live_mode:
    query_response = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True
    )
    matches = query_response.get("matches", [])
else:
    # Simulated cosine similarity calculation for demonstration
    def cosine_similarity(v1, v2):
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = sum(a * a for a in v1) ** 0.5
        norm2 = sum(b * b for b in v2) ** 0.5
        return dot / (norm1 * norm2)

    scored = []
    for doc in documents_data:
        sim = cosine_similarity(query_vector, doc["vector"])
        scored.append({
            "id": str(doc["id"]),
            "score": sim,
            "metadata": {
                "title": doc["title"],
                "category": doc["category"],
                "tags": doc["tags"],
                "content": doc["content"]
            }
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    matches = scored[:3]

print("\n--- Search Results ---")
for idx, match in enumerate(matches, start=1):
    meta = match.get("metadata", {})
    score = match.get("score", 0.0)
    print(f"Rank {idx}: [ID: {match.get('id')}] {meta.get('title')}")
    print(f"  Category   : {meta.get('category')} (tags: {meta.get('tags')})")
    print(f"  Score      : {score:.4f} (Cosine Similarity)")
    print(f"  Content    : {meta.get('content', '')[:80]}...")

# ------------------------------------------------------------------------------
# Step 4: Filtered Vector Similarity Search
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. Executing Filtered Vector Similarity Search (category == 'culinary')...")
print("=" * 60)

culinary_query = sample_queries["culinary_query"]
print(f"Query prompt : \"{culinary_query['prompt']}\"")
print(f"Filter clause: {{\"category\": {{\"$eq\": \"culinary\"}}}}")

if is_live_mode:
    filtered_response = index.query(
        vector=culinary_query["vector"],
        top_k=2,
        include_metadata=True,
        filter={"category": {"$eq": "culinary"}}
    )
    filtered_matches = filtered_response.get("matches", [])
else:
    filtered_scored = []
    for doc in documents_data:
        if doc["category"] == "culinary":
            sim = cosine_similarity(culinary_query["vector"], doc["vector"])
            filtered_scored.append({
                "id": str(doc["id"]),
                "score": sim,
                "metadata": {
                    "title": doc["title"],
                    "category": doc["category"],
                    "tags": doc["tags"],
                    "content": doc["content"]
                }
            })
    filtered_scored.sort(key=lambda x: x["score"], reverse=True)
    filtered_matches = filtered_scored[:2]

print("\n--- Filtered Search Results ---")
for idx, match in enumerate(filtered_matches, start=1):
    meta = match.get("metadata", {})
    score = match.get("score", 0.0)
    print(f"Rank {idx}: [ID: {match.get('id')}] {meta.get('title')}")
    print(f"  Category   : {meta.get('category')}")
    print(f"  Score      : {score:.4f} (Cosine Similarity)")
    print(f"  Content    : {meta.get('content', '')[:80]}...")

print("\n" + "=" * 60)
print("Pinecone demonstration completed successfully!")
print("=" * 60)
