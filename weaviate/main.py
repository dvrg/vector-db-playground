"""
Weaviate Vector Database Implementation
Demonstrating:
  1. Connection / Client Initialization (Weaviate v4 Python Client with gRPC + REST)
  2. Collection Creation & Dynamic Batch Ingestion (Vectors + Typed Properties)
  3. Vector Similarity Search (near_vector Top-K with Cosine Distance)
  4. Filtered Vector Similarity Search (Filter by Property Condition)
"""

import json
import os
import sys
from pathlib import Path

# Load environment variables if python-dotenv is installed
root_dir = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(root_dir / ".env")
except ImportError:
    pass

import weaviate
from weaviate.classes.config import Property, DataType, Configure, VectorDistances
from weaviate.classes.query import Filter, MetadataQuery

# ------------------------------------------------------------------------------
# Step 1: Configuration & Connection
# ------------------------------------------------------------------------------

WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
COLLECTION_NAME = "Article"

print("=" * 60)
print("1. Connecting to Weaviate Vector Database...")
print("=" * 60)

try:
    client = weaviate.connect_to_local(
        host=WEAVIATE_HOST,
        port=WEAVIATE_PORT,
        grpc_port=WEAVIATE_GRPC_PORT
    )
    if not client.is_ready():
        raise RuntimeError("Weaviate reported not ready state.")
    print(f"✓ Connected to Weaviate v4 at {WEAVIATE_HOST} (HTTP:{WEAVIATE_PORT}, gRPC:{WEAVIATE_GRPC_PORT})")
except Exception as e:
    print(f"✗ Failed to connect to Weaviate: {e}")
    print("\nTip: Start the Weaviate container via Docker Compose:")
    print("     docker compose up -d weaviate\n")
    sys.exit(1)

try:
    # ------------------------------------------------------------------------------
    # Step 2: Collection Management & Dynamic Batch Ingestion
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"2. Preparing Collection '{COLLECTION_NAME}' & Batch Ingestion...")
    print("=" * 60)

    # Delete collection if already exists
    if client.collections.exists(COLLECTION_NAME):
        client.collections.delete(COLLECTION_NAME)
        print(f"  Existing collection '{COLLECTION_NAME}' deleted.")

    # Create collection with typed properties and HNSW Cosine configuration
    articles = client.collections.create(
        name=COLLECTION_NAME,
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE
        ),
        properties=[
            Property(name="doc_id", data_type=DataType.INT),
            Property(name="title", data_type=DataType.TEXT),
            Property(name="category", data_type=DataType.TEXT),
            Property(name="tags", data_type=DataType.TEXT_ARRAY),
            Property(name="content", data_type=DataType.TEXT),
        ]
    )
    print(f"✓ Created collection '{COLLECTION_NAME}' with HNSW (Cosine metric).")

    # Load sample dataset
    dataset_path = root_dir / "dataset" / "sample_embeddings.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents_data = data["documents"]
    sample_queries = data["sample_queries"]

    # Ingest data using dynamic batching
    with articles.batch.dynamic() as batch:
        for doc in documents_data:
            batch.add_object(
                properties={
                    "doc_id": doc["id"],
                    "title": doc["title"],
                    "category": doc["category"],
                    "tags": doc["tags"],
                    "content": doc["content"],
                },
                vector=doc["vector"]
            )

    # Check for any batch errors
    failed_objects = articles.batch.failed_objects
    if failed_objects:
        print(f"! Warning: {len(failed_objects)} objects failed insertion: {failed_objects[:2]}")
    else:
        print(f"✓ Successfully batch ingested {len(documents_data)} objects into '{COLLECTION_NAME}'.")

    # ------------------------------------------------------------------------------
    # Step 3: Vector Similarity Search (near_vector Top-K)
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("3. Executing Vector Similarity Search (Unfiltered Top-3)...")
    print("=" * 60)

    tech_query = sample_queries["technology_query"]
    query_vector = tech_query["vector"]

    print(f"Query prompt : \"{tech_query['prompt']}\"")
    print(f"Query vector : {query_vector[:4]}... (dim={len(query_vector)})")

    response = articles.query.near_vector(
        near_vector=query_vector,
        limit=3,
        return_metadata=MetadataQuery(distance=True, certainty=True)
    )

    print("\n--- Search Results ---")
    for idx, obj in enumerate(response.objects, start=1):
        props = obj.properties
        distance = obj.metadata.distance
        certainty = obj.metadata.certainty
        similarity = 1.0 - distance if distance is not None else (certainty or 0.0)
        print(f"Rank {idx}: [ID: {props.get('doc_id')}] {props.get('title')}")
        print(f"  Category   : {props.get('category')} (tags: {props.get('tags')})")
        print(f"  Distance   : {distance:.4f} | Cosine Similarity: {similarity:.4f}")
        print(f"  Content    : {props.get('content', '')[:80]}...")

    # ------------------------------------------------------------------------------
    # Step 4: Filtered Vector Similarity Search
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("4. Executing Filtered Vector Similarity Search (category == 'culinary')...")
    print("=" * 60)

    culinary_query = sample_queries["culinary_query"]
    print(f"Query prompt : \"{culinary_query['prompt']}\"")
    print(f"Filter clause: category == 'culinary'")

    filtered_response = articles.query.near_vector(
        near_vector=culinary_query["vector"],
        limit=2,
        filters=Filter.by_property("category").equal("culinary"),
        return_metadata=MetadataQuery(distance=True, certainty=True)
    )

    print("\n--- Filtered Search Results ---")
    for idx, obj in enumerate(filtered_response.objects, start=1):
        props = obj.properties
        distance = obj.metadata.distance
        similarity = 1.0 - distance if distance is not None else 0.0
        print(f"Rank {idx}: [ID: {props.get('doc_id')}] {props.get('title')}")
        print(f"  Category   : {props.get('category')}")
        print(f"  Distance   : {distance:.4f} | Cosine Similarity: {similarity:.4f}")
        print(f"  Content    : {props.get('content', '')[:80]}...")

    print("\n" + "=" * 60)
    print("Weaviate demonstration completed successfully!")
    print("=" * 60)

finally:
    client.close()
    print("✓ Weaviate client connection closed cleanly.")
