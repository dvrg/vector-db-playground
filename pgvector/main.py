"""
PostgreSQL + pgvector Implementation
Demonstrating:
  1. Connection / Client Initialization (Psycopg2 + pgvector extension)
  2. Table Schema Execution & Data Ingestion (SQL INSERT with Vector Types)
  3. Vector Similarity Search (Top-K with Cosine Distance Operator <=>)
  4. Filtered Vector Similarity Search (SQL WHERE Clause Filter)
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

import psycopg2
from pgvector.psycopg2 import register_vector

# ------------------------------------------------------------------------------
# Step 1: Configuration & Connection
# ------------------------------------------------------------------------------

PG_HOST = os.getenv("PGVECTOR_HOST", "localhost")
PG_PORT = int(os.getenv("PGVECTOR_PORT", "5432"))
PG_USER = os.getenv("PGVECTOR_USER", "postgres")
PG_PASSWORD = os.getenv("PGVECTOR_PASSWORD", "postgres")
PG_DB = os.getenv("PGVECTOR_DB", "vectordb")

print("=" * 60)
print("1. Connecting to PostgreSQL with pgvector...")
print("=" * 60)

try:
    conn = psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASSWORD,
        dbname=PG_DB,
        connect_timeout=3
    )
    conn.autocommit = True
    # Register pgvector type handler with psycopg2
    register_vector(conn)
    cursor = conn.cursor()
    print(f"✓ Connected to PostgreSQL at {PG_HOST}:{PG_PORT}/{PG_DB}")
except Exception as e:
    print(f"✗ Failed to connect to PostgreSQL: {e}")
    print("\nTip: Start the pgvector container via Docker Compose:")
    print("     docker compose up -d pgvector\n")
    sys.exit(1)

# ------------------------------------------------------------------------------
# Step 2: Schema Management & Data Ingestion
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. Applying Schema & Ingesting Dataset...")
print("=" * 60)

schema_path = Path(__file__).resolve().parent / "schema.sql"
with open(schema_path, "r", encoding="utf-8") as f:
    schema_sql = f.read()

# Execute schema script
cursor.execute(schema_sql)
print("✓ Executed schema.sql (Extension verified, table & HNSW index ready).")

# Load sample dataset
dataset_path = root_dir / "dataset" / "sample_embeddings.json"
with open(dataset_path, "r", encoding="utf-8") as f:
    data = json.load(f)

documents_data = data["documents"]
sample_queries = data["sample_queries"]

insert_sql = """
INSERT INTO articles (id, title, category, tags, content, embedding)
VALUES (%s, %s, %s, %s, %s, %s);
"""

for doc in documents_data:
    cursor.execute(
        insert_sql,
        (
            doc["id"],
            doc["title"],
            doc["category"],
            doc["tags"],
            doc["content"],
            doc["vector"]
        )
    )

print(f"✓ Successfully inserted {len(documents_data)} rows into 'articles'.")

# ------------------------------------------------------------------------------
# Step 3: Vector Similarity Search (Top-K with Cosine Distance)
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. Executing Vector Similarity Search (Unfiltered Top-3)...")
print("=" * 60)

tech_query = sample_queries["technology_query"]
query_vector = tech_query["vector"]

print(f"Query prompt : \"{tech_query['prompt']}\"")
print(f"Query vector : {query_vector[:4]}... (dim={len(query_vector)})")

# In pgvector:
# '<=>' computes Cosine Distance (0 = identical, 2 = opposite)
# Cosine Similarity = 1 - (embedding <=> query_vector)
search_sql = """
SELECT
    id,
    title,
    category,
    tags,
    content,
    embedding <=> %s AS cosine_distance,
    1.0 - (embedding <=> %s) AS cosine_similarity
FROM articles
ORDER BY embedding <=> %s ASC
LIMIT 3;
"""

cursor.execute(search_sql, (query_vector, query_vector, query_vector))
rows = cursor.fetchall()

print("\n--- Search Results ---")
for idx, (doc_id, title, category, tags, content, dist, sim) in enumerate(rows, start=1):
    print(f"Rank {idx}: [ID: {doc_id}] {title}")
    print(f"  Category   : {category} (tags: {tags})")
    print(f"  Distance   : {dist:.4f} | Cosine Similarity: {sim:.4f}")
    print(f"  Content    : {content[:80]}...")

# ------------------------------------------------------------------------------
# Step 4: Filtered Vector Similarity Search
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. Executing Filtered Vector Similarity Search (category = 'culinary')...")
print("=" * 60)

culinary_query = sample_queries["culinary_query"]
print(f"Query prompt : \"{culinary_query['prompt']}\"")
print(f"Filter clause: WHERE category = 'culinary'")

filtered_sql = """
SELECT
    id,
    title,
    category,
    tags,
    content,
    embedding <=> %s AS cosine_distance,
    1.0 - (embedding <=> %s) AS cosine_similarity
FROM articles
WHERE category = 'culinary'
ORDER BY embedding <=> %s ASC
LIMIT 2;
"""

cursor.execute(filtered_sql, (culinary_query["vector"], culinary_query["vector"], culinary_query["vector"]))
filtered_rows = cursor.fetchall()

print("\n--- Filtered Search Results ---")
for idx, (doc_id, title, category, tags, content, dist, sim) in enumerate(filtered_rows, start=1):
    print(f"Rank {idx}: [ID: {doc_id}] {title}")
    print(f"  Category   : {category}")
    print(f"  Distance   : {dist:.4f} | Cosine Similarity: {sim:.4f}")
    print(f"  Content    : {content[:80]}...")

# Close connection
cursor.close()
conn.close()

print("\n" + "=" * 60)
print("pgvector demonstration completed successfully!")
print("=" * 60)
