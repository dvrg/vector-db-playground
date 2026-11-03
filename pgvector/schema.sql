-- ==============================================================================
-- pgvector Schema Definition
-- Creates vector extension, articles table, and HNSW index
-- ==============================================================================

-- 1. Enable the vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Drop existing table for clean setup
DROP TABLE IF EXISTS articles;

-- 3. Create articles table with vector column
CREATE TABLE articles (
    id INT PRIMARY KEY,
    title TEXT NOT NULL,
    category VARCHAR(50) NOT NULL,
    tags TEXT[] NOT NULL,
    content TEXT NOT NULL,
    embedding vector(8) NOT NULL
);

-- 4. Create an HNSW index using cosine distance (vector_cosine_ops)
-- Other operators:
--   vector_l2_ops     -> Euclidean distance (<->)
--   vector_ip_ops     -> Inner product (<#>)
--   vector_cosine_ops -> Cosine distance (<=>)
CREATE INDEX IF NOT EXISTS articles_embedding_hnsw_idx
ON articles USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
