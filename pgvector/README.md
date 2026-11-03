# PostgreSQL + pgvector Implementation

[pgvector](https://github.com/pgvector/pgvector) is an open-source vector similarity search extension for PostgreSQL. It allows storing vector embeddings alongside existing relational tables, leveraging full ACID guarantees, SQL joins, and relational integrity.

---

## 1. Overview & Architecture

- **License**: PostgreSQL License (Open Source)
- **Indexing Engines**:
  - **HNSW (Hierarchical Navigable Small World)**: Fast query times with high recall, higher build time and memory.
  - **IVFFlat (Inverted File Flat)**: Faster index build time and less memory, requires data clustering before index creation.
- **Distance Operators**:
  - `<->` : Euclidean distance ($L_2$)
  - `<#>` : Negative inner product (dot product)
  - `<=>` : Cosine distance ($1 - \text{cosine similarity}$)
- **Key Characteristics**:
  - **Zero New Infrastructure**: Keep vectors directly in your existing PostgreSQL database.
  - **Complex SQL Queries**: Easily join vectors with relational tables, user permissions, transactions, and foreign keys.

---

## 2. Prerequisites & Setup

### Running with Docker

From the project root:
```bash
docker compose up -d pgvector
```

PostgreSQL will be exposed on port `5432`:
- **User**: `postgres`
- **Password**: `postgres`
- **Database**: `vectordb`

---

## 3. Installation & Running

1. Activate your virtual environment and install requirements:
   ```bash
   pip install -r pgvector/requirements.txt
   ```

2. Run the demonstration script:
   ```bash
   python pgvector/main.py
   ```

---

## 4. Code Flow Explained

1. **Connection**: Connects via `psycopg2` and calls `register_vector(conn)` to enable native Python list to PostgreSQL `vector` type mapping.
2. **Schema Migration**: Executes `schema.sql` to install `CREATE EXTENSION IF NOT EXISTS vector;`, create table `articles`, and build an HNSW index on `embedding`.
3. **Data Ingestion**: Inserts records using standard parameterized SQL `INSERT INTO articles VALUES (%s, ...)`.
4. **Vector Search**: Executes `ORDER BY embedding <=> %s ASC LIMIT 3` to perform nearest-neighbor search with Cosine distance.
5. **Metadata Filtering**: Applies standard SQL `WHERE category = 'culinary'` alongside the vector order clause.
