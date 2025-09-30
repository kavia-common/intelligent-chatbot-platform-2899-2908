# Database Schema Setup for company_chatbot_db

This folder contains SQL text to initialize the PostgreSQL schema for:
- users (authentication)
- chat_messages (chat history)
- documents (RAG source documents with optional vector embeddings via pgvector)

Files:
- company_chatbot_db_schema.sql.txt — Copy/paste or run the contained statements against your Postgres instance.

Instructions:
1) Ensure your Postgres instance is running and you can connect to the target database (e.g., company_chatbot_db).
2) If you have a db_connection.txt with connection info, use it. Example psql:
   psql "$POSTGRES_URL"
   or
   psql "postgresql://user:password@host:5432/company_chatbot_db?sslmode=prefer"

3) Run the SQL statements in order. They are idempotent and safe to re-run.

Requirements/Notes:
- Extensions used:
  - pgcrypto (for gen_random_uuid)
  - citext (for case-insensitive emails)
  - vector (pgvector; optional: enables the vector embedding column and index)
- If pgvector is not installed the schema will fallback to using BYTEA for embeddings.

Backend Environment:
- Set POSTGRES_URL in the backend .env for SQLAlchemy (e.g., postgresql+psycopg://user:password@host:5432/company_chatbot_db)

Security:
- Store hashed_password as a salted hash (bcrypt/argon2 recommended). Do not store plain-text passwords.

Indexing:
- An HNSW or IVFFlat index is created for the embedding column when pgvector is present. If your version doesn’t support HNSW, the script falls back to IVFFlat or no vector index.

Dimension:
- The embedding vector is set to vector(384) to match the default embedding dimension in the backend. Adjust if needed.
