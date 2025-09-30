# Company Chatbot Backend (FastAPI)

Ocean Professional themed backend providing:
- Authentication (signup, token, me)
- Chat with simple Agentic reasoning using RAG contexts
- RAG document ingest and FAISS-like semantic search (with in-memory fallback)
- PostgreSQL integration (SQLAlchemy) ready for persistence

Run locally:
1) Create .env from .env.example and set variables (SECRET_KEY, POSTGRES_URL, etc.)
2) Install dependencies: pip install -r requirements.txt
3) Start: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
4) Docs: open http://localhost:8000/docs

OpenAPI is generated to interfaces/openapi.json via:
python -m src.api.generate_openapi

Notes:
- FAISS is optional; the app falls back to an in-memory cosine similarity.
- DB models are provided but routes currently use in-memory storage for simplicity.
- Migrations are not included; integrate Alembic for production.
