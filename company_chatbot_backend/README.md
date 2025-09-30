# Company Chatbot Backend (FastAPI)

Ocean Professional themed backend providing:
- Authentication (signup, token, me)
- Chat with simple Agentic reasoning using RAG contexts
- RAG document ingest and FAISS-like semantic search (with in-memory fallback)
- PostgreSQL integration (SQLAlchemy) ready for persistence

Run locally:
1) Create .env from .env.example and set variables (SECRET_KEY, POSTGRES_URL, etc.)
2) Install dependencies: pip install -r requirements.txt
3) Start on port 3001:
   - python -m src.api.server
   - or: PORT=3001 uvicorn src.api.main:app --host 0.0.0.0 --port 3001
4) Docs: open http://localhost:3001/docs

OpenAPI is generated to interfaces/openapi.json via:
python -m src.api.generate_openAPI

Notes:
- FAISS is optional; the app falls back to an in-memory cosine similarity.
- DB models are provided but routes currently use in-memory storage for simplicity.
- Migrations are not included; integrate Alembic for production.

Host and CORS configuration:
- To avoid "Invalid Host header" errors in preview or non-production environments, set:
  ALLOWED_HOSTS=["*"] in your .env (see .env.example). When '*' is present or ALLOWED_HOSTS is unset/empty, the app disables TrustedHostMiddleware.
- For production, set explicit hosts, e.g.:
  ALLOWED_HOSTS=["api.example.com","localhost","127.0.0.1"]
- The app logs the configured ALLOWED_HOSTS at startup and logs the Host header for incoming requests at debug level to aid diagnostics.
- CORS defaults to allow all origins for previews; tighten for production.
