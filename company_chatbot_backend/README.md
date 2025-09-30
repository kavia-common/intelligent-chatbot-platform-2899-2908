# Company Chatbot Backend (FastAPI)

Ocean Professional themed backend providing:
- Authentication (signup, token, me) using JWT (HS256). Requires SECRET_KEY.
- Chat with Agentic reasoning using RAG contexts and OpenAI LLM (fallback to heuristic if OpenAI not configured)
- RAG document ingest and FAISS-like semantic search (with in-memory fallback)
- PostgreSQL integration (SQLAlchemy) with persistence for users, chat history, and documents

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
- Migrations are not included; integrate Alembic for production.
- Ensure your database has been initialized using assets/company_chatbot_db_schema.sql.txt.
  The app expects tables: users, chat_messages, documents.

Auth:
- JWT tokens are created and verified using SECRET_KEY. For development without PyJWT installed, a dev token format is used automatically.
- Never commit your SECRET_KEY. Set it via environment.

OpenAI integration:
- The backend detects OPENAI_API_KEY from environment to call OpenAI Chat Completions for RAG+Agentic responses.
- If OPENAI_API_KEY is missing or invalid, the /chat/messages endpoint gracefully falls back to a heuristic answer synthesized from retrieved contexts.
- Configure the following environment variables in your .env:
  - OPENAI_API_KEY: Your OpenAI key. Required for LLM-powered answers.
  - OPENAI_MODEL: Default model (e.g., gpt-4o-mini). Optional.
  - OPENAI_BASE_URL: Base URL for the API (default https://api.openai.com/v1). Optional.
- Security: The key is never logged or hardcoded and is read only from the environment.

Host and CORS configuration:
- To avoid "Invalid Host header" errors in preview or non-production environments, set:
  ALLOWED_HOSTS=["*"] in your .env (see .env.example). When '*' is present or ALLOWED_HOSTS is unset/empty, the app disables TrustedHostMiddleware.
- For production, set explicit hosts, e.g.:
  ALLOWED_HOSTS=["api.example.com","localhost","127.0.0.1"]
- The app logs the configured ALLOWED_HOSTS at startup and logs the Host header for incoming requests at debug level to aid diagnostics.
- CORS defaults to allow all origins for previews; tighten for production.
