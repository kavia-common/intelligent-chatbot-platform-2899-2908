import uuid
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, status, Path, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi import APIRouter
from pydantic import BaseModel, Field, EmailStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# DB and repositories
from src.db.database import get_db
from src.db.repositories import (
    create_user as repo_create_user,
    get_user_by_email as repo_get_user_by_email,
    add_chat_message as repo_add_chat_message,
    ingest_documents as repo_ingest_documents,
)
from sqlalchemy.orm import Session

# Auth helpers
from .auth_utils import create_jwt_token, decode_jwt_token, hash_password, verify_password

# RAG service (FAISS or fallback)
from src.services.rag_service import (
    deterministic_embed as embed_text,
    ensure_index as ensure_faiss_index,
    add_vector as faiss_add,
    search as faiss_search,
    serialize_embedding_f32,
)

# Theme meta (reuse from rag_core to keep consistency)
from .rag_core import THEME_META as CORE_THEME_META

# OpenAI integration
from src.services.openai_client import openai_chat_completion, ensure_openai_key

logger = logging.getLogger("uvicorn.error")


# PUBLIC_INTERFACE
class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Note: The environment variables must be provided by the user and placed in the .env file.
    """
    APP_NAME: str = "Company Chatbot Backend"
    APP_DESCRIPTION: str = "Ocean Professional themed API providing chat, authentication, and RAG+FAISS semantic retrieval."
    APP_VERSION: str = "0.1.0"

    # Database configuration - must be provided in environment
    POSTGRES_URL: str = Field(default="", description="SQLAlchemy style Postgres URL, e.g., postgresql+psycopg://user:pass@host:port/db")

    # Security
    SECRET_KEY: str = Field(default="", description="JWT secret key - set via environment")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24

    # CORS
    CORS_ALLOW_ORIGINS: List[str] = Field(
        default_factory=lambda: ["*"],
        description="Comma-separated origins allowed for CORS (default: *)."
    )

    # Hosts validation (Starlette TrustedHostMiddleware)
    ALLOWED_HOSTS: List[str] = Field(
        default_factory=lambda: [
            "localhost",
            "127.0.0.1",
            "0.0.0.0",
            "*",  # Accept preview domains in containerized environments
        ],
        description="List of allowed hosts for requests (TrustedHostMiddleware)."
    )

    # Embeddings/LLM (simulated here)
    EMBEDDING_DIM: int = 384

    # OpenAI integration (documented; values taken from environment)
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key (read from environment; required for LLM features)")
    OPENAI_MODEL: str = Field(default="gpt-4o-mini", description="Default OpenAI model to use (override via OPENAI_MODEL)")
    OPENAI_BASE_URL: str = Field(default="https://api.openai.com/v1", description="Base URL for OpenAI API (override via OPENAI_BASE_URL)")

    # Pydantic v2 settings configuration
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()

# Theme metadata for Ocean Professional style embedded within API responses
THEME_META = CORE_THEME_META

# Import lifespan after settings is defined to allow it to reference settings safely
from .lifespan import lifespan  # noqa: E402

app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    contact={
        "name": "Chatbot Platform",
        "url": "https://example.com",
    },
    license_info={"name": "Proprietary"},
    terms_of_service="https://example.com/terms",
    openapi_tags=[
        {"name": "health", "description": "Service health and metadata"},
        {"name": "auth", "description": "User authentication and token management"},
        {"name": "chat", "description": "Chat and conversation endpoints"},
        {"name": "rag", "description": "Retrieval Augmented Generation and semantic search"},
    ],
    lifespan=lifespan,
)

# Log ALLOWED_HOSTS at startup for diagnostics
logger.info(f"Startup: ALLOWED_HOSTS={settings.ALLOWED_HOSTS}")

# CORS (ensure only added once)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted Host handling:
allowed_hosts = [h.strip() for h in settings.ALLOWED_HOSTS if str(h).strip()]
if allowed_hosts and "*" not in allowed_hosts:
    logger.info(f"TrustedHostMiddleware enabled with allowed_hosts={allowed_hosts}")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)
else:
    logger.warning(
        "TrustedHostMiddleware disabled (ALLOWED_HOSTS unset/empty or includes '*'). "
        "This is suitable for development/preview. Configure explicit hosts for production."
    )

# Lightweight middleware to log incoming host header for diagnostics
@app.middleware("http")
async def log_request_host(request: Request, call_next):
    host = request.headers.get("host", "<none>")
    logger.debug(f"Incoming request host={host}")
    response = await call_next(request)
    return response


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


# PUBLIC_INTERFACE
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Resolve the current user from a JWT access token."""
    try:
        payload = decode_jwt_token(token)
        subject = payload.get("sub")
        if not subject:
            raise ValueError("Missing subject")
        # subject is email for simplicity
        from src.db.repositories import get_user_by_email as _get
        user = _get(db, subject)
        if not user:
            raise ValueError("User not found")
        return {"id": user.id, "email": user.email, "name": user.name, "created_at": user.created_at}
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


# --------------------------
# Pydantic API Schemas
# --------------------------
class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    time: datetime = Field(..., description="Server time")
    meta: Dict[str, Any] = Field(..., description="Ocean Professional style metadata")


class Token(BaseModel):
    access_token: str = Field(..., description="Bearer token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_at: datetime = Field(..., description="Expiration timestamp")
    meta: Dict[str, Any] = Field(..., description="Style metadata")


class UserSignup(BaseModel):
    email: EmailStr = Field(..., description="User email")
    password: str = Field(..., min_length=6, description="Password (min 6 chars)")
    name: Optional[str] = Field(None, description="Display name")


class User(BaseModel):
    id: str = Field(..., description="User ID")
    email: EmailStr
    name: Optional[str] = None
    created_at: datetime
    meta: Dict[str, Any] = Field(default_factory=lambda: THEME_META)


class ConversationCreate(BaseModel):
    title: Optional[str] = Field(None, description="Optional session title")


class Conversation(BaseModel):
    id: str
    title: Optional[str] = None
    created_at: datetime
    meta: Dict[str, Any] = Field(default_factory=lambda: THEME_META)


class MessageCreate(BaseModel):
    content: str = Field(..., description="User message content")
    session_id: Optional[str] = Field(None, description="Conversation/session ID")


class Message(BaseModel):
    id: str
    role: str
    content: str
    created_at: datetime
    session_id: str
    meta: Dict[str, Any] = Field(default_factory=lambda: THEME_META)


class RAGQuery(BaseModel):
    query: str = Field(..., description="Natural language query")
    top_k: int = Field(3, ge=1, le=10, description="Number of passages to retrieve")


class RAGResponse(BaseModel):
    query: str
    contexts: List[Dict[str, Any]]
    answer: str
    meta: Dict[str, Any] = Field(default_factory=lambda: THEME_META)


# --------------------------
# Routers
# --------------------------
health_router = APIRouter(tags=["health"])
auth_router = APIRouter(prefix="/auth", tags=["auth"])
chat_router = APIRouter(prefix="/chat", tags=["chat"])
rag_router = APIRouter(prefix="/rag", tags=["rag"])


@health_router.get("/", summary="Health Check", description="Returns service health and Ocean Professional metadata", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="ok", time=datetime.utcnow(), meta=THEME_META)


# AUTH
@auth_router.post("/signup", summary="Signup", description="Create a new user account", response_model=User)
def signup(user_in: UserSignup, db: Session = Depends(get_db)) -> User:
    existing = repo_get_user_by_email(db, user_in.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    pw_hash = hash_password(user_in.password)
    user = repo_create_user(db, email=user_in.email, password_hash=pw_hash, name=user_in.name)
    return User(id=user.id, email=user.email, name=user.name, created_at=user.created_at)


@auth_router.post("/token", summary="Token", description="Obtain access token via OAuth2PasswordRequestForm", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = repo_get_user_by_email(db, form_data.username)
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    token_info = create_jwt_token(subject=user.email, expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    return Token(access_token=token_info["access_token"], token_type="bearer", expires_at=token_info["expires_at"], meta=THEME_META)


@auth_router.get("/me", summary="Me", description="Get current user details", response_model=User)
def me(user: Dict[str, Any] = Depends(get_current_user)):
    return User(id=user["id"], email=user["email"], name=user.get("name"), created_at=user["created_at"])


# CHAT (session handling kept in-memory for now, while messages also saved in DB as history)
@chat_router.post("/conversations", summary="Create Conversation", description="Create a new conversation/session", response_model=Conversation)
def create_conversation(payload: ConversationCreate = Body(default_factory=ConversationCreate), user: Dict[str, Any] = Depends(get_current_user)):
    session_id = str(uuid.uuid4())
    now = datetime.utcnow()
    # in-memory bucket for this session
    # stored per-session to match OpenAPI without persisting sessions
    return Conversation(id=session_id, title=payload.title, created_at=now)


@chat_router.get("/conversations", summary="List Conversations", description="List conversations (in-memory demo)", response_model=List[Conversation])
def list_conversations(user: Dict[str, Any] = Depends(get_current_user)):
    # Conversations are not persisted; return empty list placeholder
    return []


@chat_router.get("/conversations/{session_id}/messages", summary="Get Messages", description="Get messages for a session", response_model=List[Message])
def get_messages(session_id: str = Path(..., description="Conversation/session ID"), user: Dict[str, Any] = Depends(get_current_user)):
    # Messages are not persisted per-session; since we store flat history in DB, return empty for a new session.
    return []


@chat_router.post("/messages", summary="Send Message", description="Send a message and receive assistant response with simple agentic reasoning over RAG contexts (uses OpenAI if API key configured, else falls back to heuristic).", response_model=List[Message])
def send_message(payload: MessageCreate, user: Dict[str, Any] = Depends(get_current_user), db: Session = Depends(get_db)):
    # Create transient session id if missing
    session_id = payload.session_id or str(uuid.uuid4())

    # Persist user message to DB history
    user_row = repo_add_chat_message(db, user_id=user["id"], message_text=payload.content)

    # RAG retrieval
    ensure_faiss_index(settings.EMBEDDING_DIM)
    q_vec = embed_text(payload.content, settings.EMBEDDING_DIM)
    contexts = faiss_search(q_vec, top_k=3)

    # Prompt
    context_text = "\n\n".join([f"- {c.get('text') or c.get('content', '')}" for c in contexts]) if contexts else "No matching context found."
    system_prompt = (
        "You are a helpful company assistant. Use the provided context to answer concisely. "
        "If the context does not include the answer, say you don't know."
    )
    user_prompt = f"Question: {payload.content}\n\nContext:\n{context_text}\n\nAnswer:"

    # LLM with graceful fallback
    try:
        ensure_openai_key()
        oai_resp = openai_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=settings.OPENAI_MODEL or None,
            temperature=0.2,
            max_tokens=400,
        )
        answer = oai_resp.get("content") or "I'm unable to generate a response at this time."
    except Exception as e:
        if contexts:
            top_snippets = " ".join([(c.get("text") or c.get("content", ""))[:200] for c in contexts])
            answer = (
                "Based on company knowledge, here's a concise response:\n"
                f"{top_snippets}\n\n"
                "Summary: The information above best matches your query."
            )
        else:
            answer = "I could not find relevant information in the current knowledge base. Please provide more details."
        logger.warning(f"OpenAI unavailable or failed, fallback used: {e}")

    # Store assistant message to DB as part of history
    asst_row = repo_add_chat_message(db, user_id=user["id"], message_text=answer)

    # Compose API response messages (per-session in-memory representation)
    user_msg = Message(
        id=user_row.id, role="user", content=payload.content, created_at=user_row.created_at, session_id=session_id
    )
    asst_msg = Message(
        id=asst_row.id, role="assistant", content=answer, created_at=asst_row.created_at, session_id=session_id
    )
    return [user_msg, asst_msg]


# RAG
@rag_router.post("/ingest", summary="Ingest Documents", description="Ingest a list of documents into the FAISS index", response_model=Dict[str, Any])
def ingest_documents(
    docs: List[Dict[str, str]] = Body(..., description="List of documents: {text: str, source?: str}"),
    user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ensure_faiss_index(settings.EMBEDDING_DIM)
    prepared_docs: List[Dict[str, Any]] = []
    count = 0
    for d in docs:
        text = (d.get("text") or "").strip()
        if not text:
            continue
        meta = {"text": text, "source": d.get("source", "manual")}
        vec = embed_text(text, settings.EMBEDDING_DIM)
        faiss_add(vec, meta, settings.EMBEDDING_DIM)
        prepared_docs.append({"title": meta["source"], "content": text, "embedding": serialize_embedding_f32(vec)})
        count += 1

    # Persist docs
    persisted = repo_ingest_documents(db, prepared_docs)
    return {"ingested": count, "persisted": persisted, "meta": THEME_META}


@rag_router.post("/search", summary="Semantic Search", description="Search the FAISS index for semantically similar passages", response_model=RAGResponse)
def semantic_search(payload: RAGQuery, user: Dict[str, Any] = Depends(get_current_user)):
    ensure_faiss_index(settings.EMBEDDING_DIM)
    vec = embed_text(payload.query, settings.EMBEDDING_DIM)
    results = faiss_search(vec, payload.top_k)
    # simple answer
    if results:
        combined = " ".join([(r.get("text") or r.get("content", "")) for r in results])
        answer = f"Top {len(results)} results synthesized: {combined[:500]}"
    else:
        answer = "No relevant results were found."
    return RAGResponse(query=payload.query, contexts=results, answer=answer)


# Register routers
app.include_router(health_router)
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(rag_router)
