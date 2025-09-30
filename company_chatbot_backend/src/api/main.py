import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, status, Path, Body
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi import APIRouter
from pydantic import BaseModel, Field, EmailStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Import rag_core utilities and shared stores to avoid circular imports
from .rag_core import (
    get_in_memory_stores,
    faiss_search,
    faiss_add,
    embed_text,
    THEME_META as CORE_THEME_META,
)

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

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted Host validation to prevent "Invalid Host header" while allowing container preview domains.
# Defaults include localhost, 127.0.0.1, 0.0.0.0 and wildcard to support preview hosts.
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS,
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# References to shared in-memory stores
_STORES = get_in_memory_stores()
_DB_USERS: Dict[str, Dict[str, Any]] = _STORES["DB_USERS"]
_DB_SESSIONS: Dict[str, Dict[str, Any]] = _STORES["DB_SESSIONS"]
_DB_MESSAGES: Dict[str, List[Dict[str, Any]]] = _STORES["DB_MESSAGES"]
_DB_DOCS: List[Dict[str, Any]] = _STORES["DB_DOCS"]

# --------------------------
# Models
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
# Utility functions (simulated)
# --------------------------
def _hash_password(pw: str) -> str:
    # Simple demo hash (DO NOT USE IN PRODUCTION)
    import hashlib
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()


def _verify_password(pw: str, pw_hash: str) -> bool:
    return _hash_password(pw) == pw_hash


# PUBLIC_INTERFACE
def create_access_token(subject: str, expires_delta: Optional[timedelta] = None) -> Dict[str, Any]:
    """Create a pseudo JWT token for demo purposes.

    In production, use PyJWT and sign with SECRET_KEY.
    """
    exp = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    token = f"demo.{subject}.{int(exp.timestamp())}"
    return {"access_token": token, "expires_at": exp}


# PUBLIC_INTERFACE
def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """Resolve the current user from an access token.

    This demo validates the token format and resolves the email from stored sessions.
    """
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Malformed token")
        subject = parts[1]
        # demo: subject is email
        user = _DB_USERS.get(subject)
        if not user:
            raise ValueError("User not found")
        return user
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


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
def signup(user_in: UserSignup) -> User:
    if user_in.email in _DB_USERS:
        raise HTTPException(status_code=400, detail="Email already registered")
    uid = str(uuid.uuid4())
    now = datetime.utcnow()
    _DB_USERS[user_in.email] = {
        "id": uid,
        "email": user_in.email,
        "name": user_in.name,
        "password_hash": _hash_password(user_in.password),
        "created_at": now,
    }
    return User(id=uid, email=user_in.email, name=user_in.name, created_at=now)


@auth_router.post("/token", summary="Token", description="Obtain access token via OAuth2PasswordRequestForm", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = _DB_USERS.get(form_data.username)
    if not user or not _verify_password(form_data.password, user["password_hash"]):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    token_info = create_access_token(subject=form_data.username)
    _DB_SESSIONS[token_info["access_token"]] = {
        "email": form_data.username,
        "expires_at": token_info["expires_at"],
    }
    return Token(access_token=token_info["access_token"], token_type="bearer", expires_at=token_info["expires_at"], meta=THEME_META)


@auth_router.get("/me", summary="Me", description="Get current user details", response_model=User)
def me(user: Dict[str, Any] = Depends(get_current_user)):
    return User(id=user["id"], email=user["email"], name=user.get("name"), created_at=user["created_at"])


# CHAT
@chat_router.post("/conversations", summary="Create Conversation", description="Create a new conversation/session", response_model=Conversation)
def create_conversation(payload: ConversationCreate = Body(default_factory=ConversationCreate), user: Dict[str, Any] = Depends(get_current_user)):
    session_id = str(uuid.uuid4())
    now = datetime.utcnow()
    _DB_MESSAGES[session_id] = []
    return Conversation(id=session_id, title=payload.title, created_at=now)


@chat_router.get("/conversations", summary="List Conversations", description="List conversations (in-memory demo)", response_model=List[Conversation])
def list_conversations(user: Dict[str, Any] = Depends(get_current_user)):
    # In-memory demo cannot filter by user; returning sessions with simple mapping
    conversations = []
    for session_id in _DB_MESSAGES.keys():
        conversations.append(Conversation(id=session_id, title=None, created_at=datetime.utcnow()))
    return conversations


@chat_router.get("/conversations/{session_id}/messages", summary="Get Messages", description="Get messages for a session", response_model=List[Message])
def get_messages(session_id: str = Path(..., description="Conversation/session ID"), user: Dict[str, Any] = Depends(get_current_user)):
    msgs = _DB_MESSAGES.get(session_id, [])
    return [Message(**m) for m in msgs]


@chat_router.post("/messages", summary="Send Message", description="Send a message and receive assistant response with simple agentic reasoning over RAG contexts", response_model=List[Message])
def send_message(payload: MessageCreate, user: Dict[str, Any] = Depends(get_current_user)):
    # Ensure session
    session_id = payload.session_id or str(uuid.uuid4())
    if session_id not in _DB_MESSAGES:
        _DB_MESSAGES[session_id] = []

    # Store user message
    user_msg = {
        "id": str(uuid.uuid4()),
        "role": "user",
        "content": payload.content,
        "created_at": datetime.utcnow(),
        "session_id": session_id,
        "meta": THEME_META,
    }
    _DB_MESSAGES[session_id].append(user_msg)

    # Basic agentic flow:
    # 1) Retrieve contexts via RAG
    q_vec = embed_text(payload.content, settings.EMBEDDING_DIM)
    contexts = faiss_search(q_vec, top_k=3)

    # 2) Simple reasoning: create a structured answer referencing top contexts
    if contexts:
        top_snippets = " ".join([c["text"][:200] for c in contexts])
        answer = f"Based on company knowledge, here's a concise response:\n{top_snippets}\n\nSummary: The information above best matches your query."
    else:
        answer = "I could not find relevant information in the current knowledge base. Please provide more details."

    # Store assistant message
    asst_msg = {
        "id": str(uuid.uuid4()),
        "role": "assistant",
        "content": answer,
        "created_at": datetime.utcnow(),
        "session_id": session_id,
        "meta": THEME_META,
    }
    _DB_MESSAGES[session_id].append(asst_msg)

    return [Message(**user_msg), Message(**asst_msg)]


# RAG
@rag_router.post("/ingest", summary="Ingest Documents", description="Ingest a list of documents into the FAISS index", response_model=Dict[str, Any])
def ingest_documents(docs: List[Dict[str, str]] = Body(..., description="List of documents: {text: str, source?: str}"), user: Dict[str, Any] = Depends(get_current_user)):
    count = 0
    for d in docs:
        text = d.get("text", "").strip()
        if not text:
            continue
        vec = embed_text(text, settings.EMBEDDING_DIM)
        faiss_add(vec, {"text": text, "source": d.get("source", "manual")}, settings.EMBEDDING_DIM)
        count += 1
    return {"ingested": count, "meta": THEME_META}


@rag_router.post("/search", summary="Semantic Search", description="Search the FAISS index for semantically similar passages", response_model=RAGResponse)
def semantic_search(payload: RAGQuery, user: Dict[str, Any] = Depends(get_current_user)):
    vec = embed_text(payload.query, settings.EMBEDDING_DIM)
    results = faiss_search(vec, payload.top_k)
    # simple answer
    if results:
        combined = " ".join([r['text'] for r in results])
        answer = f"Top {len(results)} results synthesized: {combined[:500]}"
    else:
        answer = "No relevant results were found."
    return RAGResponse(query=payload.query, contexts=results, answer=answer)


# Register routers
app.include_router(health_router)
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(rag_router)
