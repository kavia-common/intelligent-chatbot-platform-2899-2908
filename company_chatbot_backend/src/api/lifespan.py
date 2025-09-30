from contextlib import asynccontextmanager
from typing import AsyncIterator

from src.services.rag_service import add_vector as faiss_add, deterministic_embed as embed_text, ensure_index
from src.services.rag_seed import SEED_DOCS
from .main import settings  # settings contains EMBEDDING_DIM


@asynccontextmanager
async def lifespan(app) -> AsyncIterator[None]:
    """Seed the RAG index with demo documents at application startup."""
    ensure_index(settings.EMBEDDING_DIM)
    for d in SEED_DOCS:
        vec = embed_text(d["text"], settings.EMBEDDING_DIM)
        faiss_add(vec, d, settings.EMBEDDING_DIM)
    yield
