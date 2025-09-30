from contextlib import asynccontextmanager
from typing import AsyncIterator

from .rag_core import faiss_add, embed_text
from src.services.rag_seed import SEED_DOCS
from .main import settings  # settings contains EMBEDDING_DIM


@asynccontextmanager
async def lifespan(app) -> AsyncIterator[None]:
    """Seed the RAG index with demo documents at application startup."""
    for d in SEED_DOCS:
        vec = embed_text(d["text"], settings.EMBEDDING_DIM)
        faiss_add(vec, d, settings.EMBEDDING_DIM)
    yield
