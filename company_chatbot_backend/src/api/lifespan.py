from contextlib import asynccontextmanager
from typing import AsyncIterator

from .main import _faiss_add, _embed_text
from src.services.rag_seed import SEED_DOCS


@asynccontextmanager
async def lifespan(app) -> AsyncIterator[None]:
    # Seed RAG index with some demo docs
    for d in SEED_DOCS:
        vec = _embed_text(d["text"])
        _faiss_add(vec, d)
    yield
