"""RAG core utilities and in-memory stores.

This module centralizes the FAISS/in-memory index, embeddings, and helper methods
to avoid circular imports between main FastAPI app and lifespan hooks.
"""

from typing import List, Dict, Any

# Optional imports for FAISS guarded to allow CI without native deps
try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # Will operate in in-memory fallback mode

# In-memory "databases" and RAG structures shared across the app
_DB_USERS: Dict[str, Dict[str, Any]] = {}  # key: email
_DB_SESSIONS: Dict[str, Dict[str, Any]] = {}  # key: session token
_DB_MESSAGES: Dict[str, List[Dict[str, Any]]] = {}  # key: session_id
_DB_DOCS: List[Dict[str, Any]] = []  # corpus for RAG
_FAISS_INDEX = None
_FAISS_VECS: List[List[float]] = []
_FAISS_IDS: List[int] = []

# THEME_META is referenced by responses; keep a minimal default here to avoid cross-imports.
THEME_META = {
    "theme": "Ocean Professional",
    "colors": {
        "primary": "#2563EB",
        "secondary": "#F59E0B",
        "success": "#F59E0B",
        "error": "#EF4444",
        "background": "#f9fafb",
        "surface": "#ffffff",
        "text": "#111827",
    },
}


# PUBLIC_INTERFACE
def get_in_memory_stores() -> Dict[str, Any]:
    """Return references to shared in-memory stores used by the API.

    This is useful for other modules (like main or lifespan) to access or mutate
    the same global structures without importing from each other.
    """
    return {
        "DB_USERS": _DB_USERS,
        "DB_SESSIONS": _DB_SESSIONS,
        "DB_MESSAGES": _DB_MESSAGES,
        "DB_DOCS": _DB_DOCS,
        "FAISS_INDEX": lambda: _FAISS_INDEX,
        "FAISS_VECS": _FAISS_VECS,
        "FAISS_IDS": _FAISS_IDS,
        "THEME_META": THEME_META,
    }


# PUBLIC_INTERFACE
def ensure_faiss_index(embedding_dim: int) -> None:
    """Ensure the FAISS (or fallback) index is initialized."""
    global _FAISS_INDEX
    if _FAISS_INDEX is not None:
        return
    if faiss is not None:  # pragma: no cover
        _FAISS_INDEX = faiss.IndexFlatIP(embedding_dim)
    else:
        _FAISS_INDEX = "in-memory"  # fallback flag


# PUBLIC_INTERFACE
def embed_text(text: str, embedding_dim: int) -> List[float]:
    """Create a deterministic pseudo-embedding for a text string."""
    import random

    random.seed(hash(text) & 0xFFFFFFFF)
    return [random.random() for _ in range(embedding_dim)]


# PUBLIC_INTERFACE
def faiss_add(vec: List[float], meta: Dict[str, Any], embedding_dim: int) -> None:
    """Add a vector and its metadata into the FAISS/fallback index and stores."""
    ensure_faiss_index(embedding_dim)
    idx = len(_FAISS_VECS)
    _FAISS_VECS.append(vec)
    _FAISS_IDS.append(idx)
    _DB_DOCS.append({"id": idx, "text": meta.get("text", ""), "meta": meta})
    if faiss is not None:  # pragma: no cover
        import numpy as np

        v = np.array([vec], dtype="float32")
        _FAISS_INDEX.add(v)


# PUBLIC_INTERFACE
def faiss_search(vec: List[float], top_k: int) -> List[Dict[str, Any]]:
    """Search vectors in FAISS or fallback cosine similarity in-memory store."""
    ensure_faiss_index(len(vec))
    if faiss is not None:  # pragma: no cover
        import numpy as np

        q = np.array([vec], dtype="float32")
        scores, idxs = _FAISS_INDEX.search(q, top_k)
        results = []
        for i, s in zip(idxs[0], scores[0]):
            if i == -1:
                continue
            doc = next((d for d in _DB_DOCS if d["id"] == int(i)), None)
            if doc:
                results.append(
                    {"id": int(i), "score": float(s), "text": doc["text"], "meta": doc["meta"]}
                )
        return results
    else:
        # fallback cosine similarity on in-memory vectors
        from math import sqrt

        def dot(a, b):
            return sum(x * y for x, y in zip(a, b))

        def norm(a):
            return sqrt(sum(x * x for x in a))

        qn = norm(vec) or 1.0
        scored = []
        for i, v in enumerate(_FAISS_VECS):
            s = dot(vec, v) / (qn * (norm(v) or 1.0))
            doc = next((d for d in _DB_DOCS if d["id"] == i), None)
            if doc:
                scored.append({"id": i, "score": float(s), "text": doc["text"], "meta": doc["meta"]})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
