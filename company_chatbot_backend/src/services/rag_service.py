from __future__ import annotations

from typing import List, Dict, Any, Optional
import struct

# Optional FAISS import
try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # fallback

# In-memory corpus for quick retrieval; DB persistence is handled separately
_VECS: List[List[float]] = []
_DOCS: List[Dict[str, Any]] = []
_INDEX = None


# PUBLIC_INTERFACE
def ensure_index(dim: int) -> None:
    """Ensure FAISS index or in-memory fallback is initialized."""
    global _INDEX
    if _INDEX is not None:
        return
    if faiss is not None:  # pragma: no cover
        _INDEX = faiss.IndexFlatIP(dim)
    else:
        _INDEX = "in-memory"


# PUBLIC_INTERFACE
def deterministic_embed(text: str, dim: int) -> List[float]:
    """Create deterministic pseudo-embeddings for consistent testability."""
    import random

    random.seed(hash(text) & 0xFFFFFFFF)
    return [random.random() for _ in range(dim)]


def _cosine(a: List[float], b: List[float]) -> float:
    from math import sqrt
    dot = sum(x * y for x, y in zip(a, b))
    na = sqrt(sum(x * x for x in a)) or 1.0
    nb = sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


# PUBLIC_INTERFACE
def add_vector(vec: List[float], meta: Dict[str, Any], dim: int) -> None:
    """Add vector and metadata to FAISS or in-memory index."""
    ensure_index(dim)
    idx = len(_VECS)
    _VECS.append(vec)
    _DOCS.append({"id": idx, **meta})
    if faiss is not None:  # pragma: no cover
        import numpy as np

        v = np.array([vec], dtype="float32")
        _INDEX.add(v)


# PUBLIC_INTERFACE
def search(vec: List[float], top_k: int) -> List[Dict[str, Any]]:
    """Search for nearest neighbors in FAISS or fallback cosine-similarity."""
    ensure_index(len(vec))
    if faiss is not None:  # pragma: no cover
        import numpy as np

        q = np.array([vec], dtype="float32")
        scores, idxs = _INDEX.search(q, top_k)
        out: List[Dict[str, Any]] = []
        for i, s in zip(idxs[0], scores[0]):
            if i == -1:
                continue
            doc = next((d for d in _DOCS if d["id"] == int(i)), None)
            if doc:
                out.append({"id": int(i), "score": float(s), **doc})
        return out
    # Fallback
    scored = [{"id": i, "score": _cosine(vec, v), **_DOCS[i]} for i, v in enumerate(_VECS)]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# PUBLIC_INTERFACE
def serialize_embedding_f32(vec: List[float]) -> bytes:
    """Serialize float list as bytes for BYTEA storage when pgvector unavailable."""
    return struct.pack(f"{len(vec)}f", *vec)


# PUBLIC_INTERFACE
def deserialize_embedding_f32(buf: Optional[bytes]) -> Optional[List[float]]:
    if not buf:
        return None
    cnt = len(buf) // 4
    return list(struct.unpack(f"{cnt}f", buf))
