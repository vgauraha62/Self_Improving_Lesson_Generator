"""Singleton embeddings — all-MiniLM-L6-v2 reused for disambig, relevance, covers_key_points."""
from __future__ import annotations

import os
from typing import Optional

_model = None
_disabled = os.getenv("DISABLE_EMBEDDINGS", "").lower() in ("1", "true", "yes")

def get_model():
    global _model
    if _disabled:
        return None
    if _model is not None:
        return _model
    try:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        _model = None
    return _model

def embed(text: str) -> Optional[list[float]]:
    m = get_model()
    if m is None:
        return None
    try:
        vec = m.encode(text, normalize_embeddings=False)
        return vec.tolist() if hasattr(vec, "tolist") else list(vec)
    except Exception:
        return None

def cosine_similarity_text(a: str, b: str) -> float:
    """Cosine via embeddings; falls back to jaccard if embeddings unavailable."""
    from src.common.text_utils import jaccard_word_overlap, cosine_similarity

    va = embed(a)
    vb = embed(b)
    if va is None or vb is None:
        return jaccard_word_overlap(a, b)
    return cosine_similarity(va, vb)

def combined_disambig_score(topic_hint: str, candidate_title: str) -> float:
    """score = 0.3*jaccard + 0.7*cosine — FINAL_ARCHITECTURE §3."""
    from src.common.text_utils import jaccard_word_overlap

    j = jaccard_word_overlap(topic_hint, candidate_title)
    c = cosine_similarity_text(topic_hint, candidate_title)
    return 0.3 * j + 0.7 * c
