"""Lightweight deterministic embedder (hashing-based) — zero external deps.

Swap for a real model (sentence-transformers, Voyage, OpenAI) in production;
the interface stays identical.
"""
from __future__ import annotations
import hashlib
import re
import numpy as np

DIM = 256
_token = re.compile(r"[a-z0-9]+")


def embed(text: str) -> np.ndarray:
    """Bag-of-hashed-tokens vector, L2-normalized. Deterministic, fast, offline."""
    vec = np.zeros(DIM, dtype=np.float32)
    for tok in _token.findall(text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % DIM] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec
