# src/musicrec/faiss_ann.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import numpy as np

try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # keeps import safe in envs without faiss


def _as_float32_2d(X: np.ndarray) -> np.ndarray:
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape={X.shape}")
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)
    return X


def _l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


@dataclass
class FaissANN:
    """
    Minimal FAISS wrapper for cosine similarity using IndexFlatIP
    over L2-normalized vectors.

    IMPORTANT:
    - search() returns 2D arrays (1, k) to match FAISS and your tests.
    """
    index: "faiss.Index"  # type: ignore[name-defined]
    dim: int

    @classmethod
    def build_flat_cosine(cls, X: np.ndarray) -> "FaissANN":
        if faiss is None:
            raise RuntimeError("faiss is not installed in this environment")

        X2 = _as_float32_2d(X)
        dim = int(X2.shape[1])

        Xn = _l2_normalize_rows(X2)

        index = faiss.IndexFlatIP(dim)
        index.add(Xn)

        return cls(index=index, dim=dim)

    def search(self, q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (scores, indices) as 2D arrays of shape (1, k).
        """
        if faiss is None:
            raise RuntimeError("faiss is not installed in this environment")
        if k <= 0:
            raise ValueError("k must be > 0")

        q2 = _as_float32_2d(q)
        if int(q2.shape[1]) != int(self.dim):
            raise ValueError(f"dim mismatch: query_dim={q2.shape[1]} index_dim={self.dim}")

        qn = _l2_normalize_rows(q2)
        scores, idxs = self.index.search(qn, int(k))
        return scores, idxs

    def save_flat_cosine(self, index_path: Union[str, Path]) -> None:
        if faiss is None:
            raise RuntimeError("faiss is not installed in this environment")
        p = Path(index_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p))

    @classmethod
    def load_flat_cosine(cls, index_path: Union[str, Path]) -> "FaissANN":
        if faiss is None:
            raise RuntimeError("faiss is not installed in this environment")
        p = Path(index_path)
        if not p.exists():
            raise FileNotFoundError(f"FAISS index not found: {p}")
        index = faiss.read_index(str(p))
        dim = int(index.d)
        return cls(index=index, dim=dim)