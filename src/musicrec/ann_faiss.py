from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # type: ignore


def _require_faiss() -> None:
    if faiss is None:
        raise ImportError(
            "faiss is not available. Install faiss-cpu (or faiss) in your venv to use ANN."
        )


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Row-wise L2 normalization. Returns float32.
    """
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


@dataclass(frozen=True)
class FaissIndexMeta:
    """
    Minimal metadata to keep the index usage correct.
    """
    dim: int
    metric: str  # "cosine_ip" (we use IndexFlatIP + L2 norm)
    normalized: bool  # whether vectors must be L2 normalized before add/search


class FaissANN:
    """
    Thin FAISS wrapper for fast ANN retrieval.

    We intentionally keep this small and explicit:
    - Uses cosine similarity via inner product over L2-normalized vectors.
    - Default index: IndexFlatIP (exact, fast enough for V1.1 baseline)
      Later we can upgrade to IVF/HNSW without changing call sites.
    """

    def __init__(self, meta: FaissIndexMeta, index: "faiss.Index") -> None:
        _require_faiss()
        self.meta = meta
        self.index = index

    @classmethod
    def build_flat_cosine(cls, X: np.ndarray) -> "FaissANN":
        """
        Build an IndexFlatIP cosine index from vectors X (N, D).
        """
        _require_faiss()

        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (N, D). Got shape={X.shape}")

        dim = int(X.shape[1])
        meta = FaissIndexMeta(dim=dim, metric="cosine_ip", normalized=True)

        Xn = l2_normalize(X)
        index = faiss.IndexFlatIP(dim)
        index.add(Xn)

        return cls(meta=meta, index=index)

    def search(self, q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.
        Returns:
          scores: (1, k) float32 (cosine similarity)
          idxs:   (1, k) int64
        """
        _require_faiss()

        if k <= 0:
            raise ValueError("k must be > 0")

        q = np.asarray(q, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if q.ndim != 2:
            raise ValueError(f"q must be 1D or 2D. Got shape={q.shape}")

        if q.shape[1] != self.meta.dim:
            raise ValueError(
                f"Query dim mismatch. Expected {self.meta.dim}, got {q.shape[1]}"
            )

        if self.meta.normalized:
            q = l2_normalize(q)

        scores, idxs = self.index.search(q, int(k))
        return scores, idxs

    def save(self, index_path: str | Path) -> None:
        """
        Save index to disk (binary .index file).
        Metadata is encoded in filename conventions for now.
        """
        _require_faiss()
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))

    @classmethod
    def load_flat_cosine(cls, index_path: str | Path, dim: int) -> "FaissANN":
        """
        Load an IndexFlatIP cosine index from disk.
        Caller provides dim to validate.
        """
        _require_faiss()
        index_path = Path(index_path)
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        index = faiss.read_index(str(index_path))
        meta = FaissIndexMeta(dim=int(dim), metric="cosine_ip", normalized=True)

        # sanity: FAISS index should match dim
        if hasattr(index, "d") and int(index.d) != meta.dim:
            raise ValueError(f"FAISS index dim mismatch. expected={meta.dim}, got={index.d}")

        return cls(meta=meta, index=index)