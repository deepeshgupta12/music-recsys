from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimilarTracksQuery:
    seed_track_id: str
    k: int = 10
    same_country_only: bool = False
    country: Optional[str] = None  # if same_country_only=True and provided, filter to this country
    exclude_same_artist: bool = True
    explicit_ok: bool = True  # if False, filter out explicit tracks
    debug: bool = False


@dataclass(frozen=True)
class SimilarTrackResult:
    track_id: str
    score: float
    track_name: str
    artist_name: str
    country: str
    genre: str
    popularity: int
    stream_count: int
    release_date: str


class KNNRecommender:
    """
    Cosine similarity recommender over a precomputed numeric matrix X.
    Deterministic baseline.
    """

    def __init__(self, feature_table: pd.DataFrame, X: np.ndarray):
        if "track_id" not in feature_table.columns:
            raise ValueError("feature_table must contain 'track_id'")
        if X.ndim != 2:
            raise ValueError("X must be a 2D numpy array")

        if len(feature_table) != X.shape[0]:
            raise ValueError(f"Row mismatch: feature_table={len(feature_table)} vs X={X.shape[0]}")

        self.ft = feature_table.reset_index(drop=True).copy()
        self.X = X.astype("float64", copy=False)

        # Precompute norms for cosine similarity
        self.X_norm = np.linalg.norm(self.X, axis=1)
        self.X_norm[self.X_norm == 0.0] = 1e-12

        # Index track_id -> row
        self.id_to_idx: Dict[str, int] = {}
        for i, tid in enumerate(self.ft["track_id"].astype("string").fillna("").tolist()):
            self.id_to_idx[str(tid)] = i

    def _filter_mask(self, seed_idx: int, q: SimilarTracksQuery) -> np.ndarray:
        mask = np.ones(len(self.ft), dtype=bool)

        # Exclude the seed itself
        mask[seed_idx] = False

        # Explicit filter
        if not q.explicit_ok and "explicit" in self.ft.columns:
            mask &= (~self.ft["explicit"].astype(bool).to_numpy())

        # Country filter
        if q.same_country_only:
            if q.country is not None:
                mask &= (self.ft["country"].astype("string").to_numpy() == q.country)
            else:
                seed_country = str(self.ft.loc[seed_idx, "country"])
                mask &= (self.ft["country"].astype("string").to_numpy() == seed_country)

        # Exclude same artist
        if q.exclude_same_artist and "artist_name" in self.ft.columns:
            seed_artist = str(self.ft.loc[seed_idx, "artist_name"])
            mask &= (self.ft["artist_name"].astype("string").to_numpy() != seed_artist)

        return mask

    def recommend_similar(self, q: SimilarTracksQuery) -> Tuple[List[SimilarTrackResult], Dict[str, object]]:
        # Updated upper bound: allow larger candidate sets for playlists / diversification
        if q.k <= 0 or q.k > 5000:
            raise ValueError("k must be between 1 and 5000")

        seed_id = str(q.seed_track_id)
        if seed_id not in self.id_to_idx:
            raise KeyError(f"seed_track_id not found: {seed_id}")

        seed_idx = self.id_to_idx[seed_id]
        v = self.X[seed_idx]
        v_norm = np.linalg.norm(v)
        if v_norm == 0.0:
            v_norm = 1e-12

        # Cosine similarity: (X · v) / (||X|| * ||v||)
        dots = self.X @ v
        sims = dots / (self.X_norm * v_norm)

        mask = self._filter_mask(seed_idx, q)
        sims_filtered = np.where(mask, sims, -np.inf)

        available = int(np.isfinite(sims_filtered).sum())
        k = min(q.k, available)
        if k <= 0:
            return [], {"available_candidates": available} if q.debug else {}

        # Top-k indices
        top_idx = np.argpartition(-sims_filtered, kth=min(k, len(sims_filtered) - 1))[:k]
        top_idx = top_idx[np.argsort(-sims_filtered[top_idx])]

        results: List[SimilarTrackResult] = []
        for idx in top_idx:
            row = self.ft.loc[idx]
            results.append(
                SimilarTrackResult(
                    track_id=str(row["track_id"]),
                    score=float(sims_filtered[idx]),
                    track_name=str(row.get("track_name", "")),
                    artist_name=str(row.get("artist_name", "")),
                    country=str(row.get("country", "")),
                    genre=str(row.get("genre", "")),
                    popularity=int(row.get("popularity", 0)),
                    stream_count=int(row.get("stream_count", 0)),
                    release_date=str(row.get("release_date", "")),
                )
            )

        debug = {}
        if q.debug:
            debug = {
                "seed_idx": seed_idx,
                "seed_track_name": str(self.ft.loc[seed_idx].get("track_name", "")),
                "seed_artist_name": str(self.ft.loc[seed_idx].get("artist_name", "")),
                "filters": q.__dict__,
                "available_candidates": available,
            }

        return results, debug