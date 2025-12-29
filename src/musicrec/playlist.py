from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from musicrec.recommender_hybrid import HybridRecommender, HybridWeights
from musicrec.recommender_knn import SimilarTracksQuery


@dataclass(frozen=True)
class PlaylistQuery:
    seed_track_id: str
    n_tracks: int = 25

    # retrieval
    candidate_k: int = 600

    # filters/constraints
    same_country_only: bool = False
    country: Optional[str] = None
    explicit_ok: bool = True

    unique_artist: bool = True
    max_per_genre: int = 8  # cap per genre in the playlist (excluding the seed)

    # diversity control (MMR)
    lambda_relevance: float = 0.75  # 0..1 (higher = more relevance, lower = more diversity)

    # debug
    debug: bool = False


@dataclass(frozen=True)
class PlaylistItem:
    track_id: str
    track_name: str
    artist_name: str
    genre: str
    country: str
    popularity: int
    stream_count: int
    release_date: str

    # scoring
    relevance_score: float
    redundancy_penalty: float
    mmr_score: float


class PlaylistGenerator:
    """
    Build a playlist from a seed using:
      - Candidate generation via HybridRecommender
      - MMR selection for diversity using cosine similarity on X_scaled
      - Simple constraints: unique artist, max per genre
    """

    def __init__(self, feature_table: pd.DataFrame, X_scaled: np.ndarray):
        if "track_id" not in feature_table.columns:
            raise ValueError("feature_table must contain track_id")
        if len(feature_table) != X_scaled.shape[0]:
            raise ValueError("feature_table rows must match X_scaled rows")

        self.ft = feature_table.reset_index(drop=True).copy()
        self.X = X_scaled.astype("float64", copy=False)

        # precompute norms for cosine similarity
        self.norms = np.linalg.norm(self.X, axis=1)
        self.norms[self.norms == 0.0] = 1e-12

        # id -> idx
        self.id_to_idx: Dict[str, int] = {}
        tids = self.ft["track_id"].astype("string").fillna("").tolist()
        for i, tid in enumerate(tids):
            self.id_to_idx[str(tid)] = i

        # hybrid recommender uses the same X_scaled for retrieval
        self.hybrid = HybridRecommender(self.ft, self.X)

    def _cosine_sim(self, idx_a: int, idx_b: int) -> float:
        va = self.X[idx_a]
        vb = self.X[idx_b]
        return float((va @ vb) / (self.norms[idx_a] * self.norms[idx_b]))

    def _max_sim_to_selected(self, cand_idx: int, selected_idx: List[int]) -> float:
        if not selected_idx:
            return 0.0
        v = self.X[cand_idx]
        dots = self.X[selected_idx] @ v
        sims = dots / (self.norms[selected_idx] * self.norms[cand_idx])
        return float(np.max(sims))

    def generate(
        self,
        q: PlaylistQuery,
        weights: HybridWeights = HybridWeights(),
    ) -> Tuple[List[PlaylistItem], Dict[str, object]]:
        if q.n_tracks < 5 or q.n_tracks > 100:
            raise ValueError("n_tracks must be between 5 and 100")
        if not (0.0 <= q.lambda_relevance <= 1.0):
            raise ValueError("lambda_relevance must be between 0 and 1")
        if q.candidate_k < max(200, q.n_tracks * 10):
            raise ValueError("candidate_k too small; use at least max(200, n_tracks*10)")

        seed_id = str(q.seed_track_id)
        if seed_id not in self.id_to_idx:
            raise KeyError(f"seed_track_id not found: {seed_id}")

        seed_idx = self.id_to_idx[seed_id]
        seed_row = self.ft.loc[seed_idx]

        # 1) candidate generation (hybrid relevance)
        sim_q = SimilarTracksQuery(
            seed_track_id=seed_id,
            k=q.candidate_k,
            same_country_only=q.same_country_only,
            country=q.country,
            exclude_same_artist=False,  # we'll handle artist constraint ourselves
            explicit_ok=q.explicit_ok,
            debug=False,
        )
        candidates, _dbg = self.hybrid.recommend_similar_hybrid(
            sim_q,
            candidate_k=q.candidate_k,
            weights=weights,
        )

        # Candidate pool as indices + relevance score
        cand_ids = [c.track_id for c in candidates]
        cand_scores = np.array([c.score for c in candidates], dtype="float64")  # similarity score from retrieval stage

        cand_idx = []
        for tid in cand_ids:
            if tid in self.id_to_idx:
                cand_idx.append(self.id_to_idx[tid])
        cand_idx = np.array(cand_idx, dtype=int)

        # normalize relevance to 0..1 for MMR stability
        rel = cand_scores[: len(cand_idx)]
        rel_min = float(np.min(rel)) if len(rel) else 0.0
        rel_max = float(np.max(rel)) if len(rel) else 1.0
        if abs(rel_max - rel_min) < 1e-12:
            rel_norm = np.ones_like(rel) * 0.5
        else:
            rel_norm = (rel - rel_min) / (rel_max - rel_min)

        # 2) MMR selection with constraints
        selected_idx: List[int] = []
        used_artists = set([str(seed_row.get("artist_name", ""))]) if q.unique_artist else set()
        genre_counts: Dict[str, int] = {}

        picked: List[PlaylistItem] = []

        # deterministic tie-breaker: prefer lower track_id lexicographically
        cand_track_ids = self.ft.loc[cand_idx, "track_id"].astype("string").to_numpy()

        for _ in range(q.n_tracks):
            best = None  # tuple(mmr_score, -rel, track_id, cand_idx_pos, redundancy)
            best_pos = None
            best_red = None

            for pos, idx in enumerate(cand_idx):
                tid = str(self.ft.loc[idx, "track_id"])
                if tid == seed_id:
                    continue
                # skip if already selected
                if idx in selected_idx:
                    continue

                row = self.ft.loc[idx]
                artist = str(row.get("artist_name", ""))
                genre = str(row.get("genre", ""))

                if q.unique_artist and artist in used_artists:
                    continue

                if q.max_per_genre is not None and q.max_per_genre > 0:
                    if genre_counts.get(genre, 0) >= q.max_per_genre:
                        continue

                redundancy = self._max_sim_to_selected(idx, [seed_idx] + selected_idx)
                mmr = q.lambda_relevance * float(rel_norm[pos]) - (1.0 - q.lambda_relevance) * float(redundancy)

                key = (mmr, float(rel_norm[pos]) * -1.0, tid)  # mmr desc, relevance desc, tid asc

                if best is None or key > best:
                    best = key
                    best_pos = pos
                    best_red = redundancy

            if best is None or best_pos is None:
                break  # no more feasible items

            chosen_idx = int(cand_idx[best_pos])
            chosen_row = self.ft.loc[chosen_idx]

            artist = str(chosen_row.get("artist_name", ""))
            genre = str(chosen_row.get("genre", ""))

            selected_idx.append(chosen_idx)
            if q.unique_artist:
                used_artists.add(artist)
            genre_counts[genre] = genre_counts.get(genre, 0) + 1

            item = PlaylistItem(
                track_id=str(chosen_row.get("track_id", "")),
                track_name=str(chosen_row.get("track_name", "")),
                artist_name=artist,
                genre=genre,
                country=str(chosen_row.get("country", "")),
                popularity=int(chosen_row.get("popularity", 0)),
                stream_count=int(chosen_row.get("stream_count", 0)),
                release_date=str(chosen_row.get("release_date", "")),
                relevance_score=float(rel_norm[best_pos]),
                redundancy_penalty=float(best_red if best_red is not None else 0.0),
                mmr_score=float(best[0]),
            )
            picked.append(item)

        debug = {}
        if q.debug:
            debug = {
                "seed_track_id": seed_id,
                "seed_track_name": str(seed_row.get("track_name", "")),
                "seed_artist_name": str(seed_row.get("artist_name", "")),
                "requested_n": q.n_tracks,
                "returned_n": len(picked),
                "candidate_k": q.candidate_k,
                "lambda_relevance": q.lambda_relevance,
                "unique_artist": q.unique_artist,
                "max_per_genre": q.max_per_genre,
                "genre_counts": dict(sorted(genre_counts.items(), key=lambda x: (-x[1], x[0]))),
            }

        return picked, debug