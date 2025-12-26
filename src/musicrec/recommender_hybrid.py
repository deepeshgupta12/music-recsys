from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from musicrec.recommender_knn import KNNRecommender, SimilarTrackResult, SimilarTracksQuery


@dataclass(frozen=True)
class HybridWeights:
    w_sim: float = 0.70
    w_momentum: float = 0.15
    w_popularity: float = 0.10
    w_freshness: float = 0.05


def _minmax(x: np.ndarray) -> np.ndarray:
    x = x.astype("float64", copy=False)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x)
    lo = np.min(x[finite])
    hi = np.max(x[finite])
    if hi - lo < 1e-12:
        out = np.zeros_like(x)
        out[finite] = 0.5
        return out
    out = (x - lo) / (hi - lo)
    out[~finite] = 0.0
    return out


class HybridRecommender:
    """
    Two-stage:
      1) Candidate gen by cosine similarity (using a scaled X is recommended)
      2) Re-rank using hybrid score with momentum/popularity/freshness
    """

    def __init__(self, feature_table: pd.DataFrame, X_for_retrieval: np.ndarray):
        self.ft = feature_table.reset_index(drop=True).copy()
        self.knn = KNNRecommender(self.ft, X_for_retrieval)

    def recommend_similar_hybrid(
        self,
        q: SimilarTracksQuery,
        candidate_k: int = 200,
        weights: HybridWeights = HybridWeights(),
    ) -> Tuple[List[SimilarTrackResult], Dict[str, object]]:
        if candidate_k < q.k:
            raise ValueError("candidate_k must be >= k")
        if candidate_k > 2000:
            raise ValueError("candidate_k too large for V1 baseline")

        # 1) candidates by similarity (same filters as q)
        cand_q = SimilarTracksQuery(
            seed_track_id=q.seed_track_id,
            k=candidate_k,
            same_country_only=q.same_country_only,
            country=q.country,
            exclude_same_artist=q.exclude_same_artist,
            explicit_ok=q.explicit_ok,
            debug=q.debug,
        )
        candidates, dbg = self.knn.recommend_similar(cand_q)

        # 2) gather candidate rows for rerank features
        cand_ids = [c.track_id for c in candidates]
        idx_map = {tid: i for i, tid in enumerate(self.ft["track_id"].astype("string").tolist())}
        cand_idx = np.array([idx_map[tid] for tid in cand_ids], dtype=int)

        sim = np.array([c.score for c in candidates], dtype="float64")
        momentum = self.ft.loc[cand_idx, "stream_momentum"].astype("float64").to_numpy()
        pop = self.ft.loc[cand_idx, "popularity"].astype("float64").to_numpy()
        freshness = (-self.ft.loc[cand_idx, "days_since_release"].astype("float64").to_numpy())

        # normalize all components onto 0..1 for stable weighted blend
        sim_n = _minmax(sim)
        momentum_n = _minmax(momentum)
        pop_n = _minmax(pop)
        fresh_n = _minmax(freshness)

        hybrid = (
            weights.w_sim * sim_n
            + weights.w_momentum * momentum_n
            + weights.w_popularity * pop_n
            + weights.w_freshness * fresh_n
        )

        order = np.argsort(-hybrid)
        reranked = [candidates[i] for i in order[: q.k]]

        debug = dbg if q.debug else {}
        if q.debug:
            debug.update(
                {
                    "candidate_k": candidate_k,
                    "weights": weights.__dict__,
                    "hybrid_top_scores": [float(hybrid[i]) for i in order[: min(q.k, 10)]],
                }
            )

        return reranked, debug