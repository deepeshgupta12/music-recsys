from __future__ import annotations

import numpy as np
import pandas as pd

from musicrec.recommender_hybrid import HybridRecommender
from musicrec.recommender_knn import SimilarTracksQuery


def test_hybrid_recommender_runs():
    ft = pd.read_parquet("data/processed/catalog_features.parquet")
    Xs = np.load("data/processed/catalog_X_scaled.npy")

    rec = HybridRecommender(ft, Xs)

    seed_id = str(ft.iloc[0]["track_id"])
    results, dbg = rec.recommend_similar_hybrid(
        SimilarTracksQuery(
            seed_track_id=seed_id,
            k=10,
            same_country_only=True,
            exclude_same_artist=True,
            explicit_ok=True,
            debug=True,
        ),
        candidate_k=200,
    )

    assert len(results) == 10
    assert "candidate_k" in dbg