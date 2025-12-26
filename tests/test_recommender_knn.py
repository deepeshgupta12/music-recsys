from __future__ import annotations

import numpy as np
import pandas as pd

from musicrec.config import get_paths
from musicrec.recommender_knn import KNNRecommender, SimilarTracksQuery


def _load_artifacts():
    paths = get_paths()
    ft = pd.read_parquet(paths.data_processed_dir / "catalog_features.parquet")
    X = np.load(paths.data_processed_dir / "catalog_X.npy")
    return ft, X


def test_knn_recommender_initializes():
    ft, X = _load_artifacts()
    rec = KNNRecommender(ft, X)
    assert len(rec.id_to_idx) == ft.shape[0]


def test_recommend_similar_returns_k():
    ft, X = _load_artifacts()
    rec = KNNRecommender(ft, X)

    seed_id = str(ft.iloc[0]["track_id"])
    results, debug = rec.recommend_similar(
        SimilarTracksQuery(
            seed_track_id=seed_id,
            k=10,
            same_country_only=False,
            exclude_same_artist=True,
            explicit_ok=True,
            debug=True,
        )
    )
    assert len(results) == 10
    assert debug["seed_idx"] >= 0
    assert all(r.track_id != seed_id for r in results)
    assert all(np.isfinite(r.score) for r in results)


def test_same_country_filter_works():
    ft, X = _load_artifacts()
    rec = KNNRecommender(ft, X)

    seed_row = ft.iloc[0]
    seed_id = str(seed_row["track_id"])
    seed_country = str(seed_row["country"])

    results, _ = rec.recommend_similar(
        SimilarTracksQuery(
            seed_track_id=seed_id,
            k=20,
            same_country_only=True,
            country=None,
            exclude_same_artist=False,
            explicit_ok=True,
            debug=False,
        )
    )
    assert len(results) > 0
    assert all(r.country == seed_country for r in results)


def test_explicit_filter_works_if_disabled():
    ft, X = _load_artifacts()
    rec = KNNRecommender(ft, X)

    seed_id = str(ft.iloc[0]["track_id"])
    results, _ = rec.recommend_similar(
        SimilarTracksQuery(
            seed_track_id=seed_id,
            k=20,
            same_country_only=False,
            exclude_same_artist=False,
            explicit_ok=False,
            debug=False,
        )
    )
    assert len(results) > 0
    assert all(r is not None for r in results)