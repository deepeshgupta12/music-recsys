from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from musicrec.api.main import app


def test_recommend_similar_hybrid_works():
    df = pd.read_parquet("data/processed/catalog_features.parquet")
    seed_id = str(df.iloc[0]["track_id"])

    client = TestClient(app)
    r = client.get(
        "/recommend/similar_hybrid",
        params={
            "seed_track_id": seed_id,
            "k": 5,
            "candidate_k": 200,
            "same_country_only": True,
            "exclude_same_artist": True,
            "explicit_ok": True,
            "debug": True,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["seed_track_id"] == seed_id
    assert body["k"] == 5
    assert len(body["results"]) == 5
    assert "weights" in body
    assert "debug" in body


def test_recommend_similar_hybrid_weight_guard():
    df = pd.read_parquet("data/processed/catalog_features.parquet")
    seed_id = str(df.iloc[0]["track_id"])

    client = TestClient(app)
    r = client.get(
        "/recommend/similar_hybrid",
        params={
            "seed_track_id": seed_id,
            "k": 5,
            "w_sim": 0.0,
            "w_momentum": 0.0,
            "w_popularity": 0.0,
            "w_freshness": 0.0,
        },
    )
    assert r.status_code == 400