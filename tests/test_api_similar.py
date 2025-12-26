from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from musicrec.api.main import app


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_recommend_similar_works():
    # pick a real seed id from the processed feature table
    df = pd.read_parquet("data/processed/catalog_features.parquet")
    seed_id = str(df.iloc[0]["track_id"])

    client = TestClient(app)
    r = client.get(
        "/recommend/similar",
        params={
            "seed_track_id": seed_id,
            "k": 5,
            "same_country_only": True,
            "exclude_same_artist": True,
            "explicit_ok": True,
            "debug": True,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["seed_track_id"] == seed_id
    assert len(body["results"]) == 5
    assert "debug" in body


def test_recommend_similar_404_for_bad_seed():
    client = TestClient(app)
    r = client.get("/recommend/similar", params={"seed_track_id": "TRK-NOT-EXISTS", "k": 5})
    assert r.status_code == 404