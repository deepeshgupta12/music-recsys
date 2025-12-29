from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from musicrec.api.main import app


def test_playlist_from_seed_works():
    df = pd.read_parquet("data/processed/catalog_features.parquet")
    seed_id = str(df.iloc[0]["track_id"])

    client = TestClient(app)
    r = client.get(
        "/playlist/from_seed",
        params={
            "seed_track_id": seed_id,
            "n_tracks": 25,
            "candidate_k": 800,
            "same_country_only": True,
            "unique_artist": True,
            "max_per_genre": 8,
            "explicit_ok": True,
            "debug": True,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["seed_track_id"] == seed_id
    assert body["returned"] == 25
    assert len(body["playlist"]) == 25

    # unique track_ids
    tids = [x["track_id"] for x in body["playlist"]]
    assert len(tids) == len(set(tids))

    # unique artists if requested
    artists = [x["artist_name"] for x in body["playlist"]]
    assert len(artists) == len(set(artists))


def test_playlist_weight_guard():
    df = pd.read_parquet("data/processed/catalog_features.parquet")
    seed_id = str(df.iloc[0]["track_id"])

    client = TestClient(app)
    r = client.get(
        "/playlist/from_seed",
        params={
            "seed_track_id": seed_id,
            "n_tracks": 10,
            "w_sim": 0.0,
            "w_momentum": 0.0,
            "w_popularity": 0.0,
            "w_freshness": 0.0,
        },
    )
    assert r.status_code == 400