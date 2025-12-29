from __future__ import annotations

import numpy as np
import pandas as pd

from musicrec.for_you import ForYouQuery, ForYouRecommender
from musicrec.session_store import SessionEvent


def test_for_you_recommender_builds_profile_and_returns_results():
    ft = pd.read_parquet("data/processed/catalog_features.parquet")
    Xs = np.load("data/processed/catalog_X_scaled.npy")

    seed = str(ft.iloc[0]["track_id"])
    # simple synthetic session events
    events = [
        SessionEvent(session_id="s1", track_id=seed, event_type="play", ts="2025-12-26T00:00:00+00:00"),
        SessionEvent(session_id="s1", track_id=seed, event_type="like", ts="2025-12-26T00:05:00+00:00"),
    ]

    rec = ForYouRecommender(ft, Xs)
    q = ForYouQuery(
        session_id="s1",
        n=25,
        candidate_k=1200,
        same_country_only=True,
        explicit_ok=True,
        unique_artist=True,
        max_per_genre=10,
        lambda_relevance=0.75,
        debug=True,
    )

    items, dbg = rec.recommend(q, events)
    assert len(items) >= 10
    assert "profile_debug" in dbg
    # should not recommend already seen track_id
    assert all(x.track_id != seed for x in items)
    # unique artists
    artists = [x.artist_name for x in items]
    assert len(artists) == len(set(artists))