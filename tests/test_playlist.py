from __future__ import annotations

import numpy as np
import pandas as pd

from musicrec.playlist import PlaylistGenerator, PlaylistQuery


def test_playlist_generation_constraints():
    ft = pd.read_parquet("data/processed/catalog_features.parquet")
    Xs = np.load("data/processed/catalog_X_scaled.npy")

    gen = PlaylistGenerator(ft, Xs)
    seed_id = str(ft.iloc[0]["track_id"])

    playlist, dbg = gen.generate(
        PlaylistQuery(
            seed_track_id=seed_id,
            n_tracks=25,
            candidate_k=800,
            same_country_only=True,
            explicit_ok=True,
            unique_artist=True,
            max_per_genre=8,
            lambda_relevance=0.75,
            debug=True,
        )
    )

    assert len(playlist) >= 10  # should produce a decent playlist
    track_ids = [p.track_id for p in playlist]
    assert len(track_ids) == len(set(track_ids))

    # unique artist
    artists = [p.artist_name for p in playlist]
    assert len(artists) == len(set(artists))

    # genre cap
    counts = {}
    for p in playlist:
        counts[p.genre] = counts.get(p.genre, 0) + 1
    assert max(counts.values()) <= 8

    assert "genre_counts" in dbg