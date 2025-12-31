from __future__ import annotations

import os
import tempfile

from musicrec.tagging.heuristic import heuristic_tags_from_features
from musicrec.storage.tag_store import TagStore, TagStoreConfig


def test_heuristic_tags_are_deterministic_and_structured():
    b = heuristic_tags_from_features(
        track_id="TRK-1",
        features={"energy": 0.9, "danceability": 0.8, "tempo": 140, "instrumentalness": 0.1, "loudness": -5.0},
        meta={"genre": "Electronic", "country": "Brazil", "explicit": False, "artist_name": "X", "track_name": "Y"},
    )

    assert b.track_id == "TRK-1"
    assert isinstance(b.moods, list) and len(b.moods) >= 1
    assert isinstance(b.scenes, list) and len(b.scenes) >= 1
    assert b.energy_label in ("low", "medium", "high")
    assert b.danceability_label in ("low", "medium", "high")
    assert b.tempo_label in ("slow", "mid", "fast")
    assert b.instrumental_label in ("instrumental", "mostly_instrumental", "vocal")
    assert b.source == "heuristic"


def test_tag_store_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        dbp = os.path.join(td, "tags.db")
        store = TagStore(TagStoreConfig(db_path=dbp))

        b = heuristic_tags_from_features(
            track_id="TRK-2",
            features={"energy": 0.2, "danceability": 0.3, "tempo": 85, "instrumentalness": 0.95, "loudness": -16.0},
            meta={"genre": "Classical", "country": "Brazil", "explicit": False},
        )
        store.upsert(b)

        got = store.get("TRK-2")
        assert got is not None
        assert got.track_id == "TRK-2"
        assert got.instrumental_label in ("instrumental", "mostly_instrumental", "vocal")