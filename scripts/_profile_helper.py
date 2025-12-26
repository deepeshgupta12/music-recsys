from __future__ import annotations

from typing import Dict, List

import pandas as pd

EXPECTED_COLUMNS: List[str] = [
    "track_id",
    "track_name",
    "artist_name",
    "album_name",
    "release_date",
    "genre",
    "label",
    "country",
    "explicit",
    "popularity",
    "stream_count",
    "danceability",
    "energy",
    "tempo",
    "loudness",
    "key",
    "mode",
    "instrumentalness",
    "duration_ms",
]


def profile_catalog_for_test(csv_path: str) -> Dict[str, object]:
    df = pd.read_csv(csv_path)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    return {"missing_expected_columns": missing}