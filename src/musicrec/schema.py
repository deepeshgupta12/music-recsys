from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class CatalogSchema:
    required_columns: List[str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    boolean_columns: List[str]
    date_columns: List[str]


CATALOG_SCHEMA = CatalogSchema(
    required_columns=[
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
    ],
    numeric_columns=[
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
    ],
    categorical_columns=[
        "track_id",
        "track_name",
        "artist_name",
        "album_name",
        "genre",
        "label",
        "country",
    ],
    boolean_columns=["explicit"],
    date_columns=["release_date"],
)


def expected_dtypes() -> Dict[str, str]:
    """
    Dtypes we enforce after ingestion.
    We keep strings as pandas 'string' dtype (nullable).
    """
    return {
        "track_id": "string",
        "track_name": "string",
        "artist_name": "string",
        "album_name": "string",
        "release_date": "datetime64[ns]",
        "genre": "string",
        "label": "string",
        "country": "string",
        "explicit": "bool",
        "popularity": "int64",
        "stream_count": "int64",
        "danceability": "float64",
        "energy": "float64",
        "tempo": "float64",
        "loudness": "float64",
        "key": "int64",
        "mode": "int64",
        "instrumentalness": "float64",
        "duration_ms": "int64",
    }