from __future__ import annotations

import sys
from dataclasses import asdict
from typing import Dict, List

import pandas as pd

from musicrec.config import get_paths


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


def profile_catalog(csv_path: str) -> Dict[str, object]:
    df = pd.read_csv(csv_path)

    # Basic shape
    n_rows, n_cols = df.shape

    # Column checks
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]

    # Null rates
    null_rate = (df.isna().mean().sort_values(ascending=False)).to_dict()

    # Quick stats for a few known numeric columns (if present)
    numeric_cols = [
        "popularity",
        "stream_count",
        "danceability",
        "energy",
        "tempo",
        "loudness",
        "instrumentalness",
        "duration_ms",
    ]
    numeric_present = [c for c in numeric_cols if c in df.columns]
    stats = df[numeric_present].describe().to_dict() if numeric_present else {}

    # Uniques (small set)
    uniques = {}
    for c in ["country", "genre", "explicit"]:
        if c in df.columns:
            uniques[c] = int(df[c].nunique(dropna=True))

    return {
        "shape": {"rows": int(n_rows), "cols": int(n_cols)},
        "missing_expected_columns": missing,
        "extra_columns": extra,
        "null_rate": null_rate,
        "uniques": uniques,
        "numeric_describe": stats,
    }


def main() -> int:
    paths = get_paths()
    csv_path = paths.raw_catalog_csv

    if not csv_path.exists():
        print("ERROR: raw catalog CSV not found at:", str(csv_path))
        print("Place the file at data/raw/spotify_2015_2025_85k.csv")
        return 2

    report = profile_catalog(str(csv_path))

    print("Repo paths:")
    print(asdict(paths))
    print("\nCatalog profile:")
    print(report)

    # Hard fail if required columns are missing
    if report["missing_expected_columns"]:
        print("\nERROR: Missing expected columns:", report["missing_expected_columns"])
        return 3

    # Soft warning if there are extra columns
    if report["extra_columns"]:
        print("\nWARNING: Extra columns present:", report["extra_columns"])

    print("\nOK: dataset looks compatible with our scoped pipeline (V0).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())