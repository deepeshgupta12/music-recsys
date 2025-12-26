from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd

from musicrec.schema import CATALOG_SCHEMA, expected_dtypes


@dataclass(frozen=True)
class IngestReport:
    rows: int
    cols: int
    missing_required_columns: list[str]
    extra_columns: list[str]
    null_rate: dict[str, float]
    invalid_release_date_rows: int


def _coerce_explicit(series: pd.Series) -> pd.Series:
    """
    Coerce explicit to boolean robustly.
    Accepts True/False, 0/1, 'true'/'false', 'yes'/'no'.
    """
    if series.dtype == bool:
        return series

    s = series.astype("string").str.strip().str.lower()
    truthy = {"true", "1", "yes", "y", "t"}
    falsy = {"false", "0", "no", "n", "f"}

    def to_bool(x: str) -> bool:
        if x in truthy:
            return True
        if x in falsy:
            return False
        # fallback: treat unknown as False (conservative)
        return False

    return s.fillna("false").map(to_bool).astype(bool)


def ingest_catalog(csv_path: str) -> Tuple[pd.DataFrame, IngestReport]:
    df = pd.read_csv(csv_path)

    missing = [c for c in CATALOG_SCHEMA.required_columns if c not in df.columns]
    extra = [c for c in df.columns if c not in CATALOG_SCHEMA.required_columns]

    # Keep only required columns, in a deterministic order
    if missing:
        # still return a report; caller can hard-fail
        report = IngestReport(
            rows=int(df.shape[0]),
            cols=int(df.shape[1]),
            missing_required_columns=missing,
            extra_columns=extra,
            null_rate={},
            invalid_release_date_rows=0,
        )
        return df, report

    df = df[CATALOG_SCHEMA.required_columns].copy()

    # Normalize strings
    for c in ["track_id", "track_name", "artist_name", "album_name", "genre", "label", "country"]:
        df[c] = df[c].astype("string").str.strip()

    # Dates
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    invalid_dates = int(df["release_date"].isna().sum())

    # Explicit boolean
    df["explicit"] = _coerce_explicit(df["explicit"])

    # Numeric coercions
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0).astype("int64")
    df["stream_count"] = pd.to_numeric(df["stream_count"], errors="coerce").fillna(0).astype("int64")

    float_cols = ["danceability", "energy", "tempo", "loudness", "instrumentalness"]
    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    int_cols = ["key", "mode", "duration_ms"]
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")

    # Null rates after coercion
    null_rate = (df.isna().mean().sort_values(ascending=False)).to_dict()

    # Enforce final dtypes (best-effort)
    dtype_map = expected_dtypes()
    for col, dtype in dtype_map.items():
        if col not in df.columns:
            continue
        if dtype == "datetime64[ns]":
            continue
        if dtype == "bool":
            continue
        # For strings we already casted; for numeric we casted.
        # Keep here as a consistency checkpoint.
    report = IngestReport(
        rows=int(df.shape[0]),
        cols=int(df.shape[1]),
        missing_required_columns=[],
        extra_columns=extra,
        null_rate=null_rate,
        invalid_release_date_rows=invalid_dates,
    )
    return df, report