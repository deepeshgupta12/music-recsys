from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


NUMERIC_AUDIO_COLS: List[str] = [
    "danceability",
    "energy",
    "tempo",
    "loudness",
    "instrumentalness",
    "duration_ms",
    "key",
    "mode",
]

META_COLS: List[str] = [
    "track_id",
    "track_name",
    "artist_name",
    "album_name",
    "genre",
    "label",
    "country",
    "explicit",
    "release_date",
    "popularity",
    "stream_count",
]


@dataclass(frozen=True)
class FeatureBuildReport:
    rows_in: int
    rows_out: int
    dropped_rows: int
    now_utc: str
    null_rate_out: Dict[str, float]


def _to_naive_utc_timestamp(ts: pd.Timestamp) -> pd.Timestamp:
    """
    Ensure pandas Timestamp is tz-naive (representing UTC clock time).
    Pandas cannot subtract tz-aware from tz-naive datetimes.
    """
    if getattr(ts, "tzinfo", None) is not None:
        # Convert to naive
        try:
            return ts.tz_convert(None)
        except TypeError:
            return ts.tz_localize(None)
    return ts


def _safe_days_since_release(release_dates: pd.Series, now_naive: pd.Timestamp) -> pd.Series:
    # If release_date is NaT, set days_since_release = -1 (so we can filter deterministically)
    delta = (now_naive - release_dates).dt.days
    return delta.fillna(-1).astype("int64")


def build_feature_table(df: pd.DataFrame, now_utc: str | None = None) -> Tuple[pd.DataFrame, FeatureBuildReport]:
    """
    Returns:
      - feature_table: includes engineered columns, cleaned deterministically
      - report
    """
    rows_in = int(df.shape[0])

    # IMPORTANT: make 'now' tz-naive, and keep release_date tz-naive.
    if now_utc is None:
        now_dt = datetime.utcnow().replace(microsecond=0)  # naive UTC
        now = pd.Timestamp(now_dt)
        now_utc = now_dt.isoformat() + "Z"
    else:
        now = _to_naive_utc_timestamp(pd.Timestamp(now_utc))

    # Select only columns we expect (defensive)
    keep_cols = [c for c in META_COLS + NUMERIC_AUDIO_COLS if c in df.columns]
    work = df[keep_cols].copy()

    # Ensure release_date is datetime (tz-naive)
    work["release_date"] = pd.to_datetime(work["release_date"], errors="coerce")

    # Engineered columns (DS)
    work["release_year"] = work["release_date"].dt.year.astype("Int64")
    work["days_since_release"] = _safe_days_since_release(work["release_date"], now)

    # Log transforms to reduce heavy-tail skew
    work["log_stream_count"] = np.log1p(work["stream_count"].astype("float64"))
    work["log_popularity"] = np.log1p(work["popularity"].astype("float64"))

    # Age-normalized momentum proxy (single snapshot)
    dsr = work["days_since_release"].astype("float64")
    dsr = dsr.where(dsr >= 0, np.nan)
    work["stream_momentum"] = work["log_stream_count"] / (dsr + 1.0)

    # Deterministic cleaning rules:
    # - drop rows with missing track_id
    # - drop rows with invalid release_date (days_since_release = -1)
    # - drop rows with any missing numeric audio cols (we need complete vectors later)
    before = int(work.shape[0])

    work = work[work["track_id"].notna()].copy()
    work = work[work["days_since_release"] >= 0].copy()

    required_numeric = [c for c in NUMERIC_AUDIO_COLS if c in work.columns]
    work = work.dropna(subset=required_numeric)

    rows_out = int(work.shape[0])
    dropped = before - rows_out

    null_rate_out = (work.isna().mean().sort_values(ascending=False)).to_dict()

    report = FeatureBuildReport(
        rows_in=rows_in,
        rows_out=rows_out,
        dropped_rows=int(dropped),
        now_utc=str(now_utc),
        null_rate_out=null_rate_out,
    )
    return work, report


def build_model_matrix(feature_table: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Model-ready numeric matrix for similarity retrieval.
    Returns (X, feature_names) where X is float64.
    """
    feature_names = [
        # audio
        "danceability",
        "energy",
        "tempo",
        "loudness",
        "instrumentalness",
        "duration_ms",
        "key",
        "mode",
        # engineered popularity/streams
        "log_stream_count",
        "log_popularity",
        "stream_momentum",
        "days_since_release",
    ]
    missing = [c for c in feature_names if c not in feature_table.columns]
    if missing:
        raise ValueError(f"Missing required engineered feature columns: {missing}")

    X = feature_table[feature_names].astype("float64").to_numpy(copy=True)
    return X, feature_names