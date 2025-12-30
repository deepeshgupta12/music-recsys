from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd


@lru_cache(maxsize=2)
def load_feature_table(parquet_path: str) -> pd.DataFrame:
    """
    Single place to load the feature table used by SegmentFeeds.
    Cached to avoid re-reading parquet on every request.
    """
    p = Path(parquet_path)
    if not p.exists():
        raise FileNotFoundError(f"Feature table parquet not found: {p}")

    ft = pd.read_parquet(p)

    # Normalize important columns (defensive)
    for col in ["track_id", "track_name", "artist_name", "country", "genre"]:
        if col in ft.columns:
            ft[col] = ft[col].astype(str)

    return ft