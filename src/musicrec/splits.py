from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    train_pct: int = 90  # 0..100
    valid_pct: int = 10  # 0..100
    salt: str = "musicrec_v0"


def _hash_to_bucket(track_id: str, salt: str, buckets: int = 10000) -> int:
    """
    Deterministic bucket assignment in [0, buckets).
    Uses md5(salt + track_id) to ensure stable splits.
    """
    key = (salt + "::" + track_id).encode("utf-8")
    h = hashlib.md5(key).hexdigest()
    # Take first 8 hex chars -> 32-bit int
    val = int(h[:8], 16)
    return val % buckets


def assign_split(track_id: str, config: SplitConfig) -> str:
    """
    Returns 'train' or 'valid' deterministically.
    """
    if config.train_pct + config.valid_pct != 100:
        raise ValueError("train_pct + valid_pct must equal 100")

    bucket = _hash_to_bucket(track_id, config.salt, buckets=10000)
    # bucket scaled to 0..9999 -> pct
    pct = (bucket / 10000.0) * 100.0
    if pct < config.train_pct:
        return "train"
    return "valid"


def build_splits(df_features: pd.DataFrame, config: SplitConfig) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Input: catalog_features table (must contain track_id).
    Output:
      - splits_df: columns [track_id, split]
      - counts: {'train': n, 'valid': n}
    """
    if "track_id" not in df_features.columns:
        raise ValueError("features dataframe must contain 'track_id'")

    track_ids = df_features["track_id"].astype("string").fillna("").tolist()
    splits = [assign_split(tid, config) for tid in track_ids]

    out = pd.DataFrame({"track_id": track_ids, "split": splits})

    counts = out["split"].value_counts().to_dict()
    counts = {"train": int(counts.get("train", 0)), "valid": int(counts.get("valid", 0))}
    return out, counts