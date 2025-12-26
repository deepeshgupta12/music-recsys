from __future__ import annotations

import json

import pandas as pd

from musicrec.config import get_paths
from musicrec.splits import SplitConfig, build_splits


def test_build_splits_no_overlap_and_full_coverage():
    paths = get_paths()
    features_path = paths.data_processed_dir / "catalog_features.parquet"
    df = pd.read_parquet(features_path)

    splits_df, counts = build_splits(df, SplitConfig(train_pct=90, valid_pct=10, salt="musicrec_v0"))

    assert splits_df.shape[0] == df.shape[0]
    assert set(splits_df["split"].unique()) <= {"train", "valid"}

    train = set(splits_df.loc[splits_df["split"] == "train", "track_id"])
    valid = set(splits_df.loc[splits_df["split"] == "valid", "track_id"])
    assert train.isdisjoint(valid)
    assert len(train) + len(valid) == splits_df.shape[0]

    # sanity: counts should be roughly 90/10
    ratio = counts["train"] / max(1, (counts["train"] + counts["valid"]))
    assert 0.87 <= ratio <= 0.93


def test_script_outputs_exist_after_run():
    paths = get_paths()
    splits_dir = paths.data_processed_dir / "splits"

    assert (splits_dir / "splits.parquet").exists()
    assert (splits_dir / "train_track_ids.txt").exists()
    assert (splits_dir / "valid_track_ids.txt").exists()
    assert (splits_dir / "splits_meta.json").exists()

    with open(splits_dir / "splits_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["config"]["train_pct"] + meta["config"]["valid_pct"] == 100