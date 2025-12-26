from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from musicrec.config import get_paths
from musicrec.splits import SplitConfig, build_splits


def _write_ids(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for tid in ids:
            f.write(f"{tid}\n")


def main() -> int:
    paths = get_paths()
    in_features = paths.data_processed_dir / "catalog_features.parquet"
    splits_dir = paths.data_processed_dir / "splits"
    out_splits_parquet = splits_dir / "splits.parquet"
    out_train_ids = splits_dir / "train_track_ids.txt"
    out_valid_ids = splits_dir / "valid_track_ids.txt"
    out_meta = splits_dir / "splits_meta.json"

    if not in_features.exists():
        print("ERROR: features parquet not found:", str(in_features))
        print("Run: python scripts/03_build_features.py")
        return 2

    df = pd.read_parquet(in_features)

    config = SplitConfig(train_pct=90, valid_pct=10, salt="musicrec_v0")
    splits_df, counts = build_splits(df, config)

    splits_dir.mkdir(parents=True, exist_ok=True)
    splits_df.to_parquet(out_splits_parquet, index=False)

    train_ids = splits_df.loc[splits_df["split"] == "train", "track_id"].tolist()
    valid_ids = splits_df.loc[splits_df["split"] == "valid", "track_id"].tolist()

    _write_ids(out_train_ids, train_ids)
    _write_ids(out_valid_ids, valid_ids)

    meta = {
        "config": config.__dict__,
        "counts": counts,
        "total": int(len(splits_df)),
        "outputs": {
            "splits_parquet": str(out_splits_parquet),
            "train_ids": str(out_train_ids),
            "valid_ids": str(out_valid_ids),
        },
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("OK: wrote splits parquet:", str(out_splits_parquet))
    print("OK: wrote train ids:", str(out_train_ids), "count=", len(train_ids))
    print("OK: wrote valid ids:", str(out_valid_ids), "count=", len(valid_ids))
    print("OK: wrote meta:", str(out_meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())