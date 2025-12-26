from __future__ import annotations

import json

import numpy as np
import pandas as pd

from musicrec.config import get_paths
from musicrec.features import build_model_matrix
from musicrec.scaling import fit_standard_scaler, scaler_to_jsonable


def main() -> int:
    paths = get_paths()

    ft_path = paths.data_processed_dir / "catalog_features.parquet"
    splits_path = paths.data_processed_dir / "splits" / "splits.parquet"

    out_scaled = paths.data_processed_dir / "catalog_X_scaled.npy"
    out_scaler = paths.data_processed_dir / "catalog_X_scaler.json"

    if not ft_path.exists():
        print("ERROR: missing catalog_features.parquet. Run: python scripts/03_build_features.py")
        return 2
    if not splits_path.exists():
        print("ERROR: missing splits.parquet. Run: python scripts/04_make_splits.py")
        return 3

    ft = pd.read_parquet(ft_path)
    splits = pd.read_parquet(splits_path)

    merged = ft.merge(splits, on="track_id", how="inner")
    if merged.shape[0] != ft.shape[0]:
        print("WARNING: some track_ids missing from splits merge:", ft.shape[0] - merged.shape[0])

    X, names = build_model_matrix(merged)

    train_mask = (merged["split"] == "train").to_numpy()
    if train_mask.sum() == 0:
        print("ERROR: no train rows found in splits")
        return 4

    scaler = fit_standard_scaler(X[train_mask])
    X_scaled = scaler.transform(X)

    np.save(out_scaled, X_scaled)

    with open(out_scaler, "w", encoding="utf-8") as f:
        json.dump(
            {"feature_names": names, "scaler": scaler_to_jsonable(scaler)},
            f,
            indent=2,
        )

    print("OK: wrote scaled matrix:", str(out_scaled), "shape=", X_scaled.shape)
    print("OK: wrote scaler:", str(out_scaler))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())