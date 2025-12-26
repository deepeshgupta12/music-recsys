from __future__ import annotations

import json

import numpy as np
import pandas as pd

from musicrec.config import get_paths
from musicrec.features import build_feature_table, build_model_matrix


def main() -> int:
    paths = get_paths()
    in_parquet = paths.data_processed_dir / "catalog.parquet"
    out_features_parquet = paths.data_processed_dir / "catalog_features.parquet"
    out_matrix_npy = paths.data_processed_dir / "catalog_X.npy"
    out_feature_names_json = paths.data_processed_dir / "catalog_X_feature_names.json"
    out_report_json = paths.data_processed_dir / "catalog_feature_report.json"

    if not in_parquet.exists():
        print("ERROR: input parquet not found:", str(in_parquet))
        print("Run: python scripts/02_ingest_to_parquet.py")
        return 2

    df = pd.read_parquet(in_parquet)

    feature_table, report = build_feature_table(df)

    # Save feature table
    feature_table.to_parquet(out_features_parquet, index=False)

    # Save model matrix + feature names
    X, names = build_model_matrix(feature_table)
    np.save(out_matrix_npy, X)
    with open(out_feature_names_json, "w", encoding="utf-8") as f:
        json.dump(names, f, indent=2)

    # Save report
    with open(out_report_json, "w", encoding="utf-8") as f:
        json.dump(report.__dict__, f, indent=2)

    print("OK: wrote feature table:", str(out_features_parquet))
    print("OK: wrote model matrix:", str(out_matrix_npy), "shape=", X.shape)
    print("OK: wrote feature names:", str(out_feature_names_json))
    print("OK: wrote feature report:", str(out_report_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())