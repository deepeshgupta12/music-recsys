from __future__ import annotations

import json

import numpy as np
import pandas as pd

from musicrec.config import get_paths
from musicrec.features import build_feature_table, build_model_matrix


def test_build_feature_table_outputs_reasonable_rows():
    paths = get_paths()
    df = pd.read_parquet(paths.data_processed_dir / "catalog.parquet")
    ft, report = build_feature_table(df)

    # We expect almost all rows survive (album_name/track_name small nulls are allowed).
    assert report.rows_out > 84000
    assert "days_since_release" in ft.columns
    assert "stream_momentum" in ft.columns


def test_model_matrix_shape_and_no_nans():
    paths = get_paths()
    df = pd.read_parquet(paths.data_processed_dir / "catalog.parquet")
    ft, _ = build_feature_table(df)
    X, names = build_model_matrix(ft)

    assert X.shape[0] == ft.shape[0]
    assert X.shape[1] == len(names)
    assert np.isfinite(X).all()


def test_script_outputs_exist_after_run():
    paths = get_paths()
    # These files should exist after running scripts/03_build_features.py
    assert (paths.data_processed_dir / "catalog_features.parquet").exists()
    assert (paths.data_processed_dir / "catalog_X.npy").exists()
    assert (paths.data_processed_dir / "catalog_X_feature_names.json").exists()
    assert (paths.data_processed_dir / "catalog_feature_report.json").exists()

    with open(paths.data_processed_dir / "catalog_X_feature_names.json", "r", encoding="utf-8") as f:
        names = json.load(f)
    assert isinstance(names, list)
    assert len(names) >= 10