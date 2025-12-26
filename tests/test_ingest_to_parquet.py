from __future__ import annotations

import json

import pandas as pd

from musicrec.config import get_paths
from musicrec.ingest import ingest_catalog
from musicrec.schema import CATALOG_SCHEMA


def test_ingest_has_no_missing_required_columns():
    paths = get_paths()
    df, report = ingest_catalog(str(paths.raw_catalog_csv))
    assert report.missing_required_columns == []
    assert df.shape[1] == len(CATALOG_SCHEMA.required_columns)


def test_ingest_release_date_parses():
    paths = get_paths()
    df, report = ingest_catalog(str(paths.raw_catalog_csv))
    # We allow a small number, but should not be huge
    assert report.invalid_release_date_rows < 50


def test_parquet_roundtrip(tmp_path):
    paths = get_paths()
    df, report = ingest_catalog(str(paths.raw_catalog_csv))
    out = tmp_path / "catalog.parquet"
    df.to_parquet(out, index=False)

    df2 = pd.read_parquet(out)
    assert df2.shape == df.shape
    assert set(df2.columns) == set(df.columns)