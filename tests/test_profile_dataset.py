from __future__ import annotations

from musicrec.config import get_paths
from scripts._profile_helper import profile_catalog_for_test


def test_catalog_exists_and_has_expected_columns():
    paths = get_paths()
    assert paths.raw_catalog_csv.exists(), (
        f"Dataset not found at {paths.raw_catalog_csv}. "
        "Copy it to data/raw/spotify_2015_2025_85k.csv"
    )

    report = profile_catalog_for_test(str(paths.raw_catalog_csv))
    assert report["missing_expected_columns"] == [], f"Missing columns: {report['missing_expected_columns']}"