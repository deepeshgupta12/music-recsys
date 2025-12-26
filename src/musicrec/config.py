from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    data_raw_dir: Path
    data_processed_dir: Path
    raw_catalog_csv: Path


def get_paths() -> Paths:
    """
    Single source of truth for file paths.
    Assumes this file is at: repo_root/src/musicrec/config.py
    """
    repo_root = Path(__file__).resolve().parents[3]
    data_raw_dir = repo_root / "data" / "raw"
    data_processed_dir = repo_root / "data" / "processed"
    raw_catalog_csv = data_raw_dir / "spotify_2015_2025_85k.csv"

    return Paths(
        repo_root=repo_root,
        data_raw_dir=data_raw_dir,
        data_processed_dir=data_processed_dir,
        raw_catalog_csv=raw_catalog_csv,
    )