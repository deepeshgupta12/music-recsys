from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from musicrec.ann_faiss import FaissANN


def main() -> None:
    # Paths (aligned with your existing processed artifacts)
    base = Path("data/processed")
    ft_path = base / "catalog_features.parquet"
    X_path = base / "catalog_X_scaled.npy"

    # Output artifacts
    out_index = base / "faiss_cosine_flat.index"
    out_track_ids = base / "faiss_track_ids.npy"

    if not ft_path.exists():
        raise FileNotFoundError(f"Missing features parquet: {ft_path}")
    if not X_path.exists():
        raise FileNotFoundError(f"Missing scaled vectors: {X_path}")

    ft = pd.read_parquet(ft_path)
    X = np.load(X_path)

    if len(ft) != X.shape[0]:
        raise ValueError(
            f"Row mismatch: catalog_features has {len(ft)} rows, "
            f"but catalog_X_scaled has {X.shape[0]} rows."
        )

    if "track_id" not in ft.columns:
        raise ValueError("catalog_features.parquet must contain column 'track_id'")

    track_ids = ft["track_id"].astype(str).to_numpy()

    ann = FaissANN.build_flat_cosine(X)
    ann.save(out_index)

    out_track_ids.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_track_ids, track_ids)

    print("FAISS index build complete.")
    print(f"  index_path     = {out_index}")
    print(f"  track_ids_path = {out_track_ids}")
    print(f"  n_items        = {len(track_ids)}")
    print(f"  dim            = {ann.meta.dim}")
    print("  metric         = cosine (IndexFlatIP over L2-normalized vectors)")


if __name__ == "__main__":
    main()