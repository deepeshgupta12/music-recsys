from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from musicrec.config import get_paths
from musicrec.recommender_knn import KNNRecommender, SimilarTracksQuery


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_track_id", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--same_country_only", action="store_true")
    ap.add_argument("--country", type=str, default=None)
    ap.add_argument("--exclude_same_artist", action="store_true")
    ap.add_argument("--explicit_ok", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    paths = get_paths()
    ft_path = paths.data_processed_dir / "catalog_features.parquet"
    X_path = paths.data_processed_dir / "catalog_X.npy"

    if not ft_path.exists() or not X_path.exists():
        print("ERROR: missing features artifacts. Run:")
        print("  python scripts/03_build_features.py")
        return 2

    ft = pd.read_parquet(ft_path)
    X = np.load(X_path)

    rec = KNNRecommender(ft, X)
    q = SimilarTracksQuery(
        seed_track_id=args.seed_track_id,
        k=args.k,
        same_country_only=args.same_country_only,
        country=args.country,
        exclude_same_artist=args.exclude_same_artist,
        explicit_ok=args.explicit_ok,
        debug=args.debug,
    )

    results, debug = rec.recommend_similar(q)
    out = {
        "seed_track_id": args.seed_track_id,
        "k": args.k,
        "results": [r.__dict__ for r in results],
        "debug": debug,
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())