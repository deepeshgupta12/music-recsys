from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from musicrec.config import get_paths
from musicrec.recommender_hybrid import HybridRecommender, HybridWeights
from musicrec.recommender_knn import SimilarTracksQuery


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_track_id", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--candidate_k", type=int, default=200)
    ap.add_argument("--same_country_only", action="store_true")
    ap.add_argument("--country", type=str, default=None)
    ap.add_argument("--exclude_same_artist", action="store_true")
    ap.add_argument("--explicit_ok", action="store_true")
    ap.add_argument("--debug", action="store_true")

    ap.add_argument("--w_sim", type=float, default=0.70)
    ap.add_argument("--w_momentum", type=float, default=0.15)
    ap.add_argument("--w_popularity", type=float, default=0.10)
    ap.add_argument("--w_freshness", type=float, default=0.05)

    args = ap.parse_args()

    paths = get_paths()
    ft_path = paths.data_processed_dir / "catalog_features.parquet"
    X_scaled_path = paths.data_processed_dir / "catalog_X_scaled.npy"

    if not ft_path.exists():
        print("ERROR: missing catalog_features.parquet. Run: python scripts/03_build_features.py")
        return 2
    if not X_scaled_path.exists():
        print("ERROR: missing catalog_X_scaled.npy. Run: python scripts/06_build_scaled_matrix.py")
        return 3

    ft = pd.read_parquet(ft_path)
    Xs = np.load(X_scaled_path)

    rec = HybridRecommender(ft, Xs)

    q = SimilarTracksQuery(
        seed_track_id=args.seed_track_id,
        k=args.k,
        same_country_only=args.same_country_only,
        country=args.country,
        exclude_same_artist=args.exclude_same_artist,
        explicit_ok=args.explicit_ok,
        debug=args.debug,
    )
    w = HybridWeights(args.w_sim, args.w_momentum, args.w_popularity, args.w_freshness)

    results, debug = rec.recommend_similar_hybrid(q, candidate_k=args.candidate_k, weights=w)

    out = {
        "seed_track_id": args.seed_track_id,
        "k": args.k,
        "candidate_k": args.candidate_k,
        "weights": w.__dict__,
        "results": [r.__dict__ for r in results],
        "debug": debug,
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())