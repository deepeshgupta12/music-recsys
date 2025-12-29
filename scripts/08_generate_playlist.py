from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from musicrec.config import get_paths
from musicrec.playlist import PlaylistGenerator, PlaylistQuery
from musicrec.recommender_hybrid import HybridWeights


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_track_id", required=True)
    ap.add_argument("--n_tracks", type=int, default=25)
    ap.add_argument("--candidate_k", type=int, default=600)

    ap.add_argument("--same_country_only", action="store_true")
    ap.add_argument("--country", type=str, default=None)
    ap.add_argument("--explicit_ok", action="store_true")

    ap.add_argument("--unique_artist", action="store_true")
    ap.add_argument("--max_per_genre", type=int, default=8)

    ap.add_argument("--lambda_relevance", type=float, default=0.75)
    ap.add_argument("--debug", action="store_true")

    ap.add_argument("--w_sim", type=float, default=0.70)
    ap.add_argument("--w_momentum", type=float, default=0.15)
    ap.add_argument("--w_popularity", type=float, default=0.10)
    ap.add_argument("--w_freshness", type=float, default=0.05)

    args = ap.parse_args()

    paths = get_paths()
    ft_path = paths.data_processed_dir / "catalog_features.parquet"
    x_scaled_path = paths.data_processed_dir / "catalog_X_scaled.npy"

    if not ft_path.exists():
        print("ERROR: missing catalog_features.parquet. Run: python scripts/03_build_features.py")
        return 2
    if not x_scaled_path.exists():
        print("ERROR: missing catalog_X_scaled.npy. Run: python scripts/06_build_scaled_matrix.py")
        return 3

    ft = pd.read_parquet(ft_path)
    Xs = np.load(x_scaled_path)

    gen = PlaylistGenerator(ft, Xs)

    q = PlaylistQuery(
        seed_track_id=args.seed_track_id,
        n_tracks=args.n_tracks,
        candidate_k=args.candidate_k,
        same_country_only=args.same_country_only,
        country=args.country,
        explicit_ok=args.explicit_ok,
        unique_artist=args.unique_artist,
        max_per_genre=args.max_per_genre,
        lambda_relevance=args.lambda_relevance,
        debug=args.debug,
    )
    w = HybridWeights(args.w_sim, args.w_momentum, args.w_popularity, args.w_freshness)

    playlist, debug = gen.generate(q, weights=w)

    out = {
        "seed_track_id": args.seed_track_id,
        "n_tracks": args.n_tracks,
        "returned": len(playlist),
        "query": q.__dict__,
        "weights": w.__dict__,
        "playlist": [p.__dict__ for p in playlist],
        "debug": debug,
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())