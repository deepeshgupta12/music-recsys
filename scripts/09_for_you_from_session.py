from __future__ import annotations

import argparse
import json
import uuid

import numpy as np
import pandas as pd

from musicrec.config import get_paths
from musicrec.for_you import ForYouQuery, ForYouRecommender
from musicrec.session_store import SessionStore


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session_id", type=str, default=None)
    ap.add_argument("--event", action="append", default=[], help='Format: "<track_id>:<event_type>" e.g. TRK-...:play')
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--candidate_k", type=int, default=1200)

    ap.add_argument("--same_country_only", action="store_true")
    ap.add_argument("--country", type=str, default=None)
    ap.add_argument("--explicit_ok", action="store_true")

    ap.add_argument("--unique_artist", action="store_true")
    ap.add_argument("--max_per_genre", type=int, default=10)
    ap.add_argument("--lambda_relevance", type=float, default=0.75)

    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    session_id = args.session_id or f"ses-{uuid.uuid4().hex[:12]}"

    paths = get_paths()
    ft_path = paths.data_processed_dir / "catalog_features.parquet"
    xs_path = paths.data_processed_dir / "catalog_X_scaled.npy"
    if not ft_path.exists() or not xs_path.exists():
        print("ERROR: missing processed artifacts. Ensure you ran:")
        print("  python scripts/02_ingest_to_parquet.py")
        print("  python scripts/03_build_features.py")
        print("  python scripts/06_build_scaled_matrix.py")
        return 2

    store = SessionStore(paths.data_processed_dir)

    # append events if provided
    for e in args.event:
        if ":" not in e:
            print(f"ERROR: bad --event format: {e}")
            return 3
        tid, etype = e.split(":", 1)
        store.append(session_id=session_id, track_id=tid.strip(), event_type=etype.strip())

    events = store.read(session_id)

    ft = pd.read_parquet(ft_path)
    Xs = np.load(xs_path)
    rec = ForYouRecommender(ft, Xs)

    q = ForYouQuery(
        session_id=session_id,
        n=args.n,
        candidate_k=args.candidate_k,
        same_country_only=args.same_country_only,
        country=args.country,
        explicit_ok=args.explicit_ok,
        unique_artist=args.unique_artist,
        max_per_genre=args.max_per_genre,
        lambda_relevance=args.lambda_relevance,
        debug=args.debug,
    )

    items, dbg = rec.recommend(q, events)

    out = {
        "session_id": session_id,
        "events_count": len(events),
        "query": q.__dict__,
        "results": [x.__dict__ for x in items],
        "debug": dbg,
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())