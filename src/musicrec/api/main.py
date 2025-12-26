from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from musicrec.config import get_paths
from musicrec.recommender_knn import KNNRecommender, SimilarTracksQuery

app = FastAPI(title="musicrec API", version="1.0.1")


@lru_cache(maxsize=1)
def _load_recommender() -> KNNRecommender:
    """
    Load artifacts once per process (cached).
    """
    paths = get_paths()
    ft_path = paths.data_processed_dir / "catalog_features.parquet"
    x_path = paths.data_processed_dir / "catalog_X.npy"

    if not ft_path.exists() or not x_path.exists():
        raise RuntimeError(
            "Missing processed artifacts. Run:\n"
            "  python scripts/02_ingest_to_parquet.py\n"
            "  python scripts/03_build_features.py"
        )

    ft = pd.read_parquet(ft_path)
    X = np.load(x_path)
    return KNNRecommender(ft, X)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/recommend/similar")
def recommend_similar(
    seed_track_id: str = Query(..., description="Seed track_id from catalog_features"),
    k: int = Query(10, ge=1, le=200),
    same_country_only: bool = Query(False),
    country: Optional[str] = Query(None, description="Used only if same_country_only=true"),
    exclude_same_artist: bool = Query(True),
    explicit_ok: bool = Query(True),
    debug: bool = Query(False),
) -> dict:
    try:
        rec = _load_recommender()
        q = SimilarTracksQuery(
            seed_track_id=seed_track_id,
            k=k,
            same_country_only=same_country_only,
            country=country,
            exclude_same_artist=exclude_same_artist,
            explicit_ok=explicit_ok,
            debug=debug,
        )
        results, dbg = rec.recommend_similar(q)
        return {
            "seed_track_id": seed_track_id,
            "k": k,
            "results": [r.__dict__ for r in results],
            "debug": dbg,
        }
    except KeyError:
        raise HTTPException(status_code=404, detail=f"seed_track_id not found: {seed_track_id}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))