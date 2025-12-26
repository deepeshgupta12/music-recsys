from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from musicrec.config import get_paths
from musicrec.recommender_hybrid import HybridRecommender, HybridWeights
from musicrec.recommender_knn import KNNRecommender, SimilarTracksQuery

app = FastAPI(title="musicrec API", version="1.1.1")


@lru_cache(maxsize=1)
def _load_feature_table() -> pd.DataFrame:
    paths = get_paths()
    ft_path = paths.data_processed_dir / "catalog_features.parquet"
    if not ft_path.exists():
        raise RuntimeError(
            "Missing catalog_features.parquet. Run:\n"
            "  python scripts/02_ingest_to_parquet.py\n"
            "  python scripts/03_build_features.py"
        )
    return pd.read_parquet(ft_path)


@lru_cache(maxsize=1)
def _load_knn_recommender() -> KNNRecommender:
    paths = get_paths()
    x_path = paths.data_processed_dir / "catalog_X.npy"
    if not x_path.exists():
        raise RuntimeError("Missing catalog_X.npy. Run: python scripts/03_build_features.py")
    ft = _load_feature_table()
    X = np.load(x_path)
    return KNNRecommender(ft, X)


@lru_cache(maxsize=1)
def _load_hybrid_recommender() -> HybridRecommender:
    paths = get_paths()
    x_scaled_path = paths.data_processed_dir / "catalog_X_scaled.npy"
    if not x_scaled_path.exists():
        raise RuntimeError("Missing catalog_X_scaled.npy. Run: python scripts/06_build_scaled_matrix.py")
    ft = _load_feature_table()
    Xs = np.load(x_scaled_path)
    return HybridRecommender(ft, Xs)


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
        rec = _load_knn_recommender()
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
        return {"seed_track_id": seed_track_id, "k": k, "results": [r.__dict__ for r in results], "debug": dbg}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"seed_track_id not found: {seed_track_id}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend/similar_hybrid")
def recommend_similar_hybrid(
    seed_track_id: str = Query(..., description="Seed track_id from catalog_features"),
    k: int = Query(10, ge=1, le=200),
    candidate_k: int = Query(200, ge=10, le=2000),

    same_country_only: bool = Query(False),
    country: Optional[str] = Query(None),
    exclude_same_artist: bool = Query(True),
    explicit_ok: bool = Query(True),

    w_sim: float = Query(0.70, ge=0.0, le=1.0),
    w_momentum: float = Query(0.15, ge=0.0, le=1.0),
    w_popularity: float = Query(0.10, ge=0.0, le=1.0),
    w_freshness: float = Query(0.05, ge=0.0, le=1.0),

    debug: bool = Query(False),
) -> dict:
    try:
        # weights do not need to sum to 1, but we guard against all zeros
        if (w_sim + w_momentum + w_popularity + w_freshness) <= 0.0:
            raise ValueError("Hybrid weights sum must be > 0")

        rec = _load_hybrid_recommender()

        q = SimilarTracksQuery(
            seed_track_id=seed_track_id,
            k=k,
            same_country_only=same_country_only,
            country=country,
            exclude_same_artist=exclude_same_artist,
            explicit_ok=explicit_ok,
            debug=debug,
        )
        weights = HybridWeights(w_sim=w_sim, w_momentum=w_momentum, w_popularity=w_popularity, w_freshness=w_freshness)

        results, dbg = rec.recommend_similar_hybrid(q, candidate_k=candidate_k, weights=weights)

        return {
            "seed_track_id": seed_track_id,
            "k": k,
            "candidate_k": candidate_k,
            "weights": weights.__dict__,
            "results": [r.__dict__ for r in results],
            "debug": dbg,
        }
    except KeyError:
        raise HTTPException(status_code=404, detail=f"seed_track_id not found: {seed_track_id}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))