from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from musicrec.config import get_paths
from musicrec.playlist import PlaylistGenerator, PlaylistQuery
from musicrec.recommender_hybrid import HybridRecommender, HybridWeights
from musicrec.recommender_knn import KNNRecommender, SimilarTracksQuery

app = FastAPI(title="musicrec API", version="1.2.1")


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


@lru_cache(maxsize=1)
def _load_playlist_generator() -> PlaylistGenerator:
    paths = get_paths()
    x_scaled_path = paths.data_processed_dir / "catalog_X_scaled.npy"
    if not x_scaled_path.exists():
        raise RuntimeError("Missing catalog_X_scaled.npy. Run: python scripts/06_build_scaled_matrix.py")
    ft = _load_feature_table()
    Xs = np.load(x_scaled_path)
    return PlaylistGenerator(ft, Xs)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/recommend/similar")
def recommend_similar(
    seed_track_id: str = Query(..., description="Seed track_id from catalog_features"),
    k: int = Query(10, ge=1, le=5000),
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


@app.get("/playlist/from_seed")
def playlist_from_seed(
    seed_track_id: str = Query(..., description="Seed track_id from catalog_features"),
    n_tracks: int = Query(25, ge=5, le=100),
    candidate_k: int = Query(600, ge=200, le=5000),

    same_country_only: bool = Query(False),
    country: Optional[str] = Query(None),
    explicit_ok: bool = Query(True),

    unique_artist: bool = Query(True),
    max_per_genre: int = Query(8, ge=1, le=50),

    lambda_relevance: float = Query(0.75, ge=0.0, le=1.0),

    w_sim: float = Query(0.70, ge=0.0, le=1.0),
    w_momentum: float = Query(0.15, ge=0.0, le=1.0),
    w_popularity: float = Query(0.10, ge=0.0, le=1.0),
    w_freshness: float = Query(0.05, ge=0.0, le=1.0),

    debug: bool = Query(False),
) -> dict:
    try:
        gen = _load_playlist_generator()

        q = PlaylistQuery(
            seed_track_id=seed_track_id,
            n_tracks=n_tracks,
            candidate_k=candidate_k,
            same_country_only=same_country_only,
            country=country,
            explicit_ok=explicit_ok,
            unique_artist=unique_artist,
            max_per_genre=max_per_genre,
            lambda_relevance=lambda_relevance,
            debug=debug,
        )

        weights = HybridWeights(w_sim=w_sim, w_momentum=w_momentum, w_popularity=w_popularity, w_freshness=w_freshness)
        if (w_sim + w_momentum + w_popularity + w_freshness) <= 0.0:
            raise ValueError("Hybrid weights sum must be > 0")

        playlist, dbg = gen.generate(q, weights=weights)

        return {
            "seed_track_id": seed_track_id,
            "n_tracks": n_tracks,
            "returned": len(playlist),
            "candidate_k": candidate_k,
            "lambda_relevance": lambda_relevance,
            "weights": weights.__dict__,
            "playlist": [p.__dict__ for p in playlist],
            "debug": dbg,
        }
    except KeyError:
        raise HTTPException(status_code=404, detail=f"seed_track_id not found: {seed_track_id}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))