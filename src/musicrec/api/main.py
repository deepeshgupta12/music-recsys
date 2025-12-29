from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import inspect
import json

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from musicrec.config import get_paths
from musicrec.for_you import ForYouQuery, ForYouRecommender
from musicrec.playlist import PlaylistGenerator, PlaylistQuery
from musicrec.session_store import SessionStore

import musicrec.recommender_knn as rknn
import musicrec.recommender_hybrid as rh


app = FastAPI(title="music-recsys", version="1.3.1")

DEFAULT_WEIGHTS: Dict[str, float] = {
    "w_sim": 0.7,
    "w_momentum": 0.15,
    "w_popularity": 0.1,
    "w_freshness": 0.05,
}

# -----------------------------
# Lazy-loaded singletons/caches
# -----------------------------
_FT: Optional[pd.DataFrame] = None
_TRACK_ID_SET: Optional[set] = None

_X: Optional[np.ndarray] = None
_XS: Optional[np.ndarray] = None
_SCALER: Optional[dict] = None

_KNN: Optional[Any] = None
_HYB: Optional[Any] = None
_PLAYGEN: Optional[Any] = None

_SESS: Optional[SessionStore] = None
_FORYOU: Optional[ForYouRecommender] = None


class HybridQueryShim:
    """
    A small duck-typed query object for HybridRecommender.

    Why: the Query class inside musicrec.recommender_hybrid is immutable
    (so we cannot set candidate_k after construction). This shim avoids mutation
    and provides common pydantic-like helpers used in debug.
    """

    def __init__(self, **kwargs: Any):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    def model_dump(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    def copy(self, update: Optional[Dict[str, Any]] = None) -> "HybridQueryShim":
        data = dict(self.__dict__)
        if update:
            data.update(update)
        return HybridQueryShim(**data)


def _load_feature_table() -> pd.DataFrame:
    global _FT, _TRACK_ID_SET
    if _FT is not None:
        return _FT
    paths = get_paths()
    p = paths.data_processed_dir / "catalog_features.parquet"
    if not p.exists():
        raise RuntimeError(f"missing: {p} (run scripts/03_build_features.py)")
    _FT = pd.read_parquet(p)

    if "track_id" not in _FT.columns:
        raise RuntimeError("catalog_features.parquet missing track_id column")

    _TRACK_ID_SET = set(_FT["track_id"].astype(str).tolist())
    return _FT


def _seed_exists(seed_track_id: str) -> bool:
    global _TRACK_ID_SET
    if _TRACK_ID_SET is None:
        _load_feature_table()
    return seed_track_id in (_TRACK_ID_SET or set())


def _load_X() -> np.ndarray:
    global _X
    if _X is not None:
        return _X
    paths = get_paths()
    p = paths.data_processed_dir / "catalog_X.npy"
    if not p.exists():
        raise RuntimeError(f"missing: {p} (run scripts/03_build_features.py)")
    _X = np.load(p)
    return _X


def _load_X_scaled() -> np.ndarray:
    global _XS
    if _XS is not None:
        return _XS
    paths = get_paths()
    p = paths.data_processed_dir / "catalog_X_scaled.npy"
    if not p.exists():
        raise RuntimeError(f"missing: {p} (run scripts/06_build_scaled_matrix.py)")
    _XS = np.load(p)
    return _XS


def _load_scaler() -> dict:
    global _SCALER
    if _SCALER is not None:
        return _SCALER
    paths = get_paths()
    p = paths.data_processed_dir / "catalog_X_scaler.json"
    if not p.exists():
        _SCALER = {}
        return _SCALER
    _SCALER = json.loads(p.read_text())
    return _SCALER


def _try_construct(cls: type, attempts: Tuple[Tuple[tuple, dict], ...]) -> Any:
    last_err: Optional[Exception] = None
    for args, kwargs in attempts:
        try:
            return cls(*args, **kwargs)
        except TypeError as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    raise RuntimeError(f"Could not construct {cls.__name__}")


def _construct_recommender_knn() -> Any:
    ft = _load_feature_table()
    X = _load_X()

    if not hasattr(rknn, "KNNRecommender"):
        raise RuntimeError("musicrec.recommender_knn must export KNNRecommender")

    cls = rknn.KNNRecommender
    return _try_construct(
        cls,
        attempts=(
            ((ft, X), {}),
            ((ft,), {"X": X}),
            ((), {"feature_table": ft, "X": X}),
            ((), {"df": ft, "X": X}),
        ),
    )


def _construct_recommender_hybrid(knn: Any) -> Any:
    ft = _load_feature_table()
    X = _load_X()
    Xs = _load_X_scaled()
    scaler = _load_scaler()

    if not hasattr(rh, "HybridRecommender"):
        raise RuntimeError("musicrec.recommender_hybrid must export HybridRecommender")

    cls = rh.HybridRecommender

    attempts = (
        ((knn, ft, Xs), {}),
        ((knn, ft, X, Xs), {}),
        ((knn, ft, Xs, scaler), {}),
        ((knn, ft, X, Xs, scaler), {}),
        ((), {"knn": knn, "feature_table": ft, "X_scaled": Xs}),
        ((), {"knn": knn, "df": ft, "X_scaled": Xs}),
        ((), {"knn": knn, "feature_table": ft, "X": X, "X_scaled": Xs}),
        ((), {"knn": knn, "df": ft, "X": X, "X_scaled": Xs}),
        # fallback: hybrid builds knn internally
        ((ft, Xs), {}),
        ((ft, X, Xs), {}),
        ((ft, Xs, scaler), {}),
        ((ft, X, Xs, scaler), {}),
        ((), {"feature_table": ft, "X_scaled": Xs}),
        ((), {"df": ft, "X_scaled": Xs}),
        ((), {"feature_table": ft, "X": X, "X_scaled": Xs}),
        ((), {"df": ft, "X": X, "X_scaled": Xs}),
    )

    hyb = _try_construct(cls, attempts=attempts)

    # guarantee: hybrid must have knn set for candidate retrieval
    if not hasattr(hyb, "knn") or getattr(hyb, "knn") is None:
        try:
            setattr(hyb, "knn", knn)
        except Exception:
            pass

    return hyb


def _construct_playlist_generator(hybrid: Any) -> Any:
    ft = _load_feature_table()
    Xs = _load_X_scaled()

    cls = PlaylistGenerator
    return _try_construct(
        cls,
        attempts=(
            ((hybrid, Xs), {}),
            ((hybrid, ft, Xs), {}),
            ((), {"hybrid": hybrid, "X_scaled": Xs}),
            ((), {"hybrid": hybrid, "df": ft, "X_scaled": Xs}),
            ((), {"recommender": hybrid, "X_scaled": Xs}),
            ((), {"recommender": hybrid, "df": ft, "X_scaled": Xs}),
        ),
    )


def _get_knn():
    global _KNN
    if _KNN is None:
        _KNN = _construct_recommender_knn()
    return _KNN


def _get_hybrid():
    global _HYB
    if _HYB is None:
        knn = _get_knn()
        _HYB = _construct_recommender_hybrid(knn)
    return _HYB


def _get_playlist_generator():
    global _PLAYGEN
    if _PLAYGEN is None:
        _PLAYGEN = _construct_playlist_generator(_get_hybrid())
    return _PLAYGEN


def _get_session_store() -> SessionStore:
    global _SESS
    if _SESS is not None:
        return _SESS
    paths = get_paths()
    _SESS = SessionStore(paths.data_processed_dir)
    return _SESS


def _get_for_you() -> ForYouRecommender:
    global _FORYOU
    if _FORYOU is not None:
        return _FORYOU
    ft = _load_feature_table()
    Xs = _load_X_scaled()
    _FORYOU = ForYouRecommender(ft, Xs)
    return _FORYOU


def _instantiate_query(mod, preferred_names, **kwargs):
    candidates = []
    for name in preferred_names:
        cls = getattr(mod, name, None)
        if isinstance(cls, type):
            candidates.append(cls)
    for name in dir(mod):
        if name.endswith("Query"):
            cls = getattr(mod, name, None)
            if isinstance(cls, type) and cls not in candidates:
                candidates.append(cls)

    if not candidates:
        raise RuntimeError(f"No Query classes found in {mod.__name__}")

    last = None
    for cls in candidates:
        try:
            sig = inspect.signature(cls)
            allowed = set(sig.parameters.keys())
            filtered = {k: v for k, v in kwargs.items() if k in allowed}
            return cls(**filtered)
        except TypeError as e:
            last = e
            continue
    raise RuntimeError(
        f"Could not instantiate any Query in {mod.__name__} with provided params={list(kwargs.keys())}. last={last}"
    )


def _make_knn_query(**kwargs):
    return _instantiate_query(
        rknn,
        preferred_names=["KNNQuery", "SimilarTracksQuery", "SimilarQuery", "Query"],
        **kwargs,
    )


def _parse_weights(
    w_sim: Optional[float],
    w_momentum: Optional[float],
    w_popularity: Optional[float],
    w_freshness: Optional[float],
) -> Optional[Dict[str, float]]:
    provided = [w_sim, w_momentum, w_popularity, w_freshness]
    if all(v is None for v in provided):
        return None

    w = {
        "w_sim": float(w_sim or 0.0),
        "w_momentum": float(w_momentum or 0.0),
        "w_popularity": float(w_popularity or 0.0),
        "w_freshness": float(w_freshness or 0.0),
    }

    if any(v < 0.0 for v in w.values()):
        raise HTTPException(status_code=400, detail="weights must be non-negative")

    s = sum(w.values())
    if s <= 0.0:
        raise HTTPException(status_code=400, detail="at least one weight must be > 0")

    return {k: v / s for k, v in w.items()}


def _safe_call_recommend_similar_hybrid(rec: Any, q: Any, weights: Optional[Dict[str, float]]):
    fn = getattr(rec, "recommend_similar_hybrid")
    sig = inspect.signature(fn)
    if "weights" in sig.parameters:
        return fn(q, weights=weights)
    return fn(q)


def _safe_call_playlist_generate(gen: Any, q: Any, weights: Dict[str, float]):
    fn = getattr(gen, "generate")
    sig = inspect.signature(fn)
    if "weights" in sig.parameters:
        return fn(q, weights=weights)
    return fn(q)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True}


@app.get("/recommend/similar")
def recommend_similar(
    seed_track_id: str = Query(...),
    k: int = Query(10, ge=1),
    same_country_only: bool = Query(False),
    country: Optional[str] = Query(None),
    exclude_same_artist: bool = Query(False),
    explicit_ok: bool = Query(True),
    debug: bool = Query(False),
) -> Dict[str, Any]:
    if not _seed_exists(seed_track_id):
        raise HTTPException(status_code=404, detail="seed_track_id not found")

    try:
        rec = _get_knn()
        q = _make_knn_query(
            seed_track_id=seed_track_id,
            k=k,
            same_country_only=same_country_only,
            country=country,
            exclude_same_artist=exclude_same_artist,
            explicit_ok=explicit_ok,
            debug=debug,
        )
        results, dbg = rec.recommend_similar(q)
        return {"seed_track_id": seed_track_id, "k": k, "results": results, "debug": dbg}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/recommend/similar_hybrid")
def recommend_similar_hybrid(
    seed_track_id: str = Query(...),
    k: int = Query(10, ge=1),
    candidate_k: int = Query(200, ge=1),
    same_country_only: bool = Query(False),
    country: Optional[str] = Query(None),
    exclude_same_artist: bool = Query(False),
    explicit_ok: bool = Query(True),
    debug: bool = Query(False),
    w_sim: Optional[float] = Query(None),
    w_momentum: Optional[float] = Query(None),
    w_popularity: Optional[float] = Query(None),
    w_freshness: Optional[float] = Query(None),
) -> Dict[str, Any]:
    if not _seed_exists(seed_track_id):
        raise HTTPException(status_code=404, detail="seed_track_id not found")

    weights = _parse_weights(w_sim, w_momentum, w_popularity, w_freshness)

    try:
        rec = _get_hybrid()

        # IMPORTANT: do NOT use the module's Query if it can't carry candidate_k.
        # Use shim so we never mutate frozen pydantic models.
        q = HybridQueryShim(
            seed_track_id=seed_track_id,
            k=k,
            candidate_k=candidate_k,
            same_country_only=same_country_only,
            country=country,
            exclude_same_artist=exclude_same_artist,
            explicit_ok=explicit_ok,
            debug=debug,
        )

        results, dbg = _safe_call_recommend_similar_hybrid(rec, q, weights=weights)
        return {
            "seed_track_id": seed_track_id,
            "k": k,
            "candidate_k": candidate_k,
            "weights": (weights if weights is not None else DEFAULT_WEIGHTS),
            "results": results,
            "debug": dbg,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/playlist/from_seed")
def playlist_from_seed(
    seed_track_id: str = Query(...),
    n_tracks: int = Query(25, ge=5, le=200),
    candidate_k: int = Query(800, ge=1),
    same_country_only: bool = Query(False),
    country: Optional[str] = Query(None),
    explicit_ok: bool = Query(True),
    unique_artist: bool = Query(True),
    max_per_genre: int = Query(8, ge=1, le=50),
    lambda_relevance: float = Query(0.75, ge=0.0, le=1.0),
    debug: bool = Query(False),
    w_sim: Optional[float] = Query(None),
    w_momentum: Optional[float] = Query(None),
    w_popularity: Optional[float] = Query(None),
    w_freshness: Optional[float] = Query(None),
) -> Dict[str, Any]:
    if not _seed_exists(seed_track_id):
        raise HTTPException(status_code=404, detail="seed_track_id not found")

    weights = _parse_weights(w_sim, w_momentum, w_popularity, w_freshness) or DEFAULT_WEIGHTS

    try:
        gen = _get_playlist_generator()
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
        playlist, dbg = _safe_call_playlist_generate(gen, q, weights=weights)
        return {
            "seed_track_id": seed_track_id,
            "n_tracks": n_tracks,
            "returned": len(playlist),
            "candidate_k": candidate_k,
            "lambda_relevance": lambda_relevance,
            "weights": weights,
            "playlist": playlist,
            "debug": dbg,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class SessionEventIn(BaseModel):
    session_id: str = Field(..., min_length=1)
    track_id: str = Field(..., min_length=1)
    event_type: str = Field(..., min_length=1)
    ts: Optional[str] = None


@app.post("/session/event")
def session_event(body: SessionEventIn) -> Dict[str, Any]:
    try:
        store = _get_session_store()
        store.append(
            session_id=body.session_id,
            track_id=body.track_id,
            event_type=body.event_type,
            ts=body.ts,
        )
        events = store.read(body.session_id)
        return {"ok": True, "session_id": body.session_id, "events_count": len(events)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/for_you")
def for_you(
    session_id: str = Query(..., min_length=1),
    n: int = Query(30, ge=5, le=200),
    candidate_k: int = Query(1200, ge=1),
    same_country_only: bool = Query(False),
    country: Optional[str] = Query(None),
    explicit_ok: bool = Query(True),
    unique_artist: bool = Query(True),
    max_per_genre: int = Query(10, ge=1, le=50),
    lambda_relevance: float = Query(0.75, ge=0.0, le=1.0),
    debug: bool = Query(False),
) -> Dict[str, Any]:
    try:
        store = _get_session_store()
        events = store.read(session_id)
        if not events:
            raise HTTPException(status_code=404, detail="no events for this session_id")

        rec = _get_for_you()
        q = ForYouQuery(
            session_id=session_id,
            n=n,
            candidate_k=candidate_k,
            same_country_only=same_country_only,
            country=country,
            explicit_ok=explicit_ok,
            unique_artist=unique_artist,
            max_per_genre=max_per_genre,
            lambda_relevance=lambda_relevance,
            debug=debug,
        )
        items, dbg = rec.recommend(q, events)

        out_items = []
        for it in items:
            if isinstance(it, dict):
                out_items.append(it)
            else:
                out_items.append(getattr(it, "__dict__", {"value": str(it)}))

        return {
            "session_id": session_id,
            "events_count": len(events),
            "n": n,
            "returned": len(out_items),
            "candidate_k": candidate_k,
            "lambda_relevance": lambda_relevance,
            "results": out_items,
            "debug": dbg,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))