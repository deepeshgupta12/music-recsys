from __future__ import annotations

import inspect
import os
import json
from dataclasses import asdict, is_dataclass
from importlib.metadata import PackageNotFoundError, version as pkg_version
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Any, Dict, Optional, Tuple
from musicrec.faiss_ann import FaissANN
from musicrec.api.feed_routes import router as feed_router
from musicrec.api.feedback_routes import router as feedback_router
from musicrec.api.tag_routes import router as tag_router

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="music-recsys API", version="1.3.1")
app.include_router(feed_router)
app.include_router(feedback_router)
app.include_router(tag_router)
_STARTED_AT = datetime.now(timezone.utc)


# --------------------------------------------------------------------------------------
# Globals (cached)
# --------------------------------------------------------------------------------------

_FEATURE_TABLE: Optional[pd.DataFrame] = None
_X_FOR_RETRIEVAL: Optional[np.ndarray] = None
_X_SCALED: Optional[np.ndarray] = None
_ID_TO_IDX: Optional[Dict[str, int]] = None

_KNN: Any = None
_ANN: Any = None
_HYBRID: Any = None
_PLAYLIST_GEN: Any = None
_SESSION_STORE: Any = None
_FAISS_CACHE = None


# --------------------------------------------------------------------------------------
# Utils
# --------------------------------------------------------------------------------------

def _as_jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if is_dataclass(obj):
        return {k: _as_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (pd.DataFrame,)):
        return obj.to_dict(orient="records")
    if isinstance(obj, (pd.Series,)):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {str(k): _as_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_as_jsonable(x) for x in obj]
    return obj


def _filter_kwargs_for_callable(fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return kwargs
    accepted = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in accepted}


def _construct(cls: Any, **kwargs: Any) -> Any:
    init_kwargs = _filter_kwargs_for_callable(cls.__init__, kwargs)
    return cls(**init_kwargs)


def _repo_root() -> Path:
    # src/musicrec/api/main.py -> repo root
    return Path(__file__).resolve().parents[3]


def _data_dir() -> Path:
    return _repo_root() / "data" / "processed"


def _cache_dir() -> Path:
    p = _repo_root() / ".cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------------------
# Typed builders (CRITICAL FIX: never pass dict queries/weights to recommenders)
# --------------------------------------------------------------------------------------

def _make_similar_tracks_query(**kwargs: Any) -> Any:
    """
    Builds the exact SimilarTracksQuery object from musicrec.recommender_hybrid.
    Must NOT include unknown params like candidate_k.
    """
    from musicrec.recommender_hybrid import SimilarTracksQuery  # type: ignore

    ctor_kwargs = _filter_kwargs_for_callable(SimilarTracksQuery.__init__, kwargs)
    return SimilarTracksQuery(**ctor_kwargs)


def _make_hybrid_weights(w_sim: float, w_momentum: float, w_popularity: float, w_freshness: float) -> Any:
    """
    Builds HybridWeights object so code can access weights.w_sim etc.
    """
    from musicrec.recommender_hybrid import HybridWeights  # type: ignore

    ctor_kwargs = _filter_kwargs_for_callable(
        HybridWeights.__init__,
        {
            "w_sim": float(w_sim),
            "w_momentum": float(w_momentum),
            "w_popularity": float(w_popularity),
            "w_freshness": float(w_freshness),
        },
    )
    return HybridWeights(**ctor_kwargs)


# --------------------------------------------------------------------------------------
# Catalog loading
# --------------------------------------------------------------------------------------

def _load_catalog() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, int]]:
    global _FEATURE_TABLE, _X_FOR_RETRIEVAL, _X_SCALED, _ID_TO_IDX

    if _FEATURE_TABLE is not None and _X_FOR_RETRIEVAL is not None and _X_SCALED is not None and _ID_TO_IDX is not None:
        return _FEATURE_TABLE, _X_FOR_RETRIEVAL, _X_SCALED, _ID_TO_IDX

    parquet_path = _data_dir() / "catalog_features.parquet"
    if not parquet_path.exists():
        raise HTTPException(status_code=500, detail=f"Missing catalog parquet: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    if "track_id" not in df.columns:
        raise HTTPException(status_code=500, detail="catalog_features.parquet missing 'track_id' column")
    df["track_id"] = df["track_id"].astype(str)

    ignore_cols = {
        "track_id",
        "track_name",
        "artist_name",
        "album_name",
        "genre",
        "country",
        "release_date",
    }
    num_cols = [c for c in df.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise HTTPException(status_code=500, detail="No numeric feature columns found in catalog_features.parquet")

    X = df[num_cols].to_numpy(dtype=np.float32, copy=True)
    mu = X.mean(axis=0, keepdims=True)
    sig = X.std(axis=0, keepdims=True)
    sig = np.where(sig == 0, 1.0, sig)
    X_scaled = (X - mu) / sig

    X_for_retrieval = X_scaled
    id_to_idx = {tid: i for i, tid in enumerate(df["track_id"].tolist())}

    _FEATURE_TABLE, _X_FOR_RETRIEVAL, _X_SCALED, _ID_TO_IDX = df, X_for_retrieval, X_scaled, id_to_idx
    return df, X_for_retrieval, X_scaled, id_to_idx


# --------------------------------------------------------------------------------------
# Model singletons
# --------------------------------------------------------------------------------------
def _get_faiss_ann():
    """
    Returns: (ann, track_ids_array, track_id_to_pos)

    - Loads FAISS index + track_ids if present
    - If track_ids.npy is an object array (old/unsafe), rebuilds and overwrites safely
    """
    global _FAISS_CACHE
    if _FAISS_CACHE is not None:
        return _FAISS_CACHE

    from pathlib import Path
    import numpy as np

    from musicrec.faiss_ann import FaissANN

    ft, X_scaled, _, _ = _load_catalog()

    index_path = Path("data/processed/faiss_cosine_flat.index")
    track_ids_path = Path("data/processed/faiss_track_ids.npy")

    ft_track_ids = ft["track_id"].astype(str).to_numpy()

    def _safe_string_array(arr: np.ndarray) -> np.ndarray:
        # Ensure numpy unicode dtype (NOT object)
        arr = np.asarray(arr, dtype=str)
        max_len = int(max((len(x) for x in arr.tolist()), default=16))
        return np.asarray(arr, dtype=f"U{max_len}")

    def _rebuild_and_persist():
        ann_local = FaissANN.build_flat_cosine(X_scaled)
        track_ids_local = _safe_string_array(ft_track_ids)

        index_path.parent.mkdir(parents=True, exist_ok=True)
        ann_local.save_flat_cosine(index_path)
        np.save(track_ids_path, track_ids_local)

        return ann_local, track_ids_local

    if index_path.exists() and track_ids_path.exists():
        ann = FaissANN.load_flat_cosine(index_path)

        try:
            track_ids = np.load(track_ids_path, allow_pickle=False)
        except ValueError:
            # Old file likely saved as object array -> rebuild safely
            ann, track_ids = _rebuild_and_persist()

        # If still unsafe dtype or mismatched length, rebuild
        if getattr(track_ids.dtype, "hasobject", False) or len(track_ids) != len(ft_track_ids):
            ann, track_ids = _rebuild_and_persist()
    else:
        ann, track_ids = _rebuild_and_persist()

    id_to_pos = {str(tid): int(i) for i, tid in enumerate(track_ids.tolist())}

    _FAISS_CACHE = (ann, track_ids, id_to_pos)
    return _FAISS_CACHE

def _get_knn():
    global _KNN
    if _KNN is not None:
        return _KNN

    df, X_for_retrieval, X_scaled, _ = _load_catalog()
    from musicrec.recommender_knn import KNNRecommender  # type: ignore

    _KNN = _construct(
        KNNRecommender,
        feature_table=df,
        X_for_retrieval=X_for_retrieval,
        X=X_for_retrieval,
        X_scaled=X_scaled,
    )
    return _KNN


def _get_hybrid():
    global _HYBRID
    if _HYBRID is not None:
        return _HYBRID

    df, X_for_retrieval, X_scaled, _ = _load_catalog()
    from musicrec.recommender_hybrid import HybridRecommender  # type: ignore

    _HYBRID = _construct(
        HybridRecommender,
        feature_table=df,
        X_for_retrieval=X_for_retrieval,  # required in your current class
        X=X_for_retrieval,
        X_scaled=X_scaled,
        knn=_get_knn(),
    )
    return _HYBRID


def _get_playlist_generator():
    global _PLAYLIST_GEN
    if _PLAYLIST_GEN is not None:
        return _PLAYLIST_GEN

    df, X_for_retrieval, X_scaled, _ = _load_catalog()
    from musicrec.playlist import PlaylistGenerator  # type: ignore

    _PLAYLIST_GEN = _construct(
        PlaylistGenerator,
        hybrid=_get_hybrid(),
        feature_table=df,
        X_scaled=X_scaled,
        X=X_scaled,
        X_for_retrieval=X_for_retrieval,
    )
    return _PLAYLIST_GEN


# --------------------------------------------------------------------------------------
# Session store (local, file-based: avoids mismatch with musicrec.session_store API)
# --------------------------------------------------------------------------------------


def _get_session_store():
    global _SESSION_STORE
    if _SESSION_STORE is not None:
        return _SESSION_STORE

    from musicrec.session_store import SessionStore  # type: ignore

    base_dir = Path(os.environ["MUSICREC_SESSION_DIR"]) if "MUSICREC_SESSION_DIR" in os.environ else (_cache_dir() / "sessions")
    _SESSION_STORE = SessionStore(base_dir=base_dir)
    return _SESSION_STORE

    base_dir = _cache_dir() / "sessions"
    _SESSION_STORE = SessionStore(base_dir=base_dir)
    return _SESSION_STORE


# --------------------------------------------------------------------------------------
# Request models
# --------------------------------------------------------------------------------------

class SessionEventIn(BaseModel):
    session_id: str = Field(..., min_length=3)
    track_id: str = Field(..., min_length=3)
    event_type: str = Field(..., min_length=2)


# --------------------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    """Lightweight health check for local dev + CI smoke."""
    try:
        pkg_ver = pkg_version("musicrec")
    except PackageNotFoundError:
        pkg_ver = "dev"

    now = datetime.now(timezone.utc)
    uptime_s = (now - _STARTED_AT).total_seconds()

    return {
        "ok": True,
        "service": "music-recsys-api",
        "api_version": getattr(app, "version", "unknown"),
        "package_version": pkg_ver,
        "time_utc": now.isoformat(),
        "uptime_s": round(uptime_s, 3),
    }


# --------------------------------------------------------------------------------------
# Similar (KNN)
# --------------------------------------------------------------------------------------

@app.get("/recommend/similar")
def recommend_similar(
    seed_track_id: str,
    k: int = 10,
    same_country_only: bool = False,
    exclude_same_artist: bool = False,
    explicit_ok: bool = True,
    debug: bool = False,
) -> Dict[str, Any]:
    _, _, _, id_to_idx = _load_catalog()
    if seed_track_id not in id_to_idx:
        raise HTTPException(status_code=404, detail=f"seed_track_id not found: {seed_track_id}")

    knn = _get_knn()

    q = _make_similar_tracks_query(
        seed_track_id=seed_track_id,
        k=k,
        same_country_only=same_country_only,
        exclude_same_artist=exclude_same_artist,
        explicit_ok=explicit_ok,
        debug=debug,
    )

    results, dbg = knn.recommend_similar(q)

    payload = {
        "seed_track_id": seed_track_id,
        "k": k,
        "returned": len(results),
        "results": _as_jsonable(results),
    }
    if debug:
        payload["debug"] = _as_jsonable(dbg)
    return payload


# --------------------------------------------------------------------------------------
# Similar (Hybrid)
# --------------------------------------------------------------------------------------

@app.get("/recommend/similar_hybrid")
def recommend_similar_hybrid(
    seed_track_id: str,
    k: int = 10,
    candidate_k: int = 200,
    same_country_only: bool = False,
    exclude_same_artist: bool = False,
    explicit_ok: bool = True,
    debug: bool = False,
    w_sim: float = 0.7,
    w_momentum: float = 0.15,
    w_popularity: float = 0.1,
    w_freshness: float = 0.05,
) -> Dict[str, Any]:
    _, _, _, id_to_idx = _load_catalog()
    if seed_track_id not in id_to_idx:
        raise HTTPException(status_code=404, detail=f"seed_track_id not found: {seed_track_id}")

    if (w_sim + w_momentum + w_popularity + w_freshness) <= 0:
        raise HTTPException(status_code=400, detail="All weights are zero; provide at least one positive weight")

    hybrid = _get_hybrid()

    q = _make_similar_tracks_query(
        seed_track_id=seed_track_id,
        k=k,
        same_country_only=same_country_only,
        exclude_same_artist=exclude_same_artist,
        explicit_ok=explicit_ok,
        debug=debug,
    )
    w = _make_hybrid_weights(w_sim=w_sim, w_momentum=w_momentum, w_popularity=w_popularity, w_freshness=w_freshness)

    candidates, dbg = hybrid.recommend_similar_hybrid(q, candidate_k=candidate_k, weights=w)

    payload = {
        "seed_track_id": seed_track_id,
        "k": k,
        "candidate_k": candidate_k,
        "returned": len(candidates),
        "weights": {
            "w_sim": float(w_sim),
            "w_momentum": float(w_momentum),
            "w_popularity": float(w_popularity),
            "w_freshness": float(w_freshness),
        },
        "results": _as_jsonable(candidates),
    }
    if debug:
        payload["debug"] = _as_jsonable(dbg)
    return payload


# --------------------------------------------------------------------------------------
# Playlist
# --------------------------------------------------------------------------------------

@app.get("/playlist/from_seed")
def playlist_from_seed(
    seed_track_id: str,
    n_tracks: int = 25,
    candidate_k: int = 800,
    same_country_only: bool = False,
    country: Optional[str] = None,
    explicit_ok: bool = True,
    unique_artist: bool = True,
    max_per_genre: int = 8,
    lambda_relevance: float = 0.75,
    debug: bool = False,
    w_sim: float = 0.7,
    w_momentum: float = 0.15,
    w_popularity: float = 0.1,
    w_freshness: float = 0.05,
) -> Dict[str, Any]:
    _, _, _, id_to_idx = _load_catalog()
    if seed_track_id not in id_to_idx:
        raise HTTPException(status_code=404, detail=f"seed_track_id not found: {seed_track_id}")

    if (w_sim + w_momentum + w_popularity + w_freshness) <= 0:
        raise HTTPException(status_code=400, detail="All weights are zero; provide at least one positive weight")

    gen = _get_playlist_generator()

    w = _make_hybrid_weights(w_sim=w_sim, w_momentum=w_momentum, w_popularity=w_popularity, w_freshness=w_freshness)

    try:
        from musicrec.playlist import PlaylistQuery  # type: ignore

        q_kwargs = _filter_kwargs_for_callable(
            PlaylistQuery.__init__,
            {
                "seed_track_id": seed_track_id,
                "n_tracks": n_tracks,
                "candidate_k": candidate_k,
                "same_country_only": same_country_only,
                "country": country,
                "explicit_ok": explicit_ok,
                "unique_artist": unique_artist,
                "max_per_genre": max_per_genre,
                "lambda_relevance": lambda_relevance,
                "debug": debug,
            },
        )
        q = PlaylistQuery(**q_kwargs)
    except Exception:
        class _Q:
            pass
        q = _Q()
        q.seed_track_id = seed_track_id
        q.n_tracks = n_tracks
        q.candidate_k = candidate_k
        q.same_country_only = same_country_only
        q.country = country
        q.explicit_ok = explicit_ok
        q.unique_artist = unique_artist
        q.max_per_genre = max_per_genre
        q.lambda_relevance = lambda_relevance
        q.debug = debug

    playlist, dbg = gen.generate(q, weights=w)

    payload = {
        "seed_track_id": seed_track_id,
        "n_tracks": n_tracks,
        "candidate_k": candidate_k,
        "returned": len(playlist),
        "weights": {
            "w_sim": float(w_sim),
            "w_momentum": float(w_momentum),
            "w_popularity": float(w_popularity),
            "w_freshness": float(w_freshness),
        },
        "playlist": _as_jsonable(playlist),
    }
    if debug:
        payload["debug"] = _as_jsonable(dbg)
    return payload


# --------------------------------------------------------------------------------------
# Session events + For You
# --------------------------------------------------------------------------------------

@app.post("/session/event")
def session_event(payload: SessionEventIn = Body(...)) -> Dict[str, Any]:
    _, _, _, id_to_idx = _load_catalog()
    if payload.track_id not in id_to_idx:
        raise HTTPException(status_code=404, detail=f"track_id not found: {payload.track_id}")

    store = _get_session_store()
    store.append_event(payload.session_id, payload.track_id, payload.event_type)
    events_count = store.count_events(payload.session_id)

    return {"ok": True, "session_id": payload.session_id, "events_count": events_count}

def _event_track_id(e: Any) -> str:
    """
    SessionStore may return SessionEvent objects (preferred) or dicts (legacy).
    Normalize to a track_id string.
    """
    if isinstance(e, dict):
        return str(e.get("track_id") or "")
    return str(getattr(e, "track_id", "") or "")

def _for_you_impl(
    session_id: str,
    n: int = 25,
    candidate_k: int = 1200,
    same_country_only: bool = False,
    explicit_ok: bool = True,
    unique_artist: bool = True,
    max_per_genre: int = 10,
    debug: bool = False,
) -> Dict[str, Any]:
    store = _get_session_store()
    events = store.read_events(session_id)
    if not events:
        raise HTTPException(status_code=404, detail=f"No events found for session_id: {session_id}")

    seed_track_id = _event_track_id(events[-1])
    # Defensive fallback: if last event is malformed, walk backwards
    if not seed_track_id:
        for ev in reversed(events):
            seed_track_id = _event_track_id(ev)
            if seed_track_id:
                    break

    if not seed_track_id:
        raise HTTPException(status_code=500, detail="Unable to determine seed_track_id from session events")

    hybrid = _get_hybrid()

    q = _make_similar_tracks_query(
        seed_track_id=seed_track_id,
        k=n,
        same_country_only=same_country_only,
        exclude_same_artist=unique_artist,
        explicit_ok=explicit_ok,
        debug=debug,
    )
    w = _make_hybrid_weights(w_sim=0.7, w_momentum=0.15, w_popularity=0.1, w_freshness=0.05)

    results, dbg = hybrid.recommend_similar_hybrid(q, candidate_k=candidate_k, weights=w)

    payload = {
        "session_id": session_id,
        "events_count": len(events),
        "seed_track_id": seed_track_id,
        "n": n,
        "returned": len(results),
        "results": _as_jsonable(results),
    }
    if debug:
        payload["debug"] = _as_jsonable(dbg)
    return payload


# REQUIRED BY TESTS:
@app.get("/for_you")
def for_you(
    session_id: str,
    n: int = 25,
    candidate_k: int = 1200,
    same_country_only: bool = False,
    unique_artist: bool = True,
    max_per_genre: int = 10,
    explicit_ok: bool = True,
    debug: bool = False,
) -> Dict[str, Any]:
    return _for_you_impl(
        session_id=session_id,
        n=n,
        candidate_k=candidate_k,
        same_country_only=same_country_only,
        explicit_ok=explicit_ok,
        unique_artist=unique_artist,
        max_per_genre=max_per_genre,
        debug=debug,
    )


# kept for backwards-compat
@app.get("/for_you/from_session")
def for_you_from_session(
    session_id: str,
    n: int = 25,
    candidate_k: int = 1200,
    same_country_only: bool = False,
    explicit_ok: bool = True,
    unique_artist: bool = True,
    max_per_genre: int = 10,
    debug: bool = False,
) -> Dict[str, Any]:
    return _for_you_impl(
        session_id=session_id,
        n=n,
        candidate_k=candidate_k,
        same_country_only=same_country_only,
        explicit_ok=explicit_ok,
        unique_artist=unique_artist,
        max_per_genre=max_per_genre,
        debug=debug,
    )


@app.get("/meta")
def meta() -> Dict[str, Any]:
    """Non-user-facing metadata to help debug local data + index state."""
    # Reuse whatever loader you already have in main.py
    ft, X, Xs, id_to_idx = _load_catalog()

    n_tracks = int(len(ft))
    x_shape = None if X is None else [int(X.shape[0]), int(X.shape[1])]
    xs_shape = None if Xs is None else [int(Xs.shape[0]), int(Xs.shape[1])]

    # These paths/vars MUST already exist in your working main.py.
    # If your file uses different constant names, use those existing names.
    return {
        "ok": True,
        "catalog": {
            "n_tracks": n_tracks,
            "n_ids": int(len(id_to_idx)),
            "features_path": str(_FEATURE_TABLE),
            "X_path": str(_X_FOR_RETRIEVAL) if "CATALOG_X_PATH" in globals() else None,
            "X_scaled_path": str(_X_SCALED),
            "X_shape": x_shape,
            "X_scaled_shape": xs_shape,
        },
    }

@app.get("/recommend/ann")
def recommend_ann(
    seed_track_id: str,
    k: int = 10,
    same_country_only: bool = False,
    explicit_ok: bool = True,
    unique_artist: bool = True,
    debug: bool = False,
):
    import numpy as np

    ft, X_scaled, _, id_to_idx = _load_catalog()

    if seed_track_id not in id_to_idx:
        raise HTTPException(status_code=404, detail=f"track_id not found: {seed_track_id}")

    ann, ann_track_ids, _ = _get_faiss_ann()
    ann_loaded = ann is not None and ann_track_ids is not None

    seed_row_idx = int(id_to_idx[seed_track_id])
    seed_row = ft.iloc[seed_row_idx]

    seed_country = str(seed_row.get("country", "")) if same_country_only else ""
    seed_artist = str(seed_row.get("artist_name", ""))

    # Use the seed vector from current catalog (query vector doesn't depend on FAISS order)
    q = X_scaled[seed_row_idx].astype(np.float32, copy=False)

    # retrieve more than k so filters can still return k
    k_search = int(max(k * 20, 200))
    scores2D, positions2D = ann.search(q, k_search)
    scores = scores2D[0]
    positions = positions2D[0]

    results = []
    seen_artists = set()

    for score, pos in zip(scores.tolist(), positions.tolist()):
        if pos < 0:
            continue

        cand_tid = str(ann_track_ids[int(pos)])
        if cand_tid == seed_track_id:
            continue

        cand_row_idx = id_to_idx.get(cand_tid)
        if cand_row_idx is None:
            continue

        row = ft.iloc[int(cand_row_idx)]

        # filters
        if same_country_only:
            if str(row.get("country", "")) != seed_country:
                continue

        if not explicit_ok:
            # If explicit column exists, enforce it; otherwise ignore
            if "explicit" in ft.columns and bool(row.get("explicit", False)) is True:
                continue

        if unique_artist:
            artist = str(row.get("artist_name", ""))
            if artist in seen_artists:
                continue
            seen_artists.add(artist)

        results.append(
            {
                "track_id": cand_tid,
                "score": float(score),
                "track_name": str(row.get("track_name", "")),
                "artist_name": str(row.get("artist_name", "")),
                "country": str(row.get("country", "")),
                "genre": str(row.get("genre", "")),
                "popularity": int(row.get("popularity", 0)) if "popularity" in ft.columns else 0,
                "stream_count": int(row.get("stream_count", 0)) if "stream_count" in ft.columns else 0,
                "release_date": str(row.get("release_date", "")),
            }
        )

        if len(results) >= int(k):
            break

    payload = {
        "seed_track_id": seed_track_id,
        "k": int(k),
        "returned": len(results),
        "results": results,
        "used_ann": True,
    }

    if debug:
        payload["debug"] = {
            "seed_idx": int(seed_row_idx),
            "seed_track_name": str(seed_row.get("track_name", "")),
            "seed_artist_name": seed_artist,
            "available_candidates": int(len(ann_track_ids)),
            "ann_loaded": ann_loaded,
            "k_search": int(k_search),
            "filters": {
                "same_country_only": bool(same_country_only),
                "explicit_ok": bool(explicit_ok),
                "unique_artist": bool(unique_artist),
                "debug": bool(debug),
            },
        }

    return payload