from __future__ import annotations

import inspect
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field


# --------------------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------------------

app = FastAPI(title="music-recsys API", version="1.3.1")


# --------------------------------------------------------------------------------------
# Globals (cached singletons)
# --------------------------------------------------------------------------------------

_FEATURE_TABLE: Optional[pd.DataFrame] = None
_X_FOR_RETRIEVAL: Optional[np.ndarray] = None
_X_SCALED: Optional[np.ndarray] = None
_ID_TO_IDX: Optional[Dict[str, int]] = None

_KNN: Any = None
_HYBRID: Any = None
_PLAYLIST_GEN: Any = None
_SESSION_STORE: Any = None
_FOR_YOU: Any = None


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def _as_jsonable(obj: Any) -> Any:
    """Convert dataclasses / numpy / pandas / datetime to json-safe structures."""
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
    """Filter kwargs to only those accepted by fn."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return kwargs
    accepted = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in accepted}


def _construct(cls: Any, **kwargs: Any) -> Any:
    """
    Construct a class using only kwargs it accepts.
    This avoids breakage when constructor param names differ.
    """
    init_kwargs = _filter_kwargs_for_callable(cls.__init__, kwargs)
    return cls(**init_kwargs)


def _repo_root() -> Path:
    # src/musicrec/api/main.py -> repo_root
    return Path(__file__).resolve().parents[3]


def _data_dir() -> Path:
    return _repo_root() / "data" / "processed"


def _cache_dir() -> Path:
    p = _repo_root() / ".cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _format_release_date(x: Any) -> str:
    """
    Match the repo’s current style in outputs:
    'YYYY-MM-DD HH:MM:SS' (no timezone)
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    if isinstance(x, str):
        # already formatted in many of our pipelines
        return x
    if isinstance(x, pd.Timestamp):
        return x.to_pydatetime().strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(x, datetime):
        return x.strftime("%Y-%m-%d %H:%M:%S")
    try:
        ts = pd.to_datetime(x, errors="coerce")
        if pd.isna(ts):
            return ""
        return ts.to_pydatetime().strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(x)


# --------------------------------------------------------------------------------------
# Catalog loading + matrices
# --------------------------------------------------------------------------------------

def _load_catalog() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Loads:
    - feature_table (parquet)
    Builds:
    - X_for_retrieval (used by KNN retrieval)
    - X_scaled (used by rerank/MMR)
    - id_to_idx map
    """
    global _FEATURE_TABLE, _X_FOR_RETRIEVAL, _X_SCALED, _ID_TO_IDX

    if _FEATURE_TABLE is not None and _X_FOR_RETRIEVAL is not None and _X_SCALED is not None and _ID_TO_IDX is not None:
        return _FEATURE_TABLE, _X_FOR_RETRIEVAL, _X_SCALED, _ID_TO_IDX

    parquet_path = _data_dir() / "catalog_features.parquet"
    if not parquet_path.exists():
        raise HTTPException(status_code=500, detail=f"Missing catalog parquet: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    # normalize id column
    if "track_id" not in df.columns:
        raise HTTPException(status_code=500, detail="catalog_features.parquet missing 'track_id' column")
    df["track_id"] = df["track_id"].astype(str)

    # normalize release_date for output formatting
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

    # numeric feature matrix
    # (keep this consistent and robust across synthetic datasets)
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

    # scale (avoid sklearn dependency surprises: do simple standardization)
    mu = X.mean(axis=0, keepdims=True)
    sig = X.std(axis=0, keepdims=True)
    sig = np.where(sig == 0, 1.0, sig)
    X_scaled = (X - mu) / sig

    # For retrieval we can use the same scaled matrix (works for cosine/knn),
    # but keep it separate because your HybridRecommender expects X_for_retrieval.
    X_for_retrieval = X_scaled

    id_to_idx = {tid: i for i, tid in enumerate(df["track_id"].tolist())}

    _FEATURE_TABLE, _X_FOR_RETRIEVAL, _X_SCALED, _ID_TO_IDX = df, X_for_retrieval, X_scaled, id_to_idx
    return df, X_for_retrieval, X_scaled, id_to_idx


# --------------------------------------------------------------------------------------
# Model singletons (KNN / Hybrid / Playlist / Session store / For You)
# --------------------------------------------------------------------------------------

def _get_knn():
    global _KNN
    if _KNN is not None:
        return _KNN

    df, X_for_retrieval, X_scaled, _ = _load_catalog()
    from musicrec.recommender_knn import KNNRecommender  # type: ignore

    # Try common constructor params
    # (constructor introspection keeps this stable if your class evolves)
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
        X_for_retrieval=X_for_retrieval,   # REQUIRED in your current class
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
        X_scaled=X_scaled,                 # REQUIRED by your PlaylistGenerator
        X=X_scaled,
        X_for_retrieval=X_for_retrieval,
    )
    return _PLAYLIST_GEN


def _get_session_store():
    """
    Your SessionStore requires base_dir.
    Also: method names evolved (append_event vs add_event etc),
    so API routes call helper wrappers rather than direct method name.
    """
    global _SESSION_STORE
    if _SESSION_STORE is not None:
        return _SESSION_STORE

    from musicrec.session_store import SessionStore  # type: ignore

    base_dir = _cache_dir() / "sessions"
    base_dir.mkdir(parents=True, exist_ok=True)

    _SESSION_STORE = _construct(SessionStore, base_dir=base_dir)
    return _SESSION_STORE


def _get_for_you():
    global _FOR_YOU
    if _FOR_YOU is not None:
        return _FOR_YOU

    df, X_for_retrieval, X_scaled, _ = _load_catalog()

    # Prefer your dedicated module (V1.3.0),
    # but keep a fallback if names shift.
    try:
        from musicrec.for_you import ForYouFeed  # type: ignore

        _FOR_YOU = _construct(
            ForYouFeed,
            feature_table=df,
            X_scaled=X_scaled,
            X=X_scaled,
            X_for_retrieval=X_for_retrieval,
        )
    except Exception:
        _FOR_YOU = None

    return _FOR_YOU


# --------------------------------------------------------------------------------------
# SessionStore wrappers (method-name compatibility)
# --------------------------------------------------------------------------------------

def _store_add_event(store: Any, session_id: str, track_id: str, event_type: str) -> None:
    if hasattr(store, "add_event"):
        store.add_event(session_id=session_id, track_id=track_id, event_type=event_type)
        return
    if hasattr(store, "append_event"):
        store.append_event(session_id=session_id, track_id=track_id, event_type=event_type)
        return
    if hasattr(store, "log_event"):
        store.log_event(session_id=session_id, track_id=track_id, event_type=event_type)
        return
    raise AttributeError("SessionStore missing add_event/append_event/log_event")


def _store_get_events(store: Any, session_id: str) -> list[dict]:
    if hasattr(store, "get_events"):
        return store.get_events(session_id=session_id)
    if hasattr(store, "load_events"):
        return store.load_events(session_id=session_id)
    if hasattr(store, "events_for"):
        return store.events_for(session_id=session_id)

    # fallback: try reading JSONL directly if store exposes base_dir
    base_dir = getattr(store, "base_dir", None)
    if base_dir is not None:
        p = Path(base_dir) / f"{session_id}.jsonl"
        if not p.exists():
            return []
        out: list[dict] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        return out

    return []


# --------------------------------------------------------------------------------------
# Request models
# --------------------------------------------------------------------------------------

class SessionEventIn(BaseModel):
    session_id: str = Field(..., min_length=3)
    track_id: str = Field(..., min_length=3)
    event_type: str = Field(..., min_length=2)


# --------------------------------------------------------------------------------------
# Basic health
# --------------------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "music-recsys", "version": app.version}


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
    df, _, _, id_to_idx = _load_catalog()

    if seed_track_id not in id_to_idx:
        raise HTTPException(status_code=404, detail=f"seed_track_id not found: {seed_track_id}")

    knn = _get_knn()

    # Query object name varies; safest is to reuse SimilarTracksQuery if present.
    q_obj: Any = None
    try:
        from musicrec.recommender_hybrid import SimilarTracksQuery  # type: ignore
        q_obj = SimilarTracksQuery(
            seed_track_id=seed_track_id,
            k=k,
            candidate_k=k,
            same_country_only=same_country_only,
            exclude_same_artist=exclude_same_artist,
            explicit_ok=explicit_ok,
            debug=debug,
        )
    except Exception:
        q_obj = {
            "seed_track_id": seed_track_id,
            "k": k,
            "candidate_k": k,
            "same_country_only": same_country_only,
            "exclude_same_artist": exclude_same_artist,
            "explicit_ok": explicit_ok,
            "debug": debug,
        }

    try:
        results, dbg = knn.recommend_similar(q_obj)
    except TypeError:
        # maybe expects parameters directly
        results, dbg = knn.recommend_similar(
            seed_track_id=seed_track_id,
            k=k,
            same_country_only=same_country_only,
            exclude_same_artist=exclude_same_artist,
            explicit_ok=explicit_ok,
            debug=debug,
        )

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
# Similar (Hybrid + weights)
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
    df, _, _, id_to_idx = _load_catalog()

    if seed_track_id not in id_to_idx:
        raise HTTPException(status_code=404, detail=f"seed_track_id not found: {seed_track_id}")

    # Weight guard (your tests expect 400 when all weights are 0)
    if (w_sim + w_momentum + w_popularity + w_freshness) <= 0:
        raise HTTPException(status_code=400, detail="All weights are zero; provide at least one positive weight")

    hybrid = _get_hybrid()

    # Use SimilarTracksQuery (we know it exists in your module from your dir() output)
    try:
        from musicrec.recommender_hybrid import SimilarTracksQuery  # type: ignore

        q = SimilarTracksQuery(
            seed_track_id=seed_track_id,
            k=k,
            candidate_k=candidate_k,
            same_country_only=same_country_only,
            exclude_same_artist=exclude_same_artist,
            explicit_ok=explicit_ok,
            debug=debug,
        )
    except Exception:
        q = {
            "seed_track_id": seed_track_id,
            "k": k,
            "candidate_k": candidate_k,
            "same_country_only": same_country_only,
            "exclude_same_artist": exclude_same_artist,
            "explicit_ok": explicit_ok,
            "debug": debug,
        }

    weights = {
        "w_sim": float(w_sim),
        "w_momentum": float(w_momentum),
        "w_popularity": float(w_popularity),
        "w_freshness": float(w_freshness),
    }

    try:
        candidates, dbg = hybrid.recommend_similar_hybrid(q, weights=weights)
    except TypeError:
        candidates, dbg = hybrid.recommend_similar_hybrid(q, **weights)

    payload = {
        "seed_track_id": seed_track_id,
        "k": k,
        "returned": len(candidates),
        "query": _as_jsonable(q),
        "weights": weights,
        "results": _as_jsonable(candidates),
    }
    if debug:
        payload["debug"] = _as_jsonable(dbg)
    return payload


# --------------------------------------------------------------------------------------
# Playlist (MMR)
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
    df, _, _, id_to_idx = _load_catalog()
    if seed_track_id not in id_to_idx:
        raise HTTPException(status_code=404, detail=f"seed_track_id not found: {seed_track_id}")

    gen = _get_playlist_generator()

    weights = {
        "w_sim": float(w_sim),
        "w_momentum": float(w_momentum),
        "w_popularity": float(w_popularity),
        "w_freshness": float(w_freshness),
    }

    # Build query object if your playlist module provides it
    try:
        from musicrec.playlist import PlaylistQuery  # type: ignore

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
    except Exception:
        q = {
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
        }

    try:
        playlist, dbg = gen.generate(q, weights=weights)
    except TypeError:
        playlist, dbg = gen.generate(q)

    payload = {
        "seed_track_id": seed_track_id,
        "n_tracks": n_tracks,
        "returned": len(playlist),
        "query": _as_jsonable(q),
        "weights": weights,
        "playlist": _as_jsonable(playlist),
    }
    if debug:
        payload["debug"] = _as_jsonable(dbg)
    return payload


# --------------------------------------------------------------------------------------
# Session Events + For You (API)
# --------------------------------------------------------------------------------------

@app.post("/session/event")
def session_event(payload: SessionEventIn = Body(...)) -> Dict[str, Any]:
    """
    Adds a session event and returns:
      { ok: true, session_id: ..., events_count: <int> }

    Your failing test expects 'events_count' to be present.
    """
    # validate track exists (keeps your tests/data consistent)
    _, _, _, id_to_idx = _load_catalog()
    if payload.track_id not in id_to_idx:
        raise HTTPException(status_code=404, detail=f"track_id not found: {payload.track_id}")

    store = _get_session_store()

    try:
        _store_add_event(store, payload.session_id, payload.track_id, payload.event_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record session event: {e}")

    # IMPORTANT FIX: include events_count
    events = _store_get_events(store, payload.session_id)
    events_count = len(events)

    return {
        "ok": True,
        "session_id": payload.session_id,
        "events_count": events_count,
    }


@app.get("/for_you/from_session")
def for_you_from_session(
    session_id: str,
    n: int = 25,
    candidate_k: int = 1200,
    same_country_only: bool = False,
    country: Optional[str] = None,
    explicit_ok: bool = True,
    unique_artist: bool = True,
    max_per_genre: int = 10,
    lambda_relevance: float = 0.75,
    debug: bool = False,
) -> Dict[str, Any]:
    store = _get_session_store()
    events = _store_get_events(store, session_id)
    if not events:
        raise HTTPException(status_code=404, detail=f"No events found for session_id: {session_id}")

    # Prefer your V1.3.0 ForYouFeed if available; else do a simple internal fallback
    feed = _get_for_you()

    if feed is not None:
        # Try to call your feed method (names can vary)
        try:
            results, dbg = feed.for_you_from_session(
                session_id=session_id,
                events=events,
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
        except Exception:
            # common alternate signature: (session_id, events, ...)
            try:
                results, dbg = feed.for_you_from_session(session_id, events, n=n, candidate_k=candidate_k, debug=debug)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"ForYouFeed call failed: {e}")

        payload = {
            "session_id": session_id,
            "events_count": len(events),
            "query": {
                "session_id": session_id,
                "n": n,
                "candidate_k": candidate_k,
                "same_country_only": same_country_only,
                "country": country,
                "explicit_ok": explicit_ok,
                "unique_artist": unique_artist,
                "max_per_genre": max_per_genre,
                "lambda_relevance": lambda_relevance,
                "debug": debug,
            },
            "results": _as_jsonable(results),
        }
        if debug:
            payload["debug"] = _as_jsonable(dbg)
        return payload

    # Fallback (should rarely be used if your for_you module exists)
    raise HTTPException(status_code=500, detail="ForYouFeed not available; for_you module failed to load")