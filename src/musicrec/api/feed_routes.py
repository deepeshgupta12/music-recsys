from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from musicrec.feeds import FeedQuery, SegmentFeeds

router = APIRouter()


@lru_cache(maxsize=1)
def _load_feature_table() -> pd.DataFrame:
    try:
        ft = pd.read_parquet("data/processed/catalog_features.parquet")
    except Exception as e:
        raise RuntimeError(f"Failed to load catalog_features.parquet: {e}")

    if "track_id" not in ft.columns:
        raise RuntimeError("catalog_features.parquet missing required column: track_id")

    # Ensure strings are sane
    for col in ["track_id", "track_name", "artist_name", "country", "genre"]:
        if col in ft.columns:
            ft[col] = ft[col].astype("string").fillna("")

    return ft


@lru_cache(maxsize=1)
def _get_feeds_engine() -> SegmentFeeds:
    ft = _load_feature_table()
    return SegmentFeeds(ft)


def _require_country(country: str) -> str:
    c = (country or "").strip()
    if not c:
        raise HTTPException(status_code=400, detail="country is required")
    return c


@router.get("/feed/home")
def feed_home(
    country: str = Query(..., description="Country code, e.g. IN"),
    n: int = Query(25, ge=1, le=200),
    explicit_ok: bool = Query(True),
    debug: bool = Query(False),
) -> Dict[str, Any]:
    c = _require_country(country)

    feeds = _get_feeds_engine()
    q = FeedQuery(country=c, n=n, explicit_ok=explicit_ok, debug=debug)
    sections, dbg = feeds.home_feed(q)

    body: Dict[str, Any] = {
        "ok": True,
        "country": c,
        "n": n,
        "sections": sections,
    }
    if debug:
        body["debug"] = dbg
    return body


@router.get("/feed/genre")
def feed_genre(
    country: str = Query(..., description="Country code, e.g. IN"),
    genre: str = Query(..., description="Genre name, e.g. Pop"),
    n: int = Query(25, ge=1, le=200),
    explicit_ok: bool = Query(True),
    debug: bool = Query(False),
) -> Dict[str, Any]:
    c = _require_country(country)
    g = (genre or "").strip()
    if not g:
        raise HTTPException(status_code=400, detail="genre is required")

    feeds = _get_feeds_engine()
    q = FeedQuery(country=c, genre=g, n=n, explicit_ok=explicit_ok, debug=debug)
    sections, dbg = feeds.genre_feed(q)

    body: Dict[str, Any] = {
        "ok": True,
        "country": c,
        "genre": g,
        "n": n,
        "sections": sections,
    }
    if debug:
        body["debug"] = dbg
    return body