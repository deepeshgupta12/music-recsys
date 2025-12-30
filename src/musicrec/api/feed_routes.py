from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Header, Query

from musicrec.feeds import FeedQuery, SegmentFeeds
from musicrec.api.feedback_routes import get_feedback_store
from musicrec.storage.feedback_store import FeedbackStore
from musicrec.storage.feature_table import load_feature_table

router = APIRouter()


@lru_cache(maxsize=1)
def _get_feeds_engine() -> SegmentFeeds:
    ft = load_feature_table()
    return SegmentFeeds(ft)


def _apply_suppression(
    sections: Dict[str, Any],
    suppressed: set[str],
) -> Dict[str, Any]:
    if not suppressed:
        return sections

    out: Dict[str, Any] = {}
    for name, items in sections.items():
        if not isinstance(items, list):
            out[name] = items
            continue
        out[name] = [it for it in items if str(it.get("track_id", "")) not in suppressed]
    return out


@router.get("/feed/home")
def feed_home(
    country: str = Query(..., min_length=1),
    n: int = Query(10, ge=1, le=100),
    debug: bool = Query(False),
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    store: FeedbackStore = Depends(get_feedback_store),
) -> Dict[str, Any]:
    feeds = _get_feeds_engine()

    q = FeedQuery(country=country, n=n)
    sections, dbg = feeds.home_feed(q)

    if x_user_id and x_user_id.strip():
        suppressed = store.suppressed_track_ids(user_id=x_user_id.strip())
        sections = _apply_suppression(sections, suppressed)
        if debug:
            dbg = dict(dbg)
            dbg["personalization"] = {"suppressed_n": len(suppressed)}

    resp: Dict[str, Any] = {"ok": True, "query": {"country": country, "n": n}, "sections": sections}
    if debug:
        resp["debug"] = dbg
    return resp


@router.get("/feed/genre")
def feed_genre(
    country: str = Query(..., min_length=1),
    genre: str = Query(..., min_length=1),
    n: int = Query(10, ge=1, le=100),
    debug: bool = Query(False),
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    store: FeedbackStore = Depends(get_feedback_store),
) -> Dict[str, Any]:
    feeds = _get_feeds_engine()

    q = FeedQuery(country=country, genre=genre, n=n)
    sections, dbg = feeds.genre_feed(q)

    if x_user_id and x_user_id.strip():
        suppressed = store.suppressed_track_ids(user_id=x_user_id.strip())
        sections = _apply_suppression(sections, suppressed)
        if debug:
            dbg = dict(dbg)
            dbg["personalization"] = {"suppressed_n": len(suppressed)}

    resp: Dict[str, Any] = {"ok": True, "query": {"country": country, "genre": genre, "n": n}, "sections": sections}
    if debug:
        resp["debug"] = dbg
    return resp