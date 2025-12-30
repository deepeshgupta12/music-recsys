from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from musicrec.api.feedback_routes import get_feedback_store
from musicrec.feeds import FeedQuery, SegmentFeeds
from musicrec.storage.feedback_store import FeedbackStore
from musicrec.storage.feature_table import load_feature_table

router = APIRouter(tags=["feeds"])


@lru_cache(maxsize=1)
def _get_feeds_engine() -> SegmentFeeds:
    ft = load_feature_table()
    return SegmentFeeds(ft)


def _clean_user_id(x_user_id: Optional[str]) -> Optional[str]:
    if x_user_id is None:
        return None
    u = x_user_id.strip()
    return u if u else None


def _apply_suppression(
    sections: Dict[str, list],
    suppressed_ids: set[str],
) -> Dict[str, list]:
    if not suppressed_ids:
        return sections

    out: Dict[str, list] = {}
    for k, items in sections.items():
        if not isinstance(items, list):
            out[k] = items
            continue
        out[k] = [it for it in items if isinstance(it, dict) and it.get("track_id") not in suppressed_ids]
    return out


@router.get("/feed/home")
def feed_home(
    country: str = Query(...),
    n: int = Query(default=10, ge=1, le=200),
    explicit_ok: bool = Query(default=False),
    debug: bool = Query(default=False),
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    store: FeedbackStore = Depends(get_feedback_store),
) -> Dict[str, Any]:
    feeds = _get_feeds_engine()

    q = FeedQuery(country=country, genre=None, n=n, explicit_ok=explicit_ok, debug=debug)

    sections, dbg = feeds.home_feed(q)

    user_id = _clean_user_id(x_user_id)
    if user_id:
        suppressed = store.suppressed_track_ids(user_id=user_id, event_types=("dislike", "skip"), days=365)
        sections = _apply_suppression(sections, suppressed)

    body: Dict[str, Any] = {"ok": True, "country": country, "n": int(n), "sections": sections}

    if debug:
        d = dict(dbg or {})
        d.setdefault("fallback_used", {})
        d.setdefault("sections_returned", {k: int(len(v)) for k, v in sections.items() if isinstance(v, list)})
        body["debug"] = d

    return body


@router.get("/feed/genre")
def feed_genre(
    country: str = Query(...),
    genre: str = Query(...),
    n: int = Query(default=10, ge=1, le=200),
    explicit_ok: bool = Query(default=False),
    debug: bool = Query(default=False),
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    store: FeedbackStore = Depends(get_feedback_store),
) -> Dict[str, Any]:
    feeds = _get_feeds_engine()

    if not genre or not genre.strip():
        raise HTTPException(status_code=400, detail="genre is required")

    q = FeedQuery(country=country, genre=genre.strip(), n=n, explicit_ok=explicit_ok, debug=debug)

    sections, dbg = feeds.genre_feed(q)

    user_id = _clean_user_id(x_user_id)
    if user_id:
        suppressed = store.suppressed_track_ids(user_id=user_id, event_types=("dislike", "skip"), days=365)
        sections = _apply_suppression(sections, suppressed)

    body: Dict[str, Any] = {
        "ok": True,
        "country": country,
        "genre": genre.strip(),
        "n": int(n),
        "sections": sections,
    }

    if debug:
        d = dict(dbg or {})
        d.setdefault("fallback_used", {})
        d.setdefault("sections_returned", {k: int(len(v)) for k, v in sections.items() if isinstance(v, list)})
        body["debug"] = d

    return body