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


def _section_counts(sections: Dict[str, Any]) -> Dict[str, int]:
    """
    Count list lengths per section key.
    Non-list values are ignored (but preserved in response).
    """
    return {k: int(len(v)) for k, v in sections.items() if isinstance(v, list)}


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

    # Optional user-based suppression (dislike/skip)
    user_id = _clean_user_id(x_user_id)
    suppression_dbg: Dict[str, Any] = {}
    if user_id:
        suppressed = store.suppressed_track_ids(user_id=user_id, event_types=("dislike", "skip"), days=365)

        before_counts: Dict[str, int] = {}
        if debug:
            before_counts = _section_counts(sections)

        sections = _apply_suppression(sections, suppressed)

        if debug:
            after_counts = _section_counts(sections)
            removed_by_section = {k: max(0, before_counts.get(k, 0) - after_counts.get(k, 0)) for k in before_counts}
            removed_total = int(sum(removed_by_section.values()))
            suppression_dbg = {
                "suppressed_event_types": ["dislike", "skip"],
                "suppressed_ids_n": int(len(suppressed)),
                # This is the key your next tests can rely on:
                "disliked_suppressed_count": removed_total,
                "suppressed_removed_by_section": removed_by_section,
            }

    body: Dict[str, Any] = {"ok": True, "country": country, "n": int(n), "sections": sections}

    if debug:
        d = dict(dbg or {})
        d.setdefault("fallback_used", {})
        d.setdefault("sections_returned", _section_counts(sections))
        # Always include the key (empty dict when no user_id or no removals)
        if "disliked_suppressed_count" not in d:
            d["disliked_suppressed_count"] = int(suppression_dbg.get("disliked_suppressed_count", 0))
        if "suppressed_removed_by_section" not in d:
            d["suppressed_removed_by_section"] = suppression_dbg.get("suppressed_removed_by_section", {})
        if "suppressed_ids_n" not in d:
            d["suppressed_ids_n"] = int(suppression_dbg.get("suppressed_ids_n", 0))
        if "suppressed_event_types" not in d:
            d["suppressed_event_types"] = suppression_dbg.get("suppressed_event_types", ["dislike", "skip"])
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

    genre_clean = genre.strip()
    q = FeedQuery(country=country, genre=genre_clean, n=n, explicit_ok=explicit_ok, debug=debug)

    sections, dbg = feeds.genre_feed(q)

    # Optional user-based suppression (dislike/skip)
    user_id = _clean_user_id(x_user_id)
    suppression_dbg: Dict[str, Any] = {}
    if user_id:
        suppressed = store.suppressed_track_ids(user_id=user_id, event_types=("dislike", "skip"), days=365)

        before_counts: Dict[str, int] = {}
        if debug:
            before_counts = _section_counts(sections)

        sections = _apply_suppression(sections, suppressed)

        if debug:
            after_counts = _section_counts(sections)
            removed_by_section = {k: max(0, before_counts.get(k, 0) - after_counts.get(k, 0)) for k in before_counts}
            removed_total = int(sum(removed_by_section.values()))
            suppression_dbg = {
                "suppressed_event_types": ["dislike", "skip"],
                "suppressed_ids_n": int(len(suppressed)),
                "disliked_suppressed_count": removed_total,
                "suppressed_removed_by_section": removed_by_section,
            }

    body: Dict[str, Any] = {
        "ok": True,
        "country": country,
        "genre": genre_clean,
        "n": int(n),
        "sections": sections,
    }

    if debug:
        d = dict(dbg or {})
        d.setdefault("fallback_used", {})
        d.setdefault("sections_returned", _section_counts(sections))
        if "disliked_suppressed_count" not in d:
            d["disliked_suppressed_count"] = int(suppression_dbg.get("disliked_suppressed_count", 0))
        if "suppressed_removed_by_section" not in d:
            d["suppressed_removed_by_section"] = suppression_dbg.get("suppressed_removed_by_section", {})
        if "suppressed_ids_n" not in d:
            d["suppressed_ids_n"] = int(suppression_dbg.get("suppressed_ids_n", 0))
        if "suppressed_event_types" not in d:
            d["suppressed_event_types"] = suppression_dbg.get("suppressed_event_types", ["dislike", "skip"])
        body["debug"] = d

    return body