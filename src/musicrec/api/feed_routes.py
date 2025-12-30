from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

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


def _require_user_id(x_user_id: Optional[str]) -> str:
    u = _clean_user_id(x_user_id)
    if not u:
        raise HTTPException(status_code=400, detail="X-User-Id header required")
    return u


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


def _suppression_removed_counts(before: Dict[str, list], after: Dict[str, list]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for k, b in before.items():
        if not isinstance(b, list):
            continue
        a = after.get(k, [])
        if not isinstance(a, list):
            continue
        out[k] = max(0, int(len(b)) - int(len(a)))
    return out


def _latest_feedback_by_track(events: list[dict]) -> Dict[str, str]:
    """
    events are expected to be 'recent' (desc ts). We treat first-seen per track_id as latest.
    """
    latest: Dict[str, str] = {}
    for e in events:
        if not isinstance(e, dict):
            continue
        tid = (e.get("track_id") or "").strip()
        if not tid or tid in latest:
            continue
        et = (e.get("event_type") or "").strip().lower()
        if not et:
            continue
        latest[tid] = et
    return latest


def _apply_personalization(
    sections: Dict[str, list],
    latest_event: Dict[str, str],
) -> Tuple[Dict[str, list], Dict[str, int]]:
    """
    Boost items that the user has positively interacted with recently.
      - like: strongest boost
      - play: small boost
    Returns (new_sections, boosted_counts)
    """
    if not latest_event:
        return sections, {"like": 0, "play": 0}

    weight = {"like": 2, "play": 1}
    boosted_like = 0
    boosted_play = 0

    out: Dict[str, list] = {}
    for k, items in sections.items():
        if not isinstance(items, list):
            out[k] = items
            continue

        scored = []
        for idx, it in enumerate(items):
            if not isinstance(it, dict):
                scored.append((0, 0.0, idx, it))
                continue
            tid = (it.get("track_id") or "").strip()
            et = latest_event.get(tid, "")
            w = weight.get(et, 0)
            if et == "like":
                boosted_like += 1
            elif et == "play":
                boosted_play += 1

            base_score = 0.0
            try:
                base_score = float(it.get("score", 0.0) or 0.0)
            except Exception:
                base_score = 0.0

            scored.append((w, base_score, idx, it))

        # sort: boost desc, then score desc, then stable by original idx
        scored.sort(key=lambda t: (t[0], t[1], -t[2]), reverse=True)
        out[k] = [t[3] for t in scored]

    return out, {"like": int(boosted_like), "play": int(boosted_play)}


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
    suppressed = set()
    removed_by_section: Dict[str, int] = {}
    if user_id:
        suppressed = store.suppressed_track_ids(user_id=user_id, event_types=("dislike", "skip"), days=365)
        before = sections
        sections = _apply_suppression(sections, suppressed)
        removed_by_section = _suppression_removed_counts(before, sections)

    body: Dict[str, Any] = {"ok": True, "country": country, "n": int(n), "sections": sections}

    if debug:
        d = dict(dbg or {})
        d.setdefault("fallback_used", {})
        d.setdefault("sections_returned", {k: int(len(v)) for k, v in sections.items() if isinstance(v, list)})

        if user_id:
            d["disliked_suppressed_count"] = int(len(suppressed))
            d["suppressed_removed_by_section"] = removed_by_section
            d["suppressed_ids_n"] = int(len(suppressed))
            d["suppressed_event_types"] = ["dislike", "skip"]

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
    suppressed = set()
    removed_by_section: Dict[str, int] = {}
    if user_id:
        suppressed = store.suppressed_track_ids(user_id=user_id, event_types=("dislike", "skip"), days=365)
        before = sections
        sections = _apply_suppression(sections, suppressed)
        removed_by_section = _suppression_removed_counts(before, sections)

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

        if user_id:
            d["disliked_suppressed_count"] = int(len(suppressed))
            d["suppressed_removed_by_section"] = removed_by_section
            d["suppressed_ids_n"] = int(len(suppressed))
            d["suppressed_event_types"] = ["dislike", "skip"]

        body["debug"] = d

    return body


@router.get("/feed/for-you")
def feed_for_you(
    country: str = Query(...),
    n: int = Query(default=10, ge=1, le=200),
    explicit_ok: bool = Query(default=False),
    debug: bool = Query(default=False),
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    store: FeedbackStore = Depends(get_feedback_store),
) -> Dict[str, Any]:
    """
    V1.5.6 Step 1.1: A lightweight personalized feed.
      - uses home feed as candidate generator
      - suppresses disliked/skipped tracks for the user
      - re-orders rails with small boosts from recent feedback (like/play)
    """
    user_id = _require_user_id(x_user_id)
    feeds = _get_feeds_engine()

    q = FeedQuery(country=country, genre=None, n=n, explicit_ok=explicit_ok, debug=debug)
    sections, dbg = feeds.home_feed(q)

    # suppression (dislike/skip)
    suppressed = store.suppressed_track_ids(user_id=user_id, event_types=("dislike", "skip"), days=365)
    before_supp = sections
    sections = _apply_suppression(sections, suppressed)
    removed_by_section = _suppression_removed_counts(before_supp, sections)

    # personalization (like/play boosts)
    recent = store.recent_events(user_id=user_id, limit=500)
    latest = _latest_feedback_by_track(recent)
    before_personalize = sections
    sections, boosted = _apply_personalization(sections, latest)

    body: Dict[str, Any] = {"ok": True, "country": country, "n": int(n), "sections": sections}

    if debug:
        d = dict(dbg or {})
        d.setdefault("fallback_used", {})
        d.setdefault("sections_returned", {k: int(len(v)) for k, v in sections.items() if isinstance(v, list)})

        d["disliked_suppressed_count"] = int(len(suppressed))
        d["suppressed_removed_by_section"] = removed_by_section
        d["suppressed_ids_n"] = int(len(suppressed))
        d["suppressed_event_types"] = ["dislike", "skip"]

        d["personalization"] = {
            "recent_events_considered": int(len(recent)),
            "latest_track_events_n": int(len(latest)),
            "boosted_counts": {"like": int(boosted.get("like", 0)), "play": int(boosted.get("play", 0))},
        }

        # safety: if personalization changes lengths (it shouldn't), show deltas
        deltas = _suppression_removed_counts(before_personalize, sections)
        d["personalization_reorder_only_length_deltas"] = deltas

        body["debug"] = d

    return body