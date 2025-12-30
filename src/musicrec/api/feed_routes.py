from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from musicrec.api.feedback_routes import get_feedback_store
from musicrec.feeds import FeedQuery, SegmentFeeds
from musicrec.session_store import SessionEvent
from musicrec.storage.feedback_store import FeedbackStore
from musicrec.storage.feature_table import load_feature_table
from musicrec.for_you import (
    ForYouQuery,
    build_for_you_recommender_from_feature_table,
    for_you_item_to_api_dict,
)

router = APIRouter(tags=["feeds"])


@lru_cache(maxsize=1)
def _get_feeds_engine() -> SegmentFeeds:
    ft = load_feature_table()
    return SegmentFeeds(ft)


@lru_cache(maxsize=1)
def _get_for_you_engine():
    ft = load_feature_table()
    return build_for_you_recommender_from_feature_table(ft)


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


def _ts_to_iso(ts: Any) -> str:
    """
    feedback_store may keep float epoch seconds or ISO strings.
    IMPORTANT: SessionEvent/session_store uses datetime.fromisoformat(),
    which requires '+00:00' (not trailing 'Z') in Python 3.10.
    """
    try:
        if ts is None:
            return "1970-01-01T00:00:00+00:00"

        # epoch seconds -> ISO with +00:00
        if isinstance(ts, (int, float)):
            import datetime as _dt

            return _dt.datetime.fromtimestamp(float(ts), tz=_dt.timezone.utc).isoformat()

        s = str(ts).strip()
        if not s:
            return "1970-01-01T00:00:00+00:00"

        # normalize trailing Z -> +00:00 for fromisoformat compatibility
        if s.endswith("Z"):
            return s[:-1] + "+00:00"

        return s
    except Exception:
        return "1970-01-01T00:00:00+00:00"


def _dedup_other_sections_against_for_you(sections: Dict[str, list], for_you_ids: set[str]) -> Tuple[Dict[str, list], Dict[str, int]]:
    """
    Keep 'for_you' rail intact, remove its track_ids from other list rails.
    Returns (new_sections, removed_counts).
    """
    removed: Dict[str, int] = {}
    out: Dict[str, list] = {}
    for k, items in sections.items():
        if k == "for_you":
            out[k] = items
            continue
        if not isinstance(items, list):
            out[k] = items
            continue
        before_n = len(items)
        filtered = [it for it in items if not (isinstance(it, dict) and (it.get("track_id") in for_you_ids))]
        out[k] = filtered
        removed[k] = max(0, before_n - len(filtered))
    if "for_you" not in removed:
        removed["for_you"] = 0
    return out, removed


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
    n: int = Query(default=10, ge=5, le=200),
    explicit_ok: bool = Query(default=False),
    debug: bool = Query(default=False),
    # personalization knobs
    candidate_k: int = Query(default=1200, ge=300, le=20000),
    same_country_only: bool = Query(default=True),
    unique_artist: bool = Query(default=True),
    max_per_genre: int = Query(default=10, ge=0, le=200),
    lambda_relevance: float = Query(default=0.75, ge=0.0, le=1.0),
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    store: FeedbackStore = Depends(get_feedback_store),
) -> Dict[str, Any]:
    """
    V1.5.6 Step 1.2: Real ForYou engine using session feedback events as taste signals.
    Always returns a 'for_you' rail (may be empty, with fallback reason in debug).
    """
    user_id = _require_user_id(x_user_id)
    feeds = _get_feeds_engine()

    # Base candidate rails (kept for UX + fallback)
    q = FeedQuery(country=country, genre=None, n=n, explicit_ok=explicit_ok, debug=debug)
    sections, dbg = feeds.home_feed(q)

    # Suppression (dislike/skip) applies to everything including for_you results
    suppressed = store.suppressed_track_ids(user_id=user_id, event_types=("dislike", "skip"), days=365)
    before_supp = sections
    sections = _apply_suppression(sections, suppressed)
    removed_by_section = _suppression_removed_counts(before_supp, sections)

    # Build events for ForYou from recent feedback
    recent = store.recent_events(user_id=user_id, limit=500)
    events: list[SessionEvent] = []
    for e in recent:
        if not isinstance(e, dict):
            continue
        tid = (e.get("track_id") or "").strip()
        et = (e.get("event_type") or "").strip().lower()
        if not tid or not et:
            continue

        # IMPORTANT: SessionEvent requires session_id
        events.append(
            SessionEvent(
                session_id=user_id,
                track_id=tid,
                event_type=et,
                ts=_ts_to_iso(e.get("ts")),
            )
        )

    # Run ForYou
    for_you_items = []
    for_you_dbg: Dict[str, object] = {}
    fallback_reason: Optional[str] = None

    if len(events) == 0:
        fallback_reason = "no_events"
    else:
        engine = _get_for_you_engine()
        fq = ForYouQuery(
            session_id=user_id,
            n=int(n),
            candidate_k=int(candidate_k),
            same_country_only=bool(same_country_only),
            country=country if same_country_only else None,
            explicit_ok=bool(explicit_ok),
            unique_artist=bool(unique_artist),
            max_per_genre=int(max_per_genre),
            lambda_relevance=float(lambda_relevance),
            debug=bool(debug),
        )
        try:
            recs, fydbg = engine.recommend(fq, events)
            for_you_items = [for_you_item_to_api_dict(it) for it in recs]
            for_you_dbg = dict(fydbg or {})
            if len(for_you_items) == 0:
                fallback_reason = "no_candidates"
        except Exception:
            fallback_reason = "engine_error"

    # Apply suppression to for_you rail too
    if suppressed and for_you_items:
        for_you_items = [it for it in for_you_items if it.get("track_id") not in suppressed]

    # Always include for_you rail
    sections = dict(sections)
    sections["for_you"] = for_you_items[: int(n)]

    # Cross-rail de-dup: remove for_you track_ids from other rails
    for_you_ids = {str(it.get("track_id")) for it in sections["for_you"] if isinstance(it, dict) and it.get("track_id")}
    sections, cross_removed = _dedup_other_sections_against_for_you(sections, for_you_ids)

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
            "session_events_used": int(len(events)),
            "for_you_fallback_reason": fallback_reason,
            "for_you_debug": for_you_dbg,
            "dedup_removed_against_for_you": cross_removed,
        }

        body["debug"] = d

    return body