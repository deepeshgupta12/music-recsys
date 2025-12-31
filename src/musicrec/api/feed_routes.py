from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from musicrec.api.feedback_routes import get_feedback_store
from musicrec.feeds import FeedQuery, SegmentFeeds
from musicrec.for_you import (
    ForYouQuery,
    build_for_you_recommender_from_feature_table,
    for_you_item_to_api_dict,
)
from musicrec.personalization import PersonalizationConfig, reorder_only_personalize_sections
from musicrec.session_store import SessionEvent
from musicrec.storage.feature_table import load_feature_table
from musicrec.storage.feedback_store import FeedbackStore
from musicrec.storage.tag_store import TagStore, TagStoreConfig  # <-- Step 3.1: import config

router = APIRouter(tags=["feeds"])


@lru_cache(maxsize=1)
def _get_feeds_engine() -> SegmentFeeds:
    ft = load_feature_table()
    return SegmentFeeds(ft)


@lru_cache(maxsize=1)
def _get_for_you_engine():
    ft = load_feature_table()
    return build_for_you_recommender_from_feature_table(ft)


@lru_cache(maxsize=1)
def _get_tag_store() -> TagStore:
    # Step 3.1: align with tag_routes.py (same config/path/table)
    return TagStore(TagStoreConfig())


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
        out[k] = [
            it
            for it in items
            if isinstance(it, dict) and it.get("track_id") not in suppressed_ids
        ]
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

        if isinstance(ts, (int, float)):
            import datetime as _dt

            return _dt.datetime.fromtimestamp(float(ts), tz=_dt.timezone.utc).isoformat()

        s = str(ts).strip()
        if not s:
            return "1970-01-01T00:00:00+00:00"

        if s.endswith("Z"):
            return s[:-1] + "+00:00"

        return s
    except Exception:
        return "1970-01-01T00:00:00+00:00"


def _dedup_other_sections_against_for_you(
    sections: Dict[str, list],
    for_you_ids: set[str],
) -> Tuple[Dict[str, list], Dict[str, int]]:
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
        filtered = [
            it
            for it in items
            if not (isinstance(it, dict) and (it.get("track_id") in for_you_ids))
        ]
        out[k] = filtered
        removed[k] = max(0, before_n - len(filtered))
    if "for_you" not in removed:
        removed["for_you"] = 0
    return out, removed


def _collect_track_ids(sections: Dict[str, list]) -> Tuple[set[str], int]:
    """
    Returns:
      - unique track_ids across all list rails (dict items only)
      - total_items_seen across all list rails (dict items only)
    """
    ids: set[str] = set()
    total = 0
    for items in sections.values():
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            total += 1
            tid = (it.get("track_id") or "").strip()
            if tid:
                ids.add(tid)
    return ids, total


def _safe_row_to_payload(row: Any) -> Tuple[Dict[str, Any], Optional[str], float]:
    """
    Normalize TagStore rows into:
      - tags dict
      - provider str|None
      - updated_at float
    Supports both dataclass-like objects and dict payloads.
    """
    if row is None:
        return {}, None, 0.0

    if isinstance(row, dict):
        tags = row.get("tags") or {}
        provider = row.get("provider")
        updated_at = row.get("updated_at") or 0.0
        try:
            return dict(tags), (str(provider) if provider is not None else None), float(updated_at)
        except Exception:
            return dict(tags), (str(provider) if provider is not None else None), 0.0

    tags = getattr(row, "tags", {}) or {}
    provider = getattr(row, "provider", None)
    updated_at = getattr(row, "updated_at", 0.0) or 0.0
    try:
        return dict(tags), (str(provider) if provider is not None else None), float(updated_at)
    except Exception:
        return dict(tags), (str(provider) if provider is not None else None), 0.0


def _join_tags_into_sections(
    sections: Dict[str, list],
    store: TagStore,
    debug: bool,
) -> Tuple[Dict[str, list], Dict[str, Any]]:
    """
    Adds these fields into every dict item that has a track_id:
      - tags: dict
      - tags_missing: bool
      - tags_provider: str|None
      - tags_updated_at: float

    Debug contract:
      tracks_requested = number of UNIQUE track_ids across all rails (dict items)
      total_items_seen = number of dict items scanned across all rails
    """
    unique_ids, total_items_seen = _collect_track_ids(sections)
    tracks_requested = len(unique_ids)

    found_map: Dict[str, Any] = {}
    if tracks_requested > 0:
        found_map = store.batch_get(sorted(unique_ids))  # type: ignore[attr-defined]

    unique_found = len(found_map)

    tagged_items = 0
    missing_items = 0

    out: Dict[str, list] = {}
    for k, items in sections.items():
        if not isinstance(items, list):
            out[k] = items
            continue

        new_items: list = []
        for it in items:
            if not isinstance(it, dict):
                new_items.append(it)
                continue

            tid = (it.get("track_id") or "").strip()
            if not tid:
                new_items.append(it)
                continue

            row = found_map.get(tid)
            tags, provider, updated_at = _safe_row_to_payload(row)

            missing = row is None
            if missing:
                missing_items += 1
            else:
                tagged_items += 1

            new_it = dict(it)
            new_it["tags"] = tags
            new_it["tags_missing"] = bool(missing)
            new_it["tags_provider"] = provider
            new_it["tags_updated_at"] = float(updated_at)
            new_items.append(new_it)

        out[k] = new_items

    dbg = {
        "tracks_requested": int(tracks_requested),
        "unique_found": int(unique_found),
        "total_items_seen": int(total_items_seen),
        "tagged_items": int(tagged_items),
        "missing_items": int(missing_items),
    }
    return out, (dbg if debug else {})


# ---------- Step 3 helpers (mood feed) ----------

def _slugify(s: str) -> str:
    s = (s or "").strip().lower()
    if not s:
        return ""
    out: List[str] = []
    prev_dash = False
    for ch in s:
        is_alnum = ("a" <= ch <= "z") or ("0" <= ch <= "9")
        if is_alnum:
            out.append(ch)
            prev_dash = False
        else:
            if not prev_dash:
                out.append("-")
                prev_dash = True
    slug = "".join(out).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug


def _item_has_mood(it: Dict[str, Any], mood_slug: str) -> bool:
    """
    Matches mood_slug against tags["moods"/"mood"/"scenes"/"scene"].
    Supports str or list[str] for each.
    """
    tags = it.get("tags")
    if not isinstance(tags, dict):
        return False

    keys = ["moods", "mood", "scenes", "scene"]
    vals: List[str] = []
    for k in keys:
        v = tags.get(k)
        if v is None:
            continue
        if isinstance(v, str) and v.strip():
            vals.append(v.strip())
        elif isinstance(v, list):
            for x in v:
                if isinstance(x, str) and x.strip():
                    vals.append(x.strip())

    for v in vals:
        if _slugify(v) == mood_slug:
            return True
    return False


def _strip_tag_fields_from_items(items: List[Any]) -> List[Any]:
    drop = {"tags", "tags_missing", "tags_provider", "tags_updated_at"}
    out: List[Any] = []
    for it in items:
        if not isinstance(it, dict):
            out.append(it)
            continue
        d = dict(it)
        for k in drop:
            d.pop(k, None)
        out.append(d)
    return out


@router.get("/feed/home")
def feed_home(
    country: str = Query(...),
    n: int = Query(default=10, ge=1, le=200),
    explicit_ok: bool = Query(default=False),
    debug: bool = Query(default=False),
    include_tags: bool = Query(default=False),
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    store: FeedbackStore = Depends(get_feedback_store),
) -> Dict[str, Any]:
    feeds = _get_feeds_engine()

    q = FeedQuery(country=country, genre=None, n=n, explicit_ok=explicit_ok, debug=debug)
    sections, dbg = feeds.home_feed(q)

    user_id = _clean_user_id(x_user_id)
    suppressed: set[str] = set()
    removed_by_section: Dict[str, int] = {}
    personalization_dbg: Dict[str, Any] = {}

    # 1) Suppression (dislike/skip)
    if user_id:
        suppressed = store.suppressed_track_ids(
            user_id=user_id,
            event_types=("dislike", "skip"),
            days=365,
        )
        before = sections
        sections = _apply_suppression(sections, suppressed)
        removed_by_section = _suppression_removed_counts(before, sections)

        # 2) reorder-only personalization (does not change sets; only order)
        recent = store.recent_events(user_id=user_id, limit=500)
        cfg = PersonalizationConfig()
        sections, personalization_dbg = reorder_only_personalize_sections(
            sections=sections,
            recent_events=recent,
            user_id=user_id,
            debug=bool(debug),
            config=cfg,
        )

    tags_join_dbg: Dict[str, Any] = {}
    if include_tags:
        tag_store = _get_tag_store()
        sections, tags_join_dbg = _join_tags_into_sections(sections, tag_store, debug=bool(debug))

    body: Dict[str, Any] = {"ok": True, "country": country, "n": int(n), "sections": sections}

    if debug:
        d = dict(dbg or {})
        d.setdefault("fallback_used", {})
        d.setdefault(
            "sections_returned",
            {k: int(len(v)) for k, v in sections.items() if isinstance(v, list)},
        )

        if user_id:
            d["disliked_suppressed_count"] = int(len(suppressed))
            d["suppressed_removed_by_section"] = removed_by_section
            d["suppressed_ids_n"] = int(len(suppressed))
            d["suppressed_event_types"] = ["dislike", "skip"]
            d["personalization"] = personalization_dbg

        if include_tags:
            d["tags_join"] = tags_join_dbg

        body["debug"] = d

    return body


@router.get("/feed/genre")
def feed_genre(
    country: str = Query(...),
    genre: str = Query(...),
    n: int = Query(default=10, ge=1, le=200),
    explicit_ok: bool = Query(default=False),
    debug: bool = Query(default=False),
    include_tags: bool = Query(default=False),
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    store: FeedbackStore = Depends(get_feedback_store),
) -> Dict[str, Any]:
    feeds = _get_feeds_engine()

    if not genre or not genre.strip():
        raise HTTPException(status_code=400, detail="genre is required")

    q = FeedQuery(country=country, genre=genre.strip(), n=n, explicit_ok=explicit_ok, debug=debug)
    sections, dbg = feeds.genre_feed(q)

    user_id = _clean_user_id(x_user_id)
    suppressed: set[str] = set()
    removed_by_section: Dict[str, int] = {}
    personalization_dbg: Dict[str, Any] = {}

    if user_id:
        suppressed = store.suppressed_track_ids(user_id=user_id, event_types=("dislike", "skip"), days=365)
        before = sections
        sections = _apply_suppression(sections, suppressed)
        removed_by_section = _suppression_removed_counts(before, sections)

        recent = store.recent_events(user_id=user_id, limit=500)
        cfg = PersonalizationConfig()
        sections, personalization_dbg = reorder_only_personalize_sections(
            sections=sections,
            recent_events=recent,
            user_id=user_id,
            debug=bool(debug),
            config=cfg,
        )

    tags_join_dbg: Dict[str, Any] = {}
    if include_tags:
        tag_store = _get_tag_store()
        sections, tags_join_dbg = _join_tags_into_sections(sections, tag_store, debug=bool(debug))

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
            d["personalization"] = personalization_dbg

        if include_tags:
            d["tags_join"] = tags_join_dbg

        body["debug"] = d

    return body


@router.get("/feed/mood")
def feed_mood(
    country: str = Query(...),
    mood: str = Query(..., description="Mood/scene slug, e.g. late-night-chill"),
    n: int = Query(default=10, ge=1, le=200),
    explicit_ok: bool = Query(default=False),
    debug: bool = Query(default=False),
    include_tags: bool = Query(default=False),
    pool_mult: int = Query(default=10, ge=2, le=50),
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    store: FeedbackStore = Depends(get_feedback_store),
) -> Dict[str, Any]:
    """
    V1.5.1 Step 3B:
      - Mood discovery feed (UX lift only)
      - No ranker changes. Filtering only.
      - Internally joins tags to filter, but response includes tags only if include_tags=true.
    """
    mslug = _slugify(mood)
    if not mslug:
        raise HTTPException(status_code=400, detail="mood is required")

    feeds = _get_feeds_engine()

    # Build a larger pool, then filter by mood tags
    pool_n = min(200, max(int(n) * int(pool_mult), 25))
    q = FeedQuery(country=country, genre=None, n=pool_n, explicit_ok=explicit_ok, debug=debug)
    sections, base_dbg = feeds.home_feed(q)

    user_id = _clean_user_id(x_user_id)
    suppressed: set[str] = set()
    removed_by_section: Dict[str, int] = {}
    personalization_dbg: Dict[str, Any] = {}

    # suppression + reorder-only personalization (same behavior as /feed/home)
    if user_id:
        suppressed = store.suppressed_track_ids(user_id=user_id, event_types=("dislike", "skip"), days=365)
        before = sections
        sections = _apply_suppression(sections, suppressed)
        removed_by_section = _suppression_removed_counts(before, sections)

        recent = store.recent_events(user_id=user_id, limit=500)
        cfg = PersonalizationConfig()
        sections, personalization_dbg = reorder_only_personalize_sections(
            sections=sections,
            recent_events=recent,
            user_id=user_id,
            debug=bool(debug),
            config=cfg,
        )

    # Always join tags internally (needed to filter)
    tag_store = _get_tag_store()
    sections_with_tags, tags_join_dbg = _join_tags_into_sections(sections, tag_store, debug=bool(debug))

    # Gather candidates in stable order: prefer "top" then other rails
    candidates: List[Dict[str, Any]] = []
    if isinstance(sections_with_tags.get("top"), list):
        for it in sections_with_tags.get("top", []):
            if isinstance(it, dict):
                candidates.append(it)

    for rail, items in sections_with_tags.items():
        if rail == "top":
            continue
        if not isinstance(items, list):
            continue
        for it in items:
            if isinstance(it, dict):
                candidates.append(it)

    # Filter by mood + de-dup by track_id
    seen: set[str] = set()
    mood_items: List[Dict[str, Any]] = []
    for it in candidates:
        tid = (it.get("track_id") or "").strip()
        if not tid or tid in seen:
            continue
        if _item_has_mood(it, mslug):
            seen.add(tid)
            mood_items.append(it)
        if len(mood_items) >= int(n):
            break

    # Optionally strip tag fields from response
    out_items: List[Any] = mood_items
    if not include_tags:
        out_items = _strip_tag_fields_from_items(out_items)

    body: Dict[str, Any] = {
        "ok": True,
        "country": country,
        "mood": mslug,
        "n": int(n),
        "sections": {"mood": out_items},
    }

    if debug:
        d = dict(base_dbg or {})
        d.setdefault("fallback_used", {})
        d["candidate_pool_n"] = int(pool_n)
        d["returned_n"] = int(len(mood_items))
        d["mood_filter"] = {"mood_slug": mslug}

        if user_id:
            d["disliked_suppressed_count"] = int(len(suppressed))
            d["suppressed_removed_by_section"] = removed_by_section
            d["suppressed_event_types"] = ["dislike", "skip"]
            d["personalization"] = personalization_dbg

        # Always show join debug since we always join internally here
        d["tags_join"] = tags_join_dbg
        body["debug"] = d

    return body


@router.get("/feed/for-you")
def feed_for_you(
    country: str = Query(...),
    n: int = Query(default=10, ge=5, le=200),
    explicit_ok: bool = Query(default=False),
    debug: bool = Query(default=False),
    include_tags: bool = Query(default=False),
    candidate_k: int = Query(default=1200, ge=300, le=20000),
    same_country_only: bool = Query(default=True),
    unique_artist: bool = Query(default=True),
    max_per_genre: int = Query(default=10, ge=0, le=200),
    lambda_relevance: float = Query(default=0.75, ge=0.0, le=1.0),
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    store: FeedbackStore = Depends(get_feedback_store),
) -> Dict[str, Any]:
    user_id = _require_user_id(x_user_id)
    feeds = _get_feeds_engine()

    q = FeedQuery(country=country, genre=None, n=n, explicit_ok=explicit_ok, debug=debug)
    sections, dbg = feeds.home_feed(q)

    suppressed = store.suppressed_track_ids(user_id=user_id, event_types=("dislike", "skip"), days=365)
    before_supp = sections
    sections = _apply_suppression(sections, suppressed)
    removed_by_section = _suppression_removed_counts(before_supp, sections)

    recent = store.recent_events(user_id=user_id, limit=500)
    events: list[SessionEvent] = []

    for e in recent:
        if not isinstance(e, dict):
            continue
        tid = (e.get("track_id") or "").strip()
        et = (e.get("event_type") or "").strip().lower()
        if not tid or not et:
            continue
        events.append(
            SessionEvent(
                session_id=user_id,
                track_id=tid,
                event_type=et,
                ts=_ts_to_iso(e.get("ts")),
            )
        )

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

    if suppressed and for_you_items:
        for_you_items = [it for it in for_you_items if it.get("track_id") not in suppressed]

    sections = dict(sections)
    sections["for_you"] = for_you_items[: int(n)]

    for_you_ids = {
        str(it.get("track_id"))
        for it in sections["for_you"]
        if isinstance(it, dict) and it.get("track_id")
    }
    sections, cross_removed = _dedup_other_sections_against_for_you(sections, for_you_ids)

    tags_join_dbg: Dict[str, Any] = {}
    if include_tags:
        tag_store = _get_tag_store()
        sections, tags_join_dbg = _join_tags_into_sections(sections, tag_store, debug=bool(debug))

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

        if include_tags:
            d["tags_join"] = tags_join_dbg

        body["debug"] = d

    return body