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
from musicrec.storage.tag_store import TagStore, TagStoreConfig

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
    # Align with tag_routes.py: same config/path/table
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


def _alt_track_ids(tid: str) -> List[str]:
    """
    Defensive normalization: try alternative forms to increase match rate
    between feed track_ids and tag_store primary keys.
    """
    t = (tid or "").strip()
    if not t:
        return []

    out: List[str] = []
    seen: set[str] = set()

    def add(x: str) -> None:
        x = (x or "").strip()
        if x and x not in seen:
            seen.add(x)
            out.append(x)

    add(t)
    add(t.upper())
    add(t.lower())

    # common prefix variants
    if t.upper().startswith("TRK-"):
        add(t[4:])
        add(t[4:].upper())
        add(t[4:].lower())

    if t.lower().startswith("trk-"):
        add(t[4:])
        add(t[4:].upper())
        add(t[4:].lower())

    # remove surrounding underscores sometimes introduced by clients/tests
    add(t.strip("_"))
    add(t.strip("_").upper())
    add(t.strip("_").lower())

    return out


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

    # Expand lookup ids (defensive)
    expanded_ids: set[str] = set()
    for tid in unique_ids:
        for a in _alt_track_ids(tid):
            expanded_ids.add(a)

    found_map: Dict[str, Any] = {}
    if expanded_ids:
        found_map = store.batch_get(sorted(expanded_ids))  # type: ignore[attr-defined]

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

            tid0 = (it.get("track_id") or "").strip()
            if not tid0:
                new_items.append(it)
                continue

            row = None
            for cand in _alt_track_ids(tid0):
                row = found_map.get(cand)
                if row is not None:
                    break

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
        "expanded_ids_n": int(len(expanded_ids)),
        "unique_found": int(unique_found),
        "total_items_seen": int(total_items_seen),
        "tagged_items": int(tagged_items),
        "missing_items": int(missing_items),
    }
    return out, (dbg if debug else {})


# ---------- Step 3 helpers (mood feed + rails) ----------

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


def _row_value(row: Any, col: str, idx: int = 0) -> Any:
    """
    Robust sqlite row access:
      - dict: row.get(col)
      - sqlite3.Row: row[col] or row[idx]
      - tuple/list: row[idx]
    """
    if row is None:
        return None
    try:
        if isinstance(row, dict):
            return row.get(col)
        # sqlite3.Row supports __getitem__ by col name
        try:
            return row[col]  # type: ignore[index]
        except Exception:
            pass
        try:
            return row[idx]  # type: ignore[index]
        except Exception:
            return None
    except Exception:
        return None


def _extract_multi(tags: Dict[str, Any], keys: List[str]) -> List[str]:
    out: List[str] = []
    for k in keys:
        v = tags.get(k)
        if v is None:
            continue
        if isinstance(v, str):
            x = v.strip()
            if x:
                out.append(x)
        elif isinstance(v, list):
            for it in v:
                if isinstance(it, str):
                    x = it.strip()
                    if x:
                        out.append(x)
    return out


def _mood_counts_from_store(
    store: TagStore,
    *,
    max_rows: int = 5000,
    provider: Optional[str] = None,
) -> Tuple[Dict[str, int], int]:
    """
    Best-effort sqlite scan (same approach as /tags/filters):
      - reads tags_json/payload_json from TagStore sqlite table
      - extracts moods/scenes
    Returns: (mood_counts, rows_scanned)
    """
    mood_keys = ["moods", "mood", "scenes", "scene"]
    mood_counts: Dict[str, int] = {}
    rows_scanned = 0

    connect = getattr(store, "_connect", None)
    cfg = getattr(store, "cfg", None)
    if connect is None or cfg is None:
        return {}, 0

    table_name = getattr(cfg, "table_name", None) or getattr(cfg, "table", None) or "track_tags"
    prov = (provider or "").strip() or None

    try:
        with connect() as con:
            cols: List[str] = []
            try:
                cols = [r["name"] for r in con.execute(f"PRAGMA table_info({table_name})").fetchall()]
            except Exception:
                cols = []

            tags_col = "tags_json" if "tags_json" in cols else ("payload_json" if "payload_json" in cols else None)
            prov_col = "provider" if "provider" in cols else None
            if tags_col is None:
                return {}, 0

            where = ""
            params: List[Any] = []
            if prov and prov_col:
                where = f" WHERE {prov_col} = ?"
                params.append(prov)

            limit_sql = f" LIMIT {int(max_rows)}" if int(max_rows) > 0 else ""
            q = f"SELECT {tags_col} FROM {table_name}{where}{limit_sql}"
            rows = con.execute(q, tuple(params)).fetchall()

            import json

            for r in rows:
                rows_scanned += 1
                raw = _row_value(r, tags_col, 0)
                if not raw:
                    continue
                try:
                    tags = json.loads(raw) if isinstance(raw, str) else {}
                except Exception:
                    tags = {}
                if not isinstance(tags, dict):
                    continue

                vals = _extract_multi(tags, mood_keys)
                for v in vals:
                    slug = _slugify(v)
                    if not slug:
                        continue
                    mood_counts[slug] = mood_counts.get(slug, 0) + 1
    except Exception:
        return mood_counts, rows_scanned

    return mood_counts, rows_scanned


def _pick_top_moods(
    store: TagStore,
    *,
    limit: int,
    max_rows: int = 5000,
    provider: Optional[str] = None,
) -> Tuple[List[str], Dict[str, int], int]:
    mood_counts, rows_scanned = _mood_counts_from_store(store, max_rows=max_rows, provider=provider)
    top = sorted(mood_counts.items(), key=lambda kv: (-int(kv[1]), kv[0]))[: int(limit)]
    return [k for k, _ in top], mood_counts, int(rows_scanned)


def _stable_candidates_from_sections(sections_with_tags: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Stable order:
      - "top" rail first (if present)
      - then other rails in insertion order
    """
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

    return candidates


def _dedup_items_by_track_id(items: List[Dict[str, Any]], limit: int, exclude: Optional[set[str]] = None) -> List[Dict[str, Any]]:
    exclude = exclude or set()
    seen: set[str] = set(exclude)
    out: List[Dict[str, Any]] = []
    for it in items:
        tid = (it.get("track_id") or "").strip()
        if not tid or tid in seen:
            continue
        seen.add(tid)
        out.append(it)
        if len(out) >= int(limit):
            break
    return out


@router.get("/feed/home")
def feed_home(
    country: str = Query(...),
    n: int = Query(default=10, ge=1, le=200),
    explicit_ok: bool = Query(default=False),
    debug: bool = Query(default=False),
    include_tags: bool = Query(default=False),
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    mood_rails: bool = Query(default=False),
    moods_k: int = Query(default=3, ge=1, le=8),
    mood_rail_n: int = Query(default=10, ge=3, le=30),
    mood_pool_mult: int = Query(default=10, ge=2, le=50),
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

    # Optional: include tags in the "main" rails
    tags_join_dbg: Dict[str, Any] = {}
    if include_tags:
        tag_store = _get_tag_store()
        sections, tags_join_dbg = _join_tags_into_sections(sections, tag_store, debug=bool(debug))

    # --- Mood rails (Step 4D) ---
    mood_rails_dbg: Dict[str, Any] = {}
    if mood_rails:
        tag_store = _get_tag_store()

        moods, mood_counts, rows_scanned = _pick_top_moods(
            tag_store,
            limit=int(moods_k),
            max_rows=5000,
            provider=None,
        )

        moods_source = "tag_facets"
        fallback_used = False
        fallback_moods_used: List[str] = []

        # If facet scan yields nothing, fallback to a stable list
        if not moods:
            fallback_used = True
            moods_source = "fallback_list"
            fallback_moods_used = ["focus", "chill", "study", "party", "gym", "calm"][: int(moods_k)]
            moods = list(fallback_moods_used)

        # Build a candidate pool (bigger than mood_rail_n)
        pool_n = min(200, max(int(mood_rail_n) * int(mood_pool_mult), 25))
        q2 = FeedQuery(country=country, genre=None, n=pool_n, explicit_ok=explicit_ok, debug=debug)
        pool_sections, _ = feeds.home_feed(q2)

        # Apply same suppression/personalization to candidate pool
        if user_id:
            pool_sections = _apply_suppression(pool_sections, suppressed)
            recent = store.recent_events(user_id=user_id, limit=500)
            cfg = PersonalizationConfig()
            pool_sections, _ = reorder_only_personalize_sections(
                sections=pool_sections,
                recent_events=recent,
                user_id=user_id,
                debug=False,
                config=cfg,
            )

        # Always join tags internally for filtering mood rails
        pool_with_tags, pool_tags_dbg = _join_tags_into_sections(pool_sections, tag_store, debug=bool(debug))
        candidates = _stable_candidates_from_sections(pool_with_tags)

        mood_sections_added: Dict[str, int] = {}
        used_track_ids: set[str] = set()

        # For each mood, filter candidates; if empty, fallback to unfiltered candidates
        for mslug in moods:
            out_items: List[Dict[str, Any]] = []
            for it in candidates:
                tid = (it.get("track_id") or "").strip()
                if not tid or tid in used_track_ids:
                    continue
                if _item_has_mood(it, mslug):
                    used_track_ids.add(tid)
                    out_items.append(it)
                if len(out_items) >= int(mood_rail_n):
                    break

            # Rail-level fallback to avoid empty UX rails
            rail_fallback_used = False
            if len(out_items) == 0:
                rail_fallback_used = True
                out_items = _dedup_items_by_track_id(candidates, limit=int(mood_rail_n), exclude=used_track_ids)
                for it in out_items:
                    tid = (it.get("track_id") or "").strip()
                    if tid:
                        used_track_ids.add(tid)

            final_items: List[Any] = out_items
            if not include_tags:
                final_items = _strip_tag_fields_from_items(final_items)

            sections[f"mood__{mslug}"] = final_items
            mood_sections_added[mslug] = int(len(out_items))

            # store per-rail fallback info in debug only
            if debug and rail_fallback_used:
                mood_rails_dbg.setdefault("rail_fallbacks", {})
                mood_rails_dbg["rail_fallbacks"][mslug] = True

        if debug:
            mood_rails_dbg.update(
                {
                    "enabled": True,
                    "moods_selected": moods,
                    "moods_source": moods_source,
                    "fallback_used": bool(fallback_used),
                    "fallback_moods_used": fallback_moods_used,
                    "mood_counts_selected": {m: int(mood_counts.get(m, 0)) for m in moods} if mood_counts else {},
                    "rows_scanned": int(rows_scanned),
                    "pool_n": int(pool_n),
                    "mood_rail_n": int(mood_rail_n),
                    "moods_k": int(moods_k),
                    "pool_tags_join": pool_tags_dbg,
                    "sections_added": mood_sections_added,
                }
            )

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
            d["personalization"] = personalization_dbg

        if include_tags:
            d["tags_join"] = tags_join_dbg

        if mood_rails:
            d["mood_rails"] = mood_rails_dbg

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
    fallback: bool = Query(default=True, description="If true, return fallback rail when no mood matches"),
    pool_mult: int = Query(default=10, ge=2, le=50),
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    store: FeedbackStore = Depends(get_feedback_store),
) -> Dict[str, Any]:
    """
    v1.5.1 Step 4:
      - Mood discovery feed (UX lift only)
      - No ranker changes. Filtering only.
      - If no mood matches, fallback to a non-empty rail (default: enabled).
      - Internally joins tags to filter, but response includes tags only if include_tags=true.
    """
    mslug = _slugify(mood)
    if not mslug:
        raise HTTPException(status_code=400, detail="mood is required")

    feeds = _get_feeds_engine()

    pool_n = min(200, max(int(n) * int(pool_mult), 25))
    q = FeedQuery(country=country, genre=None, n=pool_n, explicit_ok=explicit_ok, debug=debug)
    sections, base_dbg = feeds.home_feed(q)

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

    tag_store = _get_tag_store()
    sections_with_tags, tags_join_dbg = _join_tags_into_sections(sections, tag_store, debug=bool(debug))
    candidates = _stable_candidates_from_sections(sections_with_tags)

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

    fallback_used = False
    fallback_reason: Optional[str] = None

    if len(mood_items) == 0:
        fallback_reason = "no_mood_matches"
        if bool(fallback):
            fallback_used = True
            mood_items = _dedup_items_by_track_id(candidates, limit=int(n))
            if len(mood_items) == 0:
                fallback_reason = "no_candidates"

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
        d["mood_filter"] = {
            "mood_slug": mslug,
            "fallback_param": bool(fallback),
            "fallback_used": bool(fallback_used),
            "fallback_reason": fallback_reason,
        }

        if user_id:
            d["disliked_suppressed_count"] = int(len(suppressed))
            d["suppressed_removed_by_section"] = removed_by_section
            d["suppressed_event_types"] = ["dislike", "skip"]
            d["personalization"] = personalization_dbg

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