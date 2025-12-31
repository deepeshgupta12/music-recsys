from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query

from musicrec.storage.tag_store import TagStore, TagStoreConfig

router = APIRouter(tags=["tags"])


@lru_cache(maxsize=1)
def _get_tag_store_cached() -> TagStore:
    # Default matches tagging script runtime DB
    return TagStore(TagStoreConfig())


def get_tag_store() -> TagStore:
    return _get_tag_store_cached()


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


def _titleize_slug(slug: str) -> str:
    parts = [p for p in (slug or "").split("-") if p]
    return " ".join([p.capitalize() for p in parts]) if parts else ""


def _extract_multi(tags: Dict[str, Any], keys: List[str]) -> List[str]:
    """
    Extract list of string values from tags for any of the provided keys.
    Supports:
      - str
      - list[str]
    """
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


def _facet_counts_from_store(
    store: TagStore,
    *,
    max_rows: int,
    provider: Optional[str],
) -> Dict[str, Any]:
    """
    Scans tag rows in sqlite to build facets.
    We keep it best-effort and resilient to schema differences.

    Returns:
      {
        "rows_scanned": int,
        "mood_counts": dict[slug -> count],
      }
    """
    mood_keys = ["moods", "mood", "scenes", "scene"]
    mood_counts: Dict[str, int] = {}
    rows_scanned = 0

    # Best-effort: use TagStore internals if present
    connect = getattr(store, "_connect", None)
    cfg = getattr(store, "cfg", None)
    if connect is None or cfg is None:
        return {"rows_scanned": 0, "mood_counts": {}}

    table_name = getattr(cfg, "table_name", None) or getattr(cfg, "table", None) or "track_tags"
    prov = (provider or "").strip() or None

    try:
        with connect() as con:
            # discover columns
            cols: List[str] = []
            try:
                # PRAGMA table_info returns rows with "name"
                cols = [r["name"] for r in con.execute(f"PRAGMA table_info({table_name})").fetchall()]
            except Exception:
                cols = []

            # column preference order
            tags_col = "tags_json" if "tags_json" in cols else ("payload_json" if "payload_json" in cols else None)
            prov_col = "provider" if "provider" in cols else None

            if tags_col is None:
                return {"rows_scanned": 0, "mood_counts": {}}

            where = ""
            params: List[Any] = []
            if prov and prov_col:
                where = f" WHERE {prov_col} = ?"
                params.append(prov)

            limit_sql = f" LIMIT {int(max_rows)}" if int(max_rows) > 0 else ""

            # keep select minimal
            q = f"SELECT {tags_col} FROM {table_name}{where}{limit_sql}"
            rows = con.execute(q, tuple(params)).fetchall()

            for r in rows:
                rows_scanned += 1
                raw = r.get(tags_col)
                if not raw:
                    continue

                try:
                    import json

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
        # do not fail endpoint; return what we have
        return {"rows_scanned": rows_scanned, "mood_counts": mood_counts}

    return {"rows_scanned": rows_scanned, "mood_counts": mood_counts}


@router.get("/tags/stats")
def tags_stats(store: TagStore = Depends(get_tag_store)) -> Dict[str, Any]:
    # Tests expect a FLAT payload: {"rows": ..., "last_updated_at": ...}
    return store.stats()


@router.get("/tags/track/{track_id}")
def tags_track(track_id: str, store: TagStore = Depends(get_tag_store)) -> Dict[str, Any]:
    b = store.get(track_id)
    if b is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="track_id not found")
    return b.to_dict()


@router.get("/tags/batch")
def tags_batch(
    track_id: List[str] = Query(..., description="Repeatable query param. Example: /tags/batch?track_id=A&track_id=B"),
    include_missing: bool = Query(default=False),
    store: TagStore = Depends(get_tag_store),
) -> Dict[str, Any]:
    ids = [str(t).strip() for t in (track_id or []) if str(t).strip()]
    if not ids:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail="at least one track_id is required")

    found = store.batch_get(ids)  # dict(track_id -> TrackTagBundle)

    items: List[Dict[str, Any]] = []
    missing: List[str] = []

    for tid in ids:
        b = found.get(tid)
        if b is None:
            missing.append(tid)
            if include_missing:
                items.append({"track_id": tid, "missing": True})
        else:
            d = b.to_dict()
            d["missing"] = False
            items.append(d)

    return {
        "items": items,
        "missing": missing,
        "requested": len(ids),
        "found": len(found),
        "include_missing": bool(include_missing),
    }


@router.get("/tags/filters")
def tags_filters(
    limit: int = Query(default=30, ge=1, le=200),
    max_rows: int = Query(default=5000, ge=10, le=200000),
    provider: Optional[str] = Query(default=None),
    debug: bool = Query(default=False),
    store: TagStore = Depends(get_tag_store),
) -> Dict[str, Any]:
    """
    V1.5.1 Step 3A:
      - UX-facing tag facets derived from stored tags (moods/scenes)
      - Not ranker changes

    Returns:
      { ok: true, moods: [{key, label, count}], debug?: {...} }
    """
    res = _facet_counts_from_store(store, max_rows=int(max_rows), provider=provider)
    mood_counts: Dict[str, int] = res.get("mood_counts", {}) or {}
    rows_scanned = int(res.get("rows_scanned", 0) or 0)

    moods = sorted(
        [{"key": k, "label": _titleize_slug(k), "count": int(v)} for k, v in mood_counts.items()],
        key=lambda x: (-x["count"], x["key"]),
    )[: int(limit)]

    out: Dict[str, Any] = {"ok": True, "moods": moods}
    if debug:
        out["debug"] = {
            "rows_scanned": int(rows_scanned),
            "unique_moods": int(len(mood_counts)),
            "provider_filter": provider,
            "limit": int(limit),
            "max_rows": int(max_rows),
        }
    return out