from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from musicrec.storage.tag_store import TagStore

router = APIRouter(tags=["tags"])


@lru_cache(maxsize=1)
def _get_tag_store() -> TagStore:
    # TagStore is lightweight (sqlite-backed) and safe to cache per-process
    return TagStore()


def get_tag_store() -> TagStore:
    return _get_tag_store()


def _row_to_api(row) -> Dict[str, Any]:
    return {
        "track_id": row.track_id,
        "provider": row.provider,
        "tags": dict(row.tags or {}),
        "updated_at": float(row.updated_at),
    }


@router.get("/tags/stats")
def tags_stats(store: TagStore = Depends(get_tag_store)) -> Dict[str, Any]:
    """
    Basic stats for tag store.
    Expected by tests: returns 200 with stable shape.
    """
    s = store.stats()
    return {
        "ok": True,
        "rows": int(s.get("rows", 0)),
        "last_updated_at": float(s.get("last_updated_at") or 0.0),
    }


@router.get("/tags/track/{track_id}")
def tags_for_track(track_id: str, store: TagStore = Depends(get_tag_store)) -> Dict[str, Any]:
    """
    Return tags for a single track.
    If missing: 404.
    """
    tid = (track_id or "").strip()
    if not tid:
        raise HTTPException(status_code=400, detail="track_id is required")

    row = store.get(tid)
    if row is None:
        raise HTTPException(status_code=404, detail="tags not found for track_id")

    return {"ok": True, "item": _row_to_api(row)}


@router.get("/tags/batch")
def tags_batch(
    track_id: List[str] = Query(..., description="Repeatable query param. Example: /tags/batch?track_id=A&track_id=B"),
    include_missing: bool = Query(default=False),
    store: TagStore = Depends(get_tag_store),
) -> Dict[str, Any]:
    """
    Batch fetch tags for multiple track_ids.
    Always returns 200 with stable structure.

    If include_missing=true, missing track_ids are included in response under `missing`
    and also represented in `items` as {"track_id": "...", "missing": true}.
    """
    ids = [str(t).strip() for t in (track_id or []) if str(t).strip()]
    if not ids:
        raise HTTPException(status_code=400, detail="at least one track_id is required")

    rows = store.batch_get(ids)
    row_by_id: Dict[str] = {r.track_id: r for r in rows}

    items: List[Dict[str, Any]] = []
    missing: List[str] = []

    for tid in ids:
        row = row_by_id.get(tid)
        if row is None:
            missing.append(tid)
            if include_missing:
                items.append({"track_id": tid, "missing": True})
            continue
        items.append(_row_to_api(row))

    return {
        "ok": True,
        "n_requested": int(len(ids)),
        "n_found": int(len(rows)),
        "include_missing": bool(include_missing),
        "missing": missing,
        "items": items,
    }