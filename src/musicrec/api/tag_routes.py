from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from musicrec.storage.tag_store import TagStore

router = APIRouter(tags=["tags"])


# Simple singleton store for API runtime
_TAG_STORE: Optional[TagStore] = None


def get_tag_store() -> TagStore:
    global _TAG_STORE
    if _TAG_STORE is None:
        _TAG_STORE = TagStore()
    return _TAG_STORE


@router.get("/tags/stats")
def tags_stats(store: TagStore = Depends(get_tag_store)) -> Dict[str, Any]:
    # Tests expect: {"rows": ..., "last_updated_at": ...} at top-level
    return store.stats()


@router.get("/tags/track/{track_id}")
def tags_track(track_id: str, store: TagStore = Depends(get_tag_store)) -> Dict[str, Any]:
    row = store.get(track_id)
    if row is None:
        raise HTTPException(status_code=404, detail="track_id not found")

    # If it's a TrackTagBundle-like object, return dict representation when possible
    to_dict = getattr(row, "to_dict", None)
    if callable(to_dict):
        return {"ok": True, "item": to_dict()}

    pyd_dict = getattr(row, "dict", None)
    if callable(pyd_dict):
        return {"ok": True, "item": pyd_dict()}

    # fallback: dict
    return {"ok": True, "item": row}


@router.get("/tags/batch")
def tags_batch(
    track_id: List[str] = Query(..., description="Repeatable query param. Example: /tags/batch?track_id=A&track_id=B"),
    include_missing: bool = Query(default=False),
    store: TagStore = Depends(get_tag_store),
) -> Dict[str, Any]:
    """
    Batch fetch tags for multiple track_ids.
    Always returns 200 with stable structure.

    If include_missing=true, missing track_ids are listed under `missing`
    and also represented inside `items` with {"track_id": "...", "missing": true}.
    """
    ids = [str(t).strip() for t in (track_id or []) if str(t).strip()]
    if not ids:
        raise HTTPException(status_code=400, detail="at least one track_id is required")

    found = store.batch_get(ids)  # MUST be dict(track_id -> bundle/dict)

    items: List[Dict[str, Any]] = []
    missing: List[str] = []

    for tid in ids:
        r = found.get(tid)
        if r is None:
            missing.append(tid)
            if include_missing:
                items.append({"track_id": tid, "missing": True})
            continue

        to_dict = getattr(r, "to_dict", None)
        if callable(to_dict):
            items.append(to_dict())
            continue

        pyd_dict = getattr(r, "dict", None)
        if callable(pyd_dict):
            items.append(pyd_dict())
            continue

        if isinstance(r, dict):
            items.append(r)
        else:
            items.append({"track_id": tid, "value": str(r)})

    return {
        "ok": True,
        "items": items,
        "missing": missing,
        "found": len(found),
        "requested": len(ids),
        "include_missing": bool(include_missing),
    }