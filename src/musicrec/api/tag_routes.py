from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from musicrec.storage.tag_store import TagStore, TagStoreConfig


router = APIRouter(tags=["tags"])


def get_tag_store() -> TagStore:
    # Default matches tagging script runtime DB
    return TagStore(TagStoreConfig())


@router.get("/tags/stats")
def tags_stats(store: TagStore = Depends(get_tag_store)) -> Dict[str, Any]:
    # Tests expect a FLAT payload: {"rows": ..., "last_updated_at": ...}
    return store.stats()


@router.get("/tags/track/{track_id}")
def tags_track(track_id: str, store: TagStore = Depends(get_tag_store)) -> Dict[str, Any]:
    b = store.get(track_id)
    if b is None:
        raise HTTPException(status_code=404, detail="track_id not found")
    # Stable structure: always return bundle dict
    return b.to_dict()


@router.get("/tags/batch")
def tags_batch(
    track_id: List[str] = Query(..., description="Repeatable query param. Example: /tags/batch?track_id=A&track_id=B"),
    include_missing: bool = Query(default=False),
    store: TagStore = Depends(get_tag_store),
) -> Dict[str, Any]:
    ids = [str(t).strip() for t in (track_id or []) if str(t).strip()]
    if not ids:
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