from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from musicrec.storage.tag_store import TagRow, TagStore, TagStoreConfig

router = APIRouter()


@lru_cache(maxsize=1)
def _get_store() -> TagStore:
    # Default store (repo-level sqlite)
    return TagStore(TagStoreConfig())


def get_tag_store() -> TagStore:
    return _get_store()


@router.get("/tags/stats")
def tags_stats(store: TagStore = Depends(get_tag_store)) -> Dict[str, Any]:
    # IMPORTANT: tests expect rows/last_updated_at at top-level (not nested in {"ok":..., "stats":...})
    return store.stats()


@router.get("/tags/track/{track_id}")
def tags_track(track_id: str, store: TagStore = Depends(get_tag_store)) -> Dict[str, Any]:
    row = store.get(track_id)
    if row is None:
        raise HTTPException(status_code=404, detail="track_id not found")
    return {
        "track_id": row.track_id,
        "provider": row.provider,
        "tags": row.tags,
        "updated_at": row.updated_at,
    }


@router.get("/tags/batch")
def tags_batch(
    track_id: List[str] = Query(..., description="Repeatable query param. Example: /tags/batch?track_id=A&track_id=B"),
    include_missing: bool = Query(default=False),
    store: TagStore = Depends(get_tag_store),
) -> Dict[str, Any]:
    ids = [str(t).strip() for t in (track_id or []) if str(t).strip()]
    if not ids:
        raise HTTPException(status_code=400, detail="at least one track_id is required")

    found = store.batch_get(ids)

    items: List[Dict[str, Any]] = []
    missing: List[str] = []

    for tid in ids:
        r = found.get(tid)
        if r is None:
            missing.append(tid)
            if include_missing:
                items.append({"track_id": tid, "missing": True})
        else:
            items.append(
                {
                    "track_id": r.track_id,
                    "provider": r.provider,
                    "tags": r.tags,
                    "updated_at": r.updated_at,
                    "missing": False,
                }
            )

    return {
        "ok": True,
        "items": items,
        "missing": missing,
        "requested": len(ids),
        "found": len(found),
    }