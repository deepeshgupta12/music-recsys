from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query

from musicrec.storage.tag_store import TagStore

router = APIRouter(tags=["tags"])


def get_tag_store() -> TagStore:
    return TagStore()


@router.get("/tags/stats")
def tags_stats(store: TagStore = Depends(get_tag_store)) -> Dict[str, Any]:
    """
    Test expectation (tests/test_api_tags.py):
      - status 200
      - JSON has top-level keys: "rows", "last_updated_at"
    """
    stats = store.stats()
    # Put rows + last_updated_at at TOP LEVEL (no nesting).
    return {"ok": True, **stats}


@router.get("/tags/track/{track_id}")
def tags_track(track_id: str, store: TagStore = Depends(get_tag_store)) -> Dict[str, Any]:
    row = store.get(track_id)
    if not row:
        raise HTTPException(status_code=404, detail="tags not found for track_id")
    return {
        "ok": True,
        "item": {
            "track_id": row.track_id,
            "provider": row.provider,
            "tags": row.tags,
            "updated_at": row.updated_at,
        },
    }


@router.get("/tags/batch")
def tags_batch(
    track_id: List[str] = Query(..., description="Repeatable query param. Example: /tags/batch?track_id=A&track_id=B"),
    include_missing: bool = Query(default=False),
    store: TagStore = Depends(get_tag_store),
) -> Dict[str, Any]:
    """
    Always returns 200 with stable structure.

    If include_missing=true:
      - missing ids are included in `missing`
      - items includes {"track_id": "...", "missing": true} for those ids
    """
    ids = [str(t).strip() for t in (track_id or []) if str(t).strip()]
    if not ids:
        raise HTTPException(status_code=400, detail="at least one track_id is required")

    rows = store.batch_get(ids)
    found = {r.track_id: r for r in rows}

    items: List[Dict[str, Any]] = []
    missing: List[str] = []

    for tid in ids:
        r = found.get(tid)
        if r:
            items.append(
                {
                    "track_id": r.track_id,
                    "provider": r.provider,
                    "tags": r.tags,
                    "updated_at": r.updated_at,
                }
            )
        else:
            missing.append(tid)
            if include_missing:
                items.append({"track_id": tid, "missing": True})

    return {"ok": True, "items": items, "missing": missing}