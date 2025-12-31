from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from musicrec.storage.tag_store import TagStore
from musicrec.tagging.schema import TrackTagBundle

router = APIRouter(prefix="/tags", tags=["tags"])


# Keep this dependency lightweight and consistent with other stores
def get_tag_store() -> TagStore:
    return TagStore()


@router.get("/stats")
def tags_stats(store: TagStore = Depends(get_tag_store)) -> Dict[str, Any]:
    """
    Returns:
      { "rows": <int>, "last_updated_at": <float|None> }
    """
    return store.stats()


@router.get("/track/{track_id}")
def get_track_tags(
    track_id: str,
    store: TagStore = Depends(get_tag_store),
) -> Dict[str, Any]:
    """
    Returns the stored tag bundle for a track_id.
    """
    bundle = store.get(track_id)
    if not bundle:
        raise HTTPException(status_code=404, detail="tags_not_found")
    return bundle.to_dict()


@router.get("/batch")
def get_track_tags_batch(
    track_ids: List[str] = Query(..., alias="track_id"),
    include_missing: bool = Query(default=False),
    store: TagStore = Depends(get_tag_store),
) -> Dict[str, Any]:
    """
    Usage:
      /tags/batch?track_id=TRK-1&track_id=TRK-2

    Returns:
      {
        "items": [<TrackTagBundle dict> ...],
        "missing": [<track_id> ...]   # only if include_missing=true
      }
    """
    items: List[Dict[str, Any]] = []
    missing: List[str] = []

    # de-dupe while preserving input order
    seen: set[str] = set()
    cleaned: List[str] = []
    for tid in track_ids:
        t = (tid or "").strip()
        if not t:
            continue
        if t in seen:
            continue
        seen.add(t)
        cleaned.append(t)

    for tid in cleaned:
        bundle: Optional[TrackTagBundle] = store.get(tid)
        if bundle:
            items.append(bundle.to_dict())
        else:
            missing.append(tid)

    out: Dict[str, Any] = {"items": items}
    if include_missing:
        out["missing"] = missing
    return out