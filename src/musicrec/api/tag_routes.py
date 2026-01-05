from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Query

from musicrec.storage.feature_table import load_feature_table
from musicrec.storage.tag_store import TagStore, TagStoreConfig

router = APIRouter(tags=["tags"])


def _get_store() -> TagStore:
    return TagStore(TagStoreConfig())


@router.get("/tags/stats")
def tags_stats() -> Dict[str, Any]:
    store = _get_store()
    return store.stats()


@router.get("/tags/coverage")
def tags_coverage() -> Dict[str, Any]:
    """
    v1.5.2 Step 5A:
      - Exposes tag coverage vs catalog size.
      - Helps diagnose mood rails / mood feed quality.
    """
    store = _get_store()
    stats = store.stats()

    ft = load_feature_table()
    catalog_tracks = int(len(ft))
    tagged_tracks = int(store.count_distinct_track_ids())

    coverage_pct = 0.0
    if catalog_tracks > 0:
        coverage_pct = float(tagged_tracks) * 100.0 / float(catalog_tracks)

    return {
        "ok": True,
        "catalog_tracks": catalog_tracks,
        "tagged_tracks": tagged_tracks,
        "coverage_pct": coverage_pct,
        "rows": int(stats.get("rows") or 0),
        "last_updated_at": float(stats.get("last_updated_at") or 0.0),
    }


@router.get("/tags/track")
def tags_track(
    track_id: str = Query(...),
    include_meta: bool = Query(default=False),
) -> Dict[str, Any]:
    store = _get_store()
    b = store.get(track_id)
    if not b:
        return {"ok": True, "track_id": track_id, "found": False, "tags": {}}

    out: Dict[str, Any] = {"ok": True, "track_id": track_id, "found": True, "tags": (b.tags or {})}
    if include_meta:
        out["meta"] = {"source": b.source, "model_id": b.model_id}
    return out


@router.get("/tags/batch")
def tags_batch(
    ids: str = Query(..., description="Comma-separated track_ids"),
    include_meta: bool = Query(default=False),
) -> Dict[str, Any]:
    store = _get_store()
    track_ids = [x.strip() for x in (ids or "").split(",") if x.strip()]
    found = store.batch_get(track_ids)

    items = []
    for tid in track_ids:
        b = found.get(tid)
        if not b:
            items.append({"track_id": tid, "found": False, "tags": {}})
            continue
        d: Dict[str, Any] = {"track_id": tid, "found": True, "tags": (b.tags or {})}
        if include_meta:
            d["meta"] = {"source": b.source, "model_id": b.model_id}
        items.append(d)

    return {"ok": True, "n": len(track_ids), "items": items}


@router.get("/tags/filters")
def tags_filters(
    limit: int = Query(default=20, ge=1, le=50),
    debug: bool = Query(default=False),
    provider: Optional[str] = Query(default=None),
    max_rows: int = Query(default=5000, ge=100, le=50000),
) -> Dict[str, Any]:
    """
    Existing endpoint (kept as-is conceptually): returns mood facets.
    NOTE: Implementation may vary across your v1.5.1 branches — do not break it here.
    """
    # If your current branch already has a working /tags/filters implementation,
    # keep it. This placeholder exists ONLY because you asked for full-file output.
    #
    # In your repo state (v1.5.1), you already have a working /tags/filters.
    # So: copy/paste your existing implementation below this comment.
    #
    # ---- BEGIN: paste existing /tags/filters implementation ----
    store = _get_store()

    # Very small safe default: return empty if you haven't pasted your existing logic.
    # Replace this block with your existing /tags/filters code.
    out = {"ok": True, "moods": []}
    if debug:
        out["debug"] = {
            "rows_scanned": 0,
            "unique_moods": 0,
            "provider_filter": provider,
            "limit": limit,
            "max_rows": max_rows,
        }
    return out
    # ---- END: paste existing /tags/filters implementation ----