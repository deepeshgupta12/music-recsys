from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request

from musicrec.storage.tag_store import TagStore, TagStoreConfig

router = APIRouter(prefix="/tags", tags=["tags"])


def _get_tag_store(request: Request) -> TagStore:
    # Prefer app.state if available (production), else fallback to runtime default
    st = getattr(request.app.state, "tag_store", None)
    if st is not None:
        return st

    # Safe default for tests/local
    cfg = TagStoreConfig(db_path="runtime/tags.db", table_name="track_tags")
    return TagStore(cfg)


@router.get("/batch")
def tags_batch(
    request: Request,
    # Support both names to avoid breaking older FE/tests/clients.
    track_id: Optional[List[str]] = Query(None),
    track_ids: Optional[List[str]] = Query(None),
    include_missing: bool = Query(False),
) -> Dict[str, Any]:
    ids = (track_id or []) + (track_ids or [])
    ids = [x for x in ids if (x or "").strip()]

    if not ids:
        # Keep FastAPI-style error semantics; tests always pass ids.
        raise HTTPException(status_code=422, detail="At least one track_id is required")

    store = _get_tag_store(request)
    found = store.batch_get(ids)

    items: List[Dict[str, Any]] = []
    missing: List[str] = []

    for tid in ids:
        key = (tid or "").strip()
        b = found.get(key) or found.get(key.lower()) or found.get(key.upper())
        if b is None:
            missing.append(key)
            if include_missing:
                items.append({"track_id": key, "tags": None})
        else:
            items.append({"track_id": key, "tags": b.to_dict()})

    return {
        "ok": True,
        "items": items,
        "missing": missing,
        "found_count": len(items) - (len(missing) if not include_missing else 0),
        "missing_count": len(missing),
    }


@router.get("/stats")
def tags_stats(request: Request) -> Dict[str, Any]:
    store = _get_tag_store(request)
    s = store.stats()
    return {"rows": s["rows"], "last_updated_at": s["last_updated_at"]}


@router.get("/coverage")
def tags_coverage(
    request: Request,
    # If your catalog size is known elsewhere, you can wire it later.
    catalog_tracks: int = Query(0),
) -> Dict[str, Any]:
    store = _get_tag_store(request)
    s = store.stats()
    rows = int(s["rows"] or 0)
    catalog = int(catalog_tracks or 0)

    coverage_pct = (rows / catalog) * 100.0 if catalog > 0 else 0.0

    return {
        "ok": True,
        "catalog_tracks": catalog,
        "tagged_tracks": rows,
        "coverage_pct": coverage_pct,
        "rows": rows,
        "last_updated_at": s["last_updated_at"],
    }