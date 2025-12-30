from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel, Field

from musicrec.api.deps import get_feedback_store

router = APIRouter(tags=["feedback"])

VALID_EVENT_TYPES = {"like", "dislike", "skip", "save", "unsave"}


class FeedbackIn(BaseModel):
    track_id: str = Field(..., min_length=1)
    event_type: str = Field(..., min_length=1)
    meta: Dict[str, Any] = Field(default_factory=dict)


@router.post("/events/feedback")
def post_feedback(
    payload: FeedbackIn,
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
) -> Dict[str, Any]:
    user_id = (x_user_id or "").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="X-User-Id header is required")

    event_type = (payload.event_type or "").strip().lower()
    if event_type not in VALID_EVENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"invalid event_type; allowed={sorted(VALID_EVENT_TYPES)}",
        )

    track_id = (payload.track_id or "").strip()
    if not track_id:
        raise HTTPException(status_code=400, detail="track_id is required")

    store = get_feedback_store()
    event_id = store.add_event(
        user_id=user_id,
        track_id=track_id,
        event_type=event_type,
        meta=dict(payload.meta or {}),
    )
    return {"ok": True, "event_id": event_id}


@router.get("/events/feedback/recent")
def get_recent_feedback(
    limit: int = Query(50, ge=1, le=500),
    user_id: Optional[str] = Query(None, description="Optional filter by user_id"),
    event_type: Optional[str] = Query(None, description="Optional filter by event_type"),
) -> Dict[str, Any]:
    store = get_feedback_store()
    et = (event_type or "").strip().lower() or None
    if et is not None and et not in VALID_EVENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"invalid event_type; allowed={sorted(VALID_EVENT_TYPES)}",
        )

    events = store.recent_events(limit=limit, user_id=user_id, event_type=et)
    return {"ok": True, "limit": limit, "user_id": user_id, "event_type": et, "events": events}


@router.get("/events/feedback/stats")
def get_feedback_stats(
    window_days: int = Query(7, ge=1, le=365),
    user_id: Optional[str] = Query(None),
) -> Dict[str, Any]:
    store = get_feedback_store()
    stats = store.stats(window_days=window_days, user_id=user_id)
    return {"ok": True, "stats": stats}