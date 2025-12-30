from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

from musicrec.api.deps import get_feedback_store
from musicrec.storage.feedback_store import FeedbackStore


router = APIRouter(tags=["feedback"])


class FeedbackIn(BaseModel):
    track_id: str = Field(..., min_length=1)
    event_type: str = Field(..., min_length=1)  # play/like/dislike/skip
    ts: Optional[float] = None  # unix seconds (optional)


def require_user_id(
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
) -> str:
    user_id = (x_user_id or "").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="X-User-Id header is required")
    return user_id


@router.post("/events/feedback")
def post_feedback(
    payload: FeedbackIn,
    user_id: str = Depends(require_user_id),
    store: FeedbackStore = Depends(get_feedback_store),
) -> Dict[str, Any]:
    """
    Record a single feedback event.
    Anonymous-friendly: client passes X-User-Id.
    """
    try:
        store.add_event(
            user_id=user_id,
            track_id=payload.track_id,
            event_type=payload.event_type,
            ts=payload.ts,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"ok": True}


@router.get("/events/feedback/recent")
def feedback_recent(
    user_id: str = Depends(require_user_id),
    store: FeedbackStore = Depends(get_feedback_store),
    limit: int = Query(default=50, ge=1, le=200),
) -> Dict[str, Any]:
    """
    Returns most recent feedback events for the user (desc by ts).
    """
    try:
        events = store.recent_events(user_id=user_id, limit=limit)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "ok": True,
        "user_id": user_id,
        "n": len(events),
        "events": [
            {
                "user_id": e.user_id,
                "track_id": e.track_id,
                "event_type": e.event_type,
                "ts": e.ts,
            }
            for e in events
        ],
    }


@router.get("/events/feedback/stats")
def feedback_stats(
    user_id: str = Depends(require_user_id),
    store: FeedbackStore = Depends(get_feedback_store),
    days: int = Query(default=30, ge=1, le=365),
) -> Dict[str, Any]:
    """
    Counts by event_type for the last N days.
    """
    try:
        counts = store.counts_last_days(user_id=user_id, days=days)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "ok": True,
        "user_id": user_id,
        "days": days,
        "counts": counts,
        "total": int(sum(counts.values())),
    }