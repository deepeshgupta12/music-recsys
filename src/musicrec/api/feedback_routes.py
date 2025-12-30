from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from musicrec.api.deps import get_feedback_store  # you'll add this tiny dep in Step 1.5.1D


router = APIRouter(tags=["feedback"])


class FeedbackIn(BaseModel):
    track_id: str = Field(..., min_length=1)
    event_type: str = Field(..., min_length=1)  # play/like/dislike/skip
    ts: Optional[float] = None  # unix seconds (optional)


@router.post("/events/feedback")
def post_feedback(
    payload: FeedbackIn,
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
) -> Dict[str, Any]:
    """
    Record a single feedback event.
    Anonymous-friendly: client passes X-User-Id.
    """
    user_id = (x_user_id or "").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="X-User-Id header is required")

    store = get_feedback_store()
    try:
        store.add_event(user_id=user_id, track_id=payload.track_id, event_type=payload.event_type, ts=payload.ts)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"ok": True}