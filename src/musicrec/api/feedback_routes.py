from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

from musicrec.storage.feedback_store import FeedbackEvent, FeedbackStore

router = APIRouter()

ALLOWED_EVENT_TYPES = {"like", "dislike", "skip", "play"}


def get_feedback_store() -> FeedbackStore:
    # Keep default stable; tests override dependency anyway.
    db_path = os.getenv("MUSICREC_FEEDBACK_DB_PATH", "data/feedback.sqlite")
    return FeedbackStore(db_path)


def _require_user_id(x_user_id: Optional[str]) -> str:
    if x_user_id is None or not x_user_id.strip():
        raise HTTPException(status_code=400, detail="Missing X-User-Id header")
    return x_user_id.strip()


class FeedbackIn(BaseModel):
    track_id: str = Field(..., min_length=1)
    event_type: str = Field(..., min_length=1)
    ts: Optional[float] = None


@router.post("/events/feedback")
def post_feedback(
    payload: FeedbackIn,
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    store: FeedbackStore = Depends(get_feedback_store),
) -> Dict[str, Any]:
    user_id = _require_user_id(x_user_id)

    et = (payload.event_type or "").strip()
    if et not in ALLOWED_EVENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid event_type: {et}")

    store.record_event(
        user_id=user_id,
        track_id=payload.track_id.strip(),
        event_type=et,
        ts=payload.ts,
    )
    return {"ok": True}


@router.get("/events/feedback/recent")
def feedback_recent(
    limit: int = Query(20, ge=1, le=200),
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    store: FeedbackStore = Depends(get_feedback_store),
) -> Dict[str, Any]:
    user_id = _require_user_id(x_user_id)
    events = store.recent(user_id=user_id, limit=limit)

    def _ev_to_dict(e: FeedbackEvent) -> Dict[str, Any]:
        return {"user_id": e.user_id, "track_id": e.track_id, "event_type": e.event_type, "ts": e.ts}

    return {"ok": True, "n": len(events), "events": [_ev_to_dict(e) for e in events]}


@router.get("/events/feedback/stats")
def feedback_stats(
    days: int = Query(7, ge=1, le=365),
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    store: FeedbackStore = Depends(get_feedback_store),
) -> Dict[str, Any]:
    user_id = _require_user_id(x_user_id)
    raw = store.stats(user_id=user_id, days=days)

    # Tests expect missing types to be present as 0 (esp. "play")
    counts = {k: int(raw.get(k, 0)) for k in ["like", "dislike", "skip", "play"]}

    return {"ok": True, "days": int(days), "counts": counts}