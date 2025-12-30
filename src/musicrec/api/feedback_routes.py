from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel

from musicrec.storage.feedback_store import FeedbackStore

router = APIRouter(prefix="/events", tags=["events"])

VALID_EVENT_TYPES = {"like", "dislike", "skip", "play"}


class FeedbackIn(BaseModel):
    track_id: str
    event_type: str
    ts: Optional[float] = None


@lru_cache(maxsize=1)
def _default_store() -> FeedbackStore:
    env_path = os.getenv("MUSICREC_FEEDBACK_DB_PATH")
    if env_path:
        return FeedbackStore(env_path)

    repo_root = Path(__file__).resolve().parents[3]
    cache_dir = repo_root / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return FeedbackStore(str(cache_dir / "feedback.sqlite"))


def get_feedback_store() -> FeedbackStore:
    return _default_store()


def _require_user_id(x_user_id: Optional[str]) -> str:
    if x_user_id is None or not x_user_id.strip():
        raise HTTPException(status_code=400, detail="X-User-Id header required")
    return x_user_id.strip()


@router.post("/feedback")
def post_feedback(
    payload: FeedbackIn,
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    store: FeedbackStore = Depends(get_feedback_store),
) -> Dict[str, Any]:
    user_id = _require_user_id(x_user_id)

    track_id = (payload.track_id or "").strip()
    if not track_id:
        raise HTTPException(status_code=400, detail="track_id required")

    event_type = (payload.event_type or "").strip().lower()
    if event_type not in VALID_EVENT_TYPES:
        raise HTTPException(status_code=400, detail="invalid event_type")

    store.add_event(user_id=user_id, track_id=track_id, event_type=event_type, ts=payload.ts)
    return {"ok": True}


@router.get("/feedback/recent")
def feedback_recent(
    limit: int = Query(default=50, ge=1, le=200),
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    store: FeedbackStore = Depends(get_feedback_store),
) -> Dict[str, Any]:
    user_id = _require_user_id(x_user_id)
    events = store.recent_events(user_id=user_id, limit=limit)
    return {"ok": True, "n": len(events), "events": events}


@router.get("/feedback/stats")
def feedback_stats(
    days: int = Query(default=7, ge=1, le=3650),
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    store: FeedbackStore = Depends(get_feedback_store),
) -> Dict[str, Any]:
    user_id = _require_user_id(x_user_id)
    counts = store.stats_counts(user_id=user_id, days=days)
    return {"ok": True, "days": int(days), "counts": counts}