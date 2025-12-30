from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class FeedbackEvent:
    user_id: str
    track_id: str
    event_type: str  # "play" | "like" | "dislike" | "skip"
    ts: float        # unix seconds


class FeedbackStore:
    """
    Tiny SQLite-backed event store for V1.5.
    - Supports anonymous users via user_id (client generated or header-based).
    - Keeps schema minimal + indexed.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def _init_db(self) -> None:
        con = self._connect()
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    track_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    ts REAL NOT NULL
                );
                """
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_user_ts ON feedback_events(user_id, ts DESC);"
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_user_track ON feedback_events(user_id, track_id);"
            )
            con.commit()
        finally:
            con.close()

    def add_event(self, user_id: str, track_id: str, event_type: str, ts: Optional[float] = None) -> None:
        if not user_id or not user_id.strip():
            raise ValueError("user_id is required")
        if not track_id or not track_id.strip():
            raise ValueError("track_id is required")
        et = (event_type or "").strip().lower()
        if et not in {"play", "like", "dislike", "skip"}:
            raise ValueError("event_type must be one of: play, like, dislike, skip")

        t = float(ts if ts is not None else time.time())

        con = self._connect()
        try:
            con.execute(
                "INSERT INTO feedback_events(user_id, track_id, event_type, ts) VALUES (?, ?, ?, ?)",
                (user_id.strip(), track_id.strip(), et, t),
            )
            con.commit()
        finally:
            con.close()

    def recent_events(self, user_id: str, limit: int = 100) -> List[FeedbackEvent]:
        limit = max(1, min(int(limit), 1000))
        con = self._connect()
        try:
            rows = con.execute(
                """
                SELECT user_id, track_id, event_type, ts
                FROM feedback_events
                WHERE user_id = ?
                ORDER BY ts DESC
                LIMIT ?
                """,
                (user_id.strip(), limit),
            ).fetchall()
            return [
                FeedbackEvent(
                    user_id=str(r["user_id"]),
                    track_id=str(r["track_id"]),
                    event_type=str(r["event_type"]),
                    ts=float(r["ts"]),
                )
                for r in rows
            ]
        finally:
            con.close()

    def counts_by_type(self, user_id: str) -> Dict[str, int]:
        con = self._connect()
        try:
            rows = con.execute(
                """
                SELECT event_type, COUNT(1) AS c
                FROM feedback_events
                WHERE user_id = ?
                GROUP BY event_type
                """,
                (user_id.strip(),),
            ).fetchall()
            out: Dict[str, int] = {"play": 0, "like": 0, "dislike": 0, "skip": 0}
            for r in rows:
                out[str(r["event_type"])] = int(r["c"])
            return out
        finally:
            con.close()