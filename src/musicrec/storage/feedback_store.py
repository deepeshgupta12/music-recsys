from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set


@dataclass(frozen=True)
class FeedbackEvent:
    user_id: str
    track_id: str
    event_type: str
    ts: float


class FeedbackStore:
    """
    SQLite-backed feedback store.

    Design goals:
      - Simple, schema-safe init (CREATE TABLE IF NOT EXISTS)
      - Thread-safe enough for FastAPI TestClient (check_same_thread=False)
      - Explicit close() because tests expect it
    """

    def __init__(self, db_path: str = "data/feedback.sqlite") -> None:
        self.db_path = db_path

        # Allow ":memory:" and also allow paths without dirs
        if db_path != ":memory:":
            d = os.path.dirname(db_path)
            if d:
                os.makedirs(d, exist_ok=True)

        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute(
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
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_user_ts ON feedback_events(user_id, ts DESC);"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_user_event ON feedback_events(user_id, event_type);"
        )
        self._conn.commit()

    def close(self) -> None:
        """Tests call this in teardown."""
        try:
            self._conn.close()
        except Exception:
            # no-op close safety
            pass

    def record_event(
        self,
        *,
        user_id: str,
        track_id: str,
        event_type: str,
        ts: Optional[float] = None,
    ) -> None:
        ts_val = float(ts if ts is not None else time.time())
        self._conn.execute(
            "INSERT INTO feedback_events (user_id, track_id, event_type, ts) VALUES (?, ?, ?, ?)",
            (user_id, track_id, event_type, ts_val),
        )
        self._conn.commit()

    def recent(self, *, user_id: str, limit: int = 20) -> List[FeedbackEvent]:
        cur = self._conn.execute(
            """
            SELECT user_id, track_id, event_type, ts
            FROM feedback_events
            WHERE user_id = ?
            ORDER BY ts DESC
            LIMIT ?
            """,
            (user_id, int(limit)),
        )
        rows = cur.fetchall()
        return [
            FeedbackEvent(
                user_id=str(r[0]),
                track_id=str(r[1]),
                event_type=str(r[2]),
                ts=float(r[3]),
            )
            for r in rows
        ]

    def stats(self, *, user_id: str, days: int = 7) -> Dict[str, int]:
        """
        Returns counts grouped by event_type within last `days`.
        Note: this intentionally returns only observed keys; API layer can fill zeros.
        """
        cutoff = time.time() - (int(days) * 86400)
        cur = self._conn.execute(
            """
            SELECT event_type, COUNT(1)
            FROM feedback_events
            WHERE user_id = ? AND ts >= ?
            GROUP BY event_type
            """,
            (user_id, float(cutoff)),
        )
        out: Dict[str, int] = {}
        for et, n in cur.fetchall():
            out[str(et)] = int(n)
        return out

    def suppressed_track_ids(
        self,
        *,
        user_id: str,
        within_days: int = 365,
        event_types: Sequence[str] = ("dislike", "skip"),
    ) -> Set[str]:
        """
        Track IDs to suppress in feeds for this user.
        Default: suppress disliked + skipped tracks.
        """
        cutoff = time.time() - (int(within_days) * 86400)
        placeholders = ",".join(["?"] * len(event_types))
        params = [user_id, float(cutoff), *list(event_types)]
        cur = self._conn.execute(
            f"""
            SELECT DISTINCT track_id
            FROM feedback_events
            WHERE user_id = ? AND ts >= ? AND event_type IN ({placeholders})
            """,
            params,
        )
        return {str(r[0]) for r in cur.fetchall()}