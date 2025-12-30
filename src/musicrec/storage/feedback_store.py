from __future__ import annotations

import sqlite3
import threading
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple


class FeedbackStore:
    """
    SQLite-backed feedback/event store.

    Stores events like: like, dislike, skip, play.
    Used by:
      - POST /events/feedback
      - GET  /events/feedback/recent
      - GET  /events/feedback/stats
      - Feed personalization (suppress disliked/skipped track_ids)
    """

    VALID_EVENT_TYPES: Set[str] = {"like", "dislike", "skip", "play"}

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback_events (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id    TEXT NOT NULL,
                    track_id   TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    ts         REAL NOT NULL
                );
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_user_ts ON feedback_events(user_id, ts DESC);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_user_event_ts ON feedback_events(user_id, event_type, ts DESC);"
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                # Best-effort close; tests just require the method exists.
                pass

    # ---------- Writes ----------

    def add_event(
        self,
        user_id: str,
        track_id: str,
        event_type: str,
        ts: Optional[float] = None,
    ) -> None:
        u = (user_id or "").strip()
        t = (track_id or "").strip()
        e = (event_type or "").strip().lower()

        if not u:
            raise ValueError("user_id is required")
        if not t:
            raise ValueError("track_id is required")
        if e not in self.VALID_EVENT_TYPES:
            raise ValueError(f"invalid event_type: {e}")

        # IMPORTANT: respect provided ts (tests rely on this for windowing).
        event_ts = float(ts) if ts is not None else float(time.time())

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO feedback_events (user_id, track_id, event_type, ts)
                VALUES (?, ?, ?, ?)
                """,
                (u, t, e, event_ts),
            )
            self._conn.commit()

    # ---------- Reads ----------

    def recent_events(self, user_id: str, limit: int = 50) -> List[Dict[str, object]]:
        u = (user_id or "").strip()
        if not u:
            raise ValueError("user_id is required")

        lim = int(limit)
        if lim <= 0:
            lim = 1
        if lim > 200:
            lim = 200

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT track_id, event_type, ts
                FROM feedback_events
                WHERE user_id = ?
                ORDER BY ts DESC, id DESC
                LIMIT ?
                """,
                (u, lim),
            ).fetchall()

        return [
            {"track_id": r["track_id"], "event_type": r["event_type"], "ts": float(r["ts"])}
            for r in rows
        ]

    def stats_counts(self, user_id: str, days: int) -> Dict[str, int]:
        """
        Returns counts for each known event_type within last N days.
        Always includes all keys in VALID_EVENT_TYPES with zero defaults.
        """
        u = (user_id or "").strip()
        if not u:
            raise ValueError("user_id is required")

        d = int(days)
        if d <= 0:
            d = 1
        cutoff = float(time.time()) - (d * 86400.0)

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT event_type, COUNT(*) AS c
                FROM feedback_events
                WHERE user_id = ?
                  AND ts >= ?
                GROUP BY event_type
                """,
                (u, cutoff),
            ).fetchall()

        out = {k: 0 for k in self.VALID_EVENT_TYPES}
        for r in rows:
            et = str(r["event_type"])
            out[et] = int(r["c"])
        return out

    def suppressed_track_ids(
        self,
        user_id: str,
        event_types: Iterable[str] = ("dislike", "skip"),
        days: int = 365,
    ) -> Set[str]:
        """
        Track IDs to suppress in personalized feeds for the user.
        """
        u = (user_id or "").strip()
        if not u:
            return set()

        types = [str(x).strip().lower() for x in event_types]
        types = [x for x in types if x in self.VALID_EVENT_TYPES]
        if not types:
            return set()

        cutoff = float(time.time()) - (int(days) * 86400.0)

        placeholders = ",".join("?" for _ in types)
        params: Tuple[object, ...] = (u, *types, cutoff)

        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT DISTINCT track_id
                FROM feedback_events
                WHERE user_id = ?
                  AND event_type IN ({placeholders})
                  AND ts >= ?
                """,
                params,
            ).fetchall()

        return {str(r["track_id"]) for r in rows}