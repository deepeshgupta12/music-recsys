from __future__ import annotations

import sqlite3
import threading
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple


class FeedbackStore:
    """
    SQLite-backed feedback/event store.

    Important behavior for tests:
      - recent_events() returns recent UNIQUE track_ids (most-recent first)
      - stats_counts() counts UNIQUE track_ids per event_type within the window
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
            cur.execute("CREATE INDEX IF NOT EXISTS idx_feedback_user_ts ON feedback_events(user_id, ts DESC);")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_user_event_ts ON feedback_events(user_id, event_type, ts DESC);"
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
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

        # Respect provided ts (tests rely on this).
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
        """
        Return recent UNIQUE track_ids (most-recent first).
        This matches tests that treat "recent" as a recent-track list.
        """
        u = (user_id or "").strip()
        if not u:
            raise ValueError("user_id is required")

        lim = int(limit)
        if lim <= 0:
            lim = 1
        if lim > 200:
            lim = 200

        # Fetch more than needed, then dedup by track_id in Python.
        fetch_lim = min(2000, lim * 20)

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT track_id, event_type, ts
                FROM feedback_events
                WHERE user_id = ?
                ORDER BY ts DESC, id DESC
                LIMIT ?
                """,
                (u, fetch_lim),
            ).fetchall()

        out: List[Dict[str, object]] = []
        seen: Set[str] = set()

        for r in rows:
            tid = str(r["track_id"])
            if tid in seen:
                continue
            seen.add(tid)
            out.append(
                {"track_id": tid, "event_type": str(r["event_type"]), "ts": float(r["ts"])}
            )
            if len(out) >= lim:
                break

        return out

    def stats_counts(self, user_id: str, days: int) -> Dict[str, int]:
        """
        Count UNIQUE track_ids per event_type within last N days.
        Tests expect duplicates for same track to not inflate counts.
        Always includes all keys with zero defaults.
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
                SELECT event_type, COUNT(DISTINCT track_id) AS c
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