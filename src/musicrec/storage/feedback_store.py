from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class FeedbackEvent:
    user_id: str
    track_id: str
    event_type: str
    ts_utc: float
    meta: Dict[str, Any]


class FeedbackStore:
    """
    SQLite-backed event store for lightweight feedback instrumentation.

    V1.5.1: write feedback events
    V1.5.3: read-back + basic aggregation for QA/debugging
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    track_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    ts_utc REAL NOT NULL,
                    meta_json TEXT NOT NULL
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_user_ts ON feedback_events(user_id, ts_utc DESC);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_ts ON feedback_events(ts_utc DESC);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_event_type ON feedback_events(event_type);"
            )

    def add_event(
        self,
        *,
        user_id: str,
        track_id: str,
        event_type: str,
        meta: Optional[Dict[str, Any]] = None,
        ts_utc: Optional[float] = None,
    ) -> int:
        meta = meta or {}
        ts_utc = float(ts_utc if ts_utc is not None else time.time())

        meta_json = json.dumps(meta, ensure_ascii=False, separators=(",", ":"))

        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO feedback_events (user_id, track_id, event_type, ts_utc, meta_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (user_id, track_id, event_type, ts_utc, meta_json),
                )
                return int(cur.lastrowid)

    def recent_events(
        self,
        *,
        limit: int = 50,
        user_id: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Read-back for debugging/QA.
        Returns newest-first.
        """
        limit = int(max(1, min(limit, 500)))

        where: List[str] = []
        params: List[Any] = []
        if user_id:
            where.append("user_id = ?")
            params.append(user_id)
        if event_type:
            where.append("event_type = ?")
            params.append(event_type)

        where_sql = (" WHERE " + " AND ".join(where)) if where else ""

        sql = (
            "SELECT id, user_id, track_id, event_type, ts_utc, meta_json "
            "FROM feedback_events"
            f"{where_sql} "
            "ORDER BY ts_utc DESC "
            "LIMIT ?"
        )
        params.append(limit)

        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(sql, params).fetchall()

        out: List[Dict[str, Any]] = []
        for r in rows:
            try:
                meta = json.loads(r["meta_json"]) if r["meta_json"] else {}
            except Exception:
                meta = {}
            out.append(
                {
                    "id": int(r["id"]),
                    "user_id": str(r["user_id"]),
                    "track_id": str(r["track_id"]),
                    "event_type": str(r["event_type"]),
                    "ts_utc": float(r["ts_utc"]),
                    "meta": meta,
                }
            )
        return out

    def stats(
        self,
        *,
        window_days: int = 7,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Basic rollup for observability:
        - count by event_type
        - total events
        - unique tracks
        """
        window_days = int(max(1, min(window_days, 365)))
        cutoff = time.time() - (window_days * 86400)

        where: List[str] = ["ts_utc >= ?"]
        params: List[Any] = [cutoff]
        if user_id:
            where.append("user_id = ?")
            params.append(user_id)

        where_sql = " WHERE " + " AND ".join(where)

        with self._lock:
            with self._connect() as conn:
                total = conn.execute(
                    f"SELECT COUNT(1) AS c FROM feedback_events{where_sql}",
                    params,
                ).fetchone()["c"]

                uniq_tracks = conn.execute(
                    f"SELECT COUNT(DISTINCT track_id) AS c FROM feedback_events{where_sql}",
                    params,
                ).fetchone()["c"]

                rows = conn.execute(
                    f"""
                    SELECT event_type, COUNT(1) AS c
                    FROM feedback_events
                    {where_sql}
                    GROUP BY event_type
                    ORDER BY c DESC
                    """,
                    params,
                ).fetchall()

        counts = {str(r["event_type"]): int(r["c"]) for r in rows}
        return {
            "window_days": window_days,
            "cutoff_ts_utc": float(cutoff),
            "user_id": user_id,
            "total_events": int(total),
            "unique_tracks": int(uniq_tracks),
            "counts_by_event_type": counts,
        }