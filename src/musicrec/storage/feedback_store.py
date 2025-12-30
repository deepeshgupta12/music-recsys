from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class FeedbackEvent:
    user_id: str
    track_id: str
    event_type: str  # play | like | dislike | skip
    ts: float        # unix seconds (UTC)


class FeedbackStore:
    """
    SQLite-backed event store for feedback signals.

    Supports:
    - insert events (optionally with ts)
    - recent events (desc by time)
    - counts by type within a time window (last N days)

    Schema strategy:
    - New schema uses ts_utc + meta_json
    - If an older DB exists with `ts` column, we migrate by adding ts_utc and backfilling.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

        # Ensure parent directory exists
        parent = Path(db_path).expanduser().resolve().parent
        parent.mkdir(parents=True, exist_ok=True)

        # Flags after init/migration
        self._has_ts_utc = False
        self._has_ts = False
        self._has_meta_json = False

        self._init_or_migrate()

    def close(self) -> None:
        """
        For compatibility with tests/fixtures.
        We use short-lived connections per call, so nothing persistent to close.
        """
        return

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def _table_cols(self, con: sqlite3.Connection, table: str) -> List[str]:
        rows = con.execute(f"PRAGMA table_info({table});").fetchall()
        return [str(r["name"]) for r in rows]

    def _init_or_migrate(self) -> None:
        with self._connect() as con:
            # Create the "new" table if it doesn't exist
            con.execute(
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

            # Inspect actual columns (because CREATE TABLE IF NOT EXISTS won't change old DBs)
            cols = set(self._table_cols(con, "feedback_events"))

            # Handle legacy schema: if some DB was created earlier as (.., ts REAL NOT NULL)
            # Add missing columns + backfill.
            if "ts" in cols and "ts_utc" not in cols:
                con.execute("ALTER TABLE feedback_events ADD COLUMN ts_utc REAL;")
                con.execute("UPDATE feedback_events SET ts_utc = ts WHERE ts_utc IS NULL;")
                cols.add("ts_utc")

            if "meta_json" not in cols:
                # SQLite requires DEFAULT if adding NOT NULL column
                con.execute("ALTER TABLE feedback_events ADD COLUMN meta_json TEXT NOT NULL DEFAULT '{}';")
                cols.add("meta_json")

            # Remember capabilities
            self._has_ts_utc = "ts_utc" in cols
            self._has_ts = "ts" in cols
            self._has_meta_json = "meta_json" in cols

            # Indexes: always use ts_utc if present, else fallback to ts
            time_col = "ts_utc" if self._has_ts_utc else "ts"

            con.execute(
                f"CREATE INDEX IF NOT EXISTS idx_feedback_user_ts ON feedback_events(user_id, {time_col} DESC);"
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_user_track ON feedback_events(user_id, track_id);"
            )
            con.execute(
                f"CREATE INDEX IF NOT EXISTS idx_feedback_ts ON feedback_events({time_col} DESC);"
            )
            con.commit()

    def add_event(
        self,
        user_id: str,
        track_id: str,
        event_type: str,
        ts: Optional[float] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        user_id = (user_id or "").strip()
        track_id = (track_id or "").strip()
        et = (event_type or "").strip().lower()

        if not user_id:
            raise ValueError("user_id is required")
        if not track_id:
            raise ValueError("track_id is required")
        if et not in {"play", "like", "dislike", "skip"}:
            raise ValueError("event_type must be one of: play, like, dislike, skip")

        t = float(ts if ts is not None else time.time())
        meta_json = json.dumps(meta or {}, ensure_ascii=False)

        with self._connect() as con:
            cols = set(self._table_cols(con, "feedback_events"))
            if "ts_utc" in cols:
                con.execute(
                    """
                    INSERT INTO feedback_events(user_id, track_id, event_type, ts_utc, meta_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (user_id, track_id, et, t, meta_json),
                )
            else:
                # legacy insert
                con.execute(
                    """
                    INSERT INTO feedback_events(user_id, track_id, event_type, ts)
                    VALUES (?, ?, ?, ?)
                    """,
                    (user_id, track_id, et, t),
                )
                # best-effort backfill if meta_json exists
                if "meta_json" in cols and "id" in cols:
                    last_id = con.execute("SELECT last_insert_rowid() AS rid;").fetchone()["rid"]
                    con.execute(
                        "UPDATE feedback_events SET meta_json = ? WHERE id = ?",
                        (meta_json, int(last_id)),
                    )
                if "ts_utc" in cols and "id" in cols:
                    last_id = con.execute("SELECT last_insert_rowid() AS rid;").fetchone()["rid"]
                    con.execute(
                        "UPDATE feedback_events SET ts_utc = ? WHERE id = ?",
                        (t, int(last_id)),
                    )

            con.commit()

    def recent_events(self, user_id: str, limit: int = 100) -> List[FeedbackEvent]:
        user_id = (user_id or "").strip()
        if not user_id:
            raise ValueError("user_id is required")

        limit = max(1, min(int(limit), 1000))

        with self._connect() as con:
            cols = set(self._table_cols(con, "feedback_events"))
            time_col = "ts_utc" if "ts_utc" in cols else "ts"

            rows = con.execute(
                f"""
                SELECT user_id, track_id, event_type, {time_col} AS ts
                FROM feedback_events
                WHERE user_id = ?
                ORDER BY {time_col} DESC
                LIMIT ?
                """,
                (user_id, limit),
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

    def counts_last_days(self, user_id: str, days: int = 30) -> Dict[str, int]:
        user_id = (user_id or "").strip()
        if not user_id:
            raise ValueError("user_id is required")

        days = max(1, min(int(days), 3650))
        since = time.time() - float(days) * 86400.0

        with self._connect() as con:
            cols = set(self._table_cols(con, "feedback_events"))
            time_col = "ts_utc" if "ts_utc" in cols else "ts"

            rows = con.execute(
                f"""
                SELECT event_type, COUNT(1) AS c
                FROM feedback_events
                WHERE user_id = ?
                  AND {time_col} >= ?
                GROUP BY event_type
                """,
                (user_id, since),
            ).fetchall()

            out: Dict[str, int] = {"play": 0, "like": 0, "dislike": 0, "skip": 0}
            for r in rows:
                out[str(r["event_type"])] = int(r["c"])
            return out