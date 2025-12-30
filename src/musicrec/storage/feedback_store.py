from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set


@dataclass(frozen=True)
class FeedbackEvent:
    user_id: str
    track_id: str
    event_type: str
    ts: float
    meta: Dict[str, Any]


class FeedbackStore:
    """
    SQLite-backed feedback event store.

    Design goals:
    - Simple local persistence (sqlite) for MVP
    - Safe schema initialization and mild migration for older tables
    - Test-friendly (pass a temp db path)
    """

    def __init__(self, db_path: Optional[str] = None):
        # IMPORTANT: tests call FeedbackStore(str(tmp_path/"x.sqlite")), so keep db_path positional-friendly.
        if db_path is None or not str(db_path).strip():
            db_path = os.environ.get("MUSICREC_FEEDBACK_DB_PATH", ".cache/feedback/feedback.sqlite")

        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # Keep one connection per store instance (tests expect close()).
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        # Reasonable defaults for local sqlite usage
        try:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
        except Exception:
            # Not critical for tests / some environments
            pass

        self._init_or_migrate_schema()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    # ---------------------------
    # Schema init / migrate
    # ---------------------------

    def _table_columns(self, table: str) -> Set[str]:
        cur = self._conn.execute(f"PRAGMA table_info({table});")
        return {str(r["name"]) for r in cur.fetchall()}

    def _init_or_migrate_schema(self) -> None:
        # Create table if missing (new schema uses ts_utc).
        self._conn.execute(
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

        cols = self._table_columns("feedback_events")

        # Mild migration for older variants (if you ever had ts instead of ts_utc)
        if "ts_utc" not in cols and "ts" in cols:
            self._conn.execute("ALTER TABLE feedback_events ADD COLUMN ts_utc REAL;")
            self._conn.execute("UPDATE feedback_events SET ts_utc = ts WHERE ts_utc IS NULL;")
            cols = self._table_columns("feedback_events")

        # Ensure meta_json exists (older tables may not have it)
        if "meta_json" not in cols:
            # Need DEFAULT to add NOT NULL column in sqlite
            self._conn.execute("ALTER TABLE feedback_events ADD COLUMN meta_json TEXT NOT NULL DEFAULT '{}';")
            cols = self._table_columns("feedback_events")

        # If somehow ts_utc is still missing (corrupt/very old schema), add it
        if "ts_utc" not in cols:
            self._conn.execute("ALTER TABLE feedback_events ADD COLUMN ts_utc REAL;")
            self._conn.execute("UPDATE feedback_events SET ts_utc = COALESCE(ts_utc, 0.0);")

        # Indexes (schema-safe)
        cols = self._table_columns("feedback_events")
        if "user_id" in cols and "ts_utc" in cols:
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_user_ts ON feedback_events(user_id, ts_utc DESC);"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_ts ON feedback_events(ts_utc DESC);"
            )

        self._conn.commit()

    # ---------------------------
    # Writes
    # ---------------------------

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
        event_type = (event_type or "").strip().lower()

        if not user_id:
            raise ValueError("user_id is required")
        if not track_id:
            raise ValueError("track_id is required")
        if event_type not in {"play", "like", "dislike", "skip"}:
            raise ValueError("event_type must be one of: play, like, dislike, skip")

        ts_utc = float(ts if ts is not None else time.time())
        meta_json = json.dumps(meta or {}, ensure_ascii=False)

        self._conn.execute(
            """
            INSERT INTO feedback_events (user_id, track_id, event_type, ts_utc, meta_json)
            VALUES (?, ?, ?, ?, ?);
            """,
            (user_id, track_id, event_type, ts_utc, meta_json),
        )
        self._conn.commit()

    # ---------------------------
    # Readback (V1.5.3)
    # ---------------------------

    def recent(
        self,
        user_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[FeedbackEvent]:
        limit = int(limit)
        if limit <= 0:
            return []

        if user_id:
            cur = self._conn.execute(
                """
                SELECT user_id, track_id, event_type, ts_utc, meta_json
                FROM feedback_events
                WHERE user_id = ?
                ORDER BY ts_utc DESC
                LIMIT ?;
                """,
                (user_id, limit),
            )
        else:
            cur = self._conn.execute(
                """
                SELECT user_id, track_id, event_type, ts_utc, meta_json
                FROM feedback_events
                ORDER BY ts_utc DESC
                LIMIT ?;
                """,
                (limit,),
            )

        out: List[FeedbackEvent] = []
        for r in cur.fetchall():
            try:
                meta = json.loads(r["meta_json"]) if r["meta_json"] else {}
            except Exception:
                meta = {}
            out.append(
                FeedbackEvent(
                    user_id=str(r["user_id"]),
                    track_id=str(r["track_id"]),
                    event_type=str(r["event_type"]),
                    ts=float(r["ts_utc"]),
                    meta=meta,
                )
            )
        return out

    def stats(self, last_n_days: int = 7) -> Dict[str, Any]:
        days = int(last_n_days)
        if days <= 0:
            days = 1
        since = time.time() - (days * 86400)

        cur = self._conn.execute(
            """
            SELECT event_type, COUNT(*) AS c
            FROM feedback_events
            WHERE ts_utc >= ?
            GROUP BY event_type;
            """,
            (since,),
        )
        counts = {str(r["event_type"]): int(r["c"]) for r in cur.fetchall()}

        cur2 = self._conn.execute(
            """
            SELECT COUNT(DISTINCT user_id) AS u
            FROM feedback_events
            WHERE ts_utc >= ?;
            """,
            (since,),
        )
        urow = cur2.fetchone()
        unique_users = int(urow["u"]) if urow and urow["u"] is not None else 0

        return {
            "window_days": days,
            "since_ts_utc": since,
            "counts": counts,
            "unique_users": unique_users,
        }

    # ---------------------------
    # Personalization helper (V1.5.4)
    # ---------------------------

    def suppressed_track_ids(
        self,
        user_id: str,
        event_types: Sequence[str] = ("dislike", "skip"),
        window_days: int = 30,
        limit: int = 5000,
    ) -> Set[str]:
        """
        Return track_ids a user has negatively signaled recently (dislike/skip).
        Used by feeds to suppress items.
        """
        user_id = (user_id or "").strip()
        if not user_id:
            return set()

        types = [str(t).strip().lower() for t in event_types if str(t).strip()]
        if not types:
            return set()

        days = int(window_days)
        if days <= 0:
            days = 1

        since = time.time() - (days * 86400)
        qmarks = ",".join(["?"] * len(types))

        cur = self._conn.execute(
            f"""
            SELECT DISTINCT track_id
            FROM feedback_events
            WHERE user_id = ?
              AND ts_utc >= ?
              AND event_type IN ({qmarks})
            LIMIT ?;
            """,
            (user_id, since, *types, int(limit)),
        )
        return {str(r["track_id"]) for r in cur.fetchall() if r["track_id"]}