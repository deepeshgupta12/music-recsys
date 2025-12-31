from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from musicrec.tagging.schema import TrackTagBundle


@dataclass(frozen=True)
class TagStoreConfig:
    db_path: str = "runtime/tags.db"


class TagStore:
    """
    Simple SQLite-backed store for track tag bundles.

    Table:
      track_tags(
        track_id TEXT PRIMARY KEY,
        tags_json TEXT NOT NULL,
        updated_at REAL NOT NULL,
        source TEXT NOT NULL,
        model_id TEXT NOT NULL
      )
    """

    def __init__(self, cfg: Optional[TagStoreConfig] = None) -> None:
        self.cfg = cfg or TagStoreConfig()
        os.makedirs(os.path.dirname(self.cfg.db_path), exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.cfg.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS track_tags (
                  track_id TEXT PRIMARY KEY,
                  tags_json TEXT NOT NULL,
                  updated_at REAL NOT NULL,
                  source TEXT NOT NULL,
                  model_id TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_track_tags_updated_at ON track_tags(updated_at)")
            conn.commit()

    def upsert(self, bundle: TrackTagBundle) -> None:
        now = time.time()
        d = bundle.to_dict()
        d["created_at_ts"] = d.get("created_at_ts") or now
        tags_json = json.dumps(d, ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO track_tags(track_id, tags_json, updated_at, source, model_id)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(track_id) DO UPDATE SET
                  tags_json=excluded.tags_json,
                  updated_at=excluded.updated_at,
                  source=excluded.source,
                  model_id=excluded.model_id
                """,
                (bundle.track_id, tags_json, now, bundle.source, bundle.model_id),
            )
            conn.commit()

    def get(self, track_id: str) -> Optional[TrackTagBundle]:
        tid = (track_id or "").strip()
        if not tid:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT tags_json FROM track_tags WHERE track_id=?",
                (tid,),
            ).fetchone()
            if not row:
                return None
            d = json.loads(row["tags_json"])
            return TrackTagBundle.from_dict(d)

    def stats(self) -> Dict[str, Any]:
        with self._connect() as conn:
            n = conn.execute("SELECT COUNT(1) AS n FROM track_tags").fetchone()["n"]
            last = conn.execute("SELECT MAX(updated_at) AS t FROM track_tags").fetchone()["t"]
        return {"rows": int(n or 0), "last_updated_at": float(last) if last else None}