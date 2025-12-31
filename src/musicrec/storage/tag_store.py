from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from musicrec.tagging.schema import TrackTagBundle


PathLike = Union[str, Path]


@dataclass(frozen=True)
class TagStoreConfig:
    """
    db_path: where SQLite lives
    table_name: default matches the schema used by scripts/tagging/run_tagging.py
    """
    db_path: PathLike = "runtime/tags.db"
    table_name: str = "track_tags"


class TagStore:
    """
    SQLite-backed store for TrackTagBundle.

    Canonical table schema (matches tagging script):
      track_tags(
        track_id TEXT PRIMARY KEY,
        tags_json TEXT NOT NULL,
        updated_at REAL NOT NULL,
        source TEXT NOT NULL,
        model_id TEXT NOT NULL
      )

    Notes:
    - We store the full TrackTagBundle.to_dict() into tags_json
    - get()/batch_get() return TrackTagBundle objects
    - stats() returns {"rows": int, "last_updated_at": float}
    """

    def __init__(self, cfg: Optional[TagStoreConfig] = None) -> None:
        self.cfg = cfg or TagStoreConfig()
        self.db_path = Path(self.cfg.db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.db_path))
        con.row_factory = sqlite3.Row
        return con

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.cfg.table_name} (
                  track_id TEXT PRIMARY KEY,
                  tags_json TEXT NOT NULL,
                  updated_at REAL NOT NULL,
                  source TEXT NOT NULL,
                  model_id TEXT NOT NULL
                )
                """
            )
            con.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.cfg.table_name}_updated_at ON {self.cfg.table_name}(updated_at)"
            )
            con.commit()

    def upsert(self, bundle: TrackTagBundle) -> None:
        now = float(time.time())

        d = bundle.to_dict()
        # Keep a created timestamp if the schema supports it in the dict
        d["created_at_ts"] = d.get("created_at_ts") or now
        tags_json = json.dumps(d, ensure_ascii=False)

        source = getattr(bundle, "source", None) or "unknown"
        model_id = getattr(bundle, "model_id", None) or "unknown"

        with self._connect() as con:
            con.execute(
                f"""
                INSERT INTO {self.cfg.table_name}(track_id, tags_json, updated_at, source, model_id)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(track_id) DO UPDATE SET
                  tags_json=excluded.tags_json,
                  updated_at=excluded.updated_at,
                  source=excluded.source,
                  model_id=excluded.model_id
                """,
                (str(bundle.track_id), tags_json, now, str(source), str(model_id)),
            )
            con.commit()

    def get(self, track_id: str) -> Optional[TrackTagBundle]:
        tid = (track_id or "").strip()
        if not tid:
            return None

        with self._connect() as con:
            row = con.execute(
                f"SELECT tags_json FROM {self.cfg.table_name} WHERE track_id=?",
                (tid,),
            ).fetchone()

        if not row:
            return None

        try:
            d = json.loads(row["tags_json"])
            return TrackTagBundle.from_dict(d)
        except Exception:
            # Corrupt row should behave as missing for API/tests
            return None

    def batch_get(self, track_ids: List[str]) -> Dict[str, TrackTagBundle]:
        ids = [str(t).strip() for t in (track_ids or []) if str(t).strip()]
        if not ids:
            return {}

        placeholders = ",".join(["?"] * len(ids))
        with self._connect() as con:
            rows = con.execute(
                f"""
                SELECT track_id, tags_json
                FROM {self.cfg.table_name}
                WHERE track_id IN ({placeholders})
                """,
                tuple(ids),
            ).fetchall()

        out: Dict[str, TrackTagBundle] = {}
        for r in rows:
            tid = str(r["track_id"])
            try:
                d = json.loads(r["tags_json"])
                out[tid] = TrackTagBundle.from_dict(d)
            except Exception:
                # Skip corrupt rows
                continue
        return out

    def stats(self) -> Dict[str, Any]:
        with self._connect() as con:
            n = con.execute(f"SELECT COUNT(1) AS n FROM {self.cfg.table_name}").fetchone()["n"]
            last = con.execute(f"SELECT MAX(updated_at) AS t FROM {self.cfg.table_name}").fetchone()["t"]

        return {
            "rows": int(n or 0),
            "last_updated_at": float(last or 0.0),
        }