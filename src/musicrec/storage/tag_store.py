from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from musicrec.tagging.schema import TrackTagBundle


@dataclass
class TagStoreConfig:
    db_path: str = "runtime/tags.db"
    table: str = "track_tags"


class TagStore:
    """
    SQLite-backed store for per-track tag bundles.

    Table schema:
      track_id TEXT PRIMARY KEY
      tags_json TEXT (bundle.tags only)
      updated_at REAL
      source TEXT
      model_id TEXT
      payload_json TEXT (full bundle)
    """

    def __init__(self, cfg: TagStoreConfig):
        self.cfg = cfg
        os.makedirs(os.path.dirname(cfg.db_path) or ".", exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.cfg.db_path)
        con.row_factory = sqlite3.Row
        return con

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.cfg.table} (
                  track_id TEXT PRIMARY KEY,
                  tags_json TEXT NOT NULL,
                  updated_at REAL NOT NULL,
                  source TEXT,
                  model_id TEXT,
                  payload_json TEXT
                )
                """
            )
            con.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.cfg.table}_updated_at ON {self.cfg.table}(updated_at)")
            con.commit()

    def upsert(self, bundle: TrackTagBundle, *, updated_at: Optional[float] = None) -> None:
        """
        Inserts/updates a single track's tags.
        """
        ts = float(updated_at) if updated_at is not None else float(time.time())
        payload = bundle.to_dict()

        # Store "tags_json" separately for lightweight scanning/filtering.
        import json

        tags_json = json.dumps(payload.get("tags") or {}, ensure_ascii=False)
        payload_json = json.dumps(payload, ensure_ascii=False)

        with self._connect() as con:
            con.execute(
                f"""
                INSERT INTO {self.cfg.table} (track_id, tags_json, updated_at, source, model_id, payload_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(track_id) DO UPDATE SET
                  tags_json=excluded.tags_json,
                  updated_at=excluded.updated_at,
                  source=excluded.source,
                  model_id=excluded.model_id,
                  payload_json=excluded.payload_json
                """,
                (
                    bundle.track_id,
                    tags_json,
                    ts,
                    getattr(bundle, "source", None),
                    getattr(bundle, "model_id", None),
                    payload_json,
                ),
            )
            con.commit()

    def get(self, track_id: str) -> Optional[TrackTagBundle]:
        track_id = (track_id or "").strip()
        if not track_id:
            return None

        with self._connect() as con:
            row = con.execute(
                f"SELECT payload_json FROM {self.cfg.table} WHERE track_id = ?",
                (track_id,),
            ).fetchone()

        if not row:
            return None

        raw = row["payload_json"]
        if not raw:
            return None

        import json

        try:
            payload = json.loads(raw)
        except Exception:
            return None

        try:
            return TrackTagBundle.from_dict(payload)
        except Exception:
            return None

    def batch_get(self, track_ids: Iterable[str]) -> Dict[str, TrackTagBundle]:
        """
        Returns a map: track_id -> TrackTagBundle for those present.
        """
        ids = [((x or "").strip()) for x in (track_ids or [])]
        ids = [x for x in ids if x]
        if not ids:
            return {}

        # SQLite parameter limit is typically 999.
        # Chunk safely to avoid issues.
        out: Dict[str, TrackTagBundle] = {}

        def chunks(xs: list[str], n: int) -> Iterable[list[str]]:
            for i in range(0, len(xs), n):
                yield xs[i : i + n]

        with self._connect() as con:
            for part in chunks(ids, 800):
                qmarks = ",".join(["?"] * len(part))
                rows = con.execute(
                    f"SELECT track_id, payload_json FROM {self.cfg.table} WHERE track_id IN ({qmarks})",
                    tuple(part),
                ).fetchall()

                import json

                for r in rows:
                    tid = (r["track_id"] or "").strip()
                    raw = r["payload_json"]
                    if not tid or not raw:
                        continue
                    try:
                        payload = json.loads(raw)
                        bundle = TrackTagBundle.from_dict(payload)
                        out[tid] = bundle
                    except Exception:
                        continue

        return out

    def stats(self) -> Dict[str, Any]:
        """
        Lightweight store stats used by /tags/stats.
        """
        with self._connect() as con:
            row = con.execute(f"SELECT COUNT(*) AS n, MAX(updated_at) AS mx FROM {self.cfg.table}").fetchone()
        return {
            "rows": int(row["n"] or 0),
            "last_updated_at": float(row["mx"] or 0.0),
        }

    def count_rows(self) -> int:
        with self._connect() as con:
            row = con.execute(f"SELECT COUNT(*) AS n FROM {self.cfg.table}").fetchone()
        return int(row["n"] or 0)

    def count_distinct_track_ids(self) -> int:
        with self._connect() as con:
            row = con.execute(f"SELECT COUNT(DISTINCT track_id) AS n FROM {self.cfg.table}").fetchone()
        return int(row["n"] or 0)