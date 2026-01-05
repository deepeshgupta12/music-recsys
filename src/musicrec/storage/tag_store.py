from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from musicrec.tagging.schema import TrackTagBundle


@dataclass(frozen=True)
class TagStoreConfig:
    db_path: str
    table_name: str = "track_tags"

    # Optional explicit override; if None, we auto-detect.
    payload_col: Optional[str] = None


class TagStore:
    """
    SQLite-backed tag store.

    IMPORTANT: We support legacy schema variants:
      - column name: tags_json (newer)
      - column name: payload_json (older)
    """

    def __init__(self, cfg: TagStoreConfig):
        self.cfg = cfg
        self._db_path = Path(cfg.db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure table exists (with modern schema), but don’t break old ones.
        self._ensure_table_minimum()

        # Resolve which payload column exists (tags_json vs payload_json).
        self._payload_col = self._resolve_payload_col()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self._db_path))
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        return con

    def _table_exists(self, con: sqlite3.Connection) -> bool:
        row = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (self.cfg.table_name,),
        ).fetchone()
        return bool(row)

    def _get_columns(self, con: sqlite3.Connection) -> List[str]:
        rows = con.execute(f"PRAGMA table_info({self.cfg.table_name})").fetchall()
        # PRAGMA table_info: cid, name, type, notnull, dflt_value, pk
        return [r[1] for r in rows]

    def _ensure_table_minimum(self) -> None:
        """
        Create table if missing. If present, ensure at least updated_at exists
        (non-breaking migrations only).
        """
        with self._connect() as con:
            if not self._table_exists(con):
                con.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.cfg.table_name} (
                        track_id TEXT PRIMARY KEY,
                        tags_json TEXT NOT NULL,
                        updated_at REAL NOT NULL
                    )
                    """
                )
                con.commit()
                return

            cols = set(self._get_columns(con))
            # Non-breaking: add updated_at if missing.
            if "updated_at" not in cols:
                con.execute(f"ALTER TABLE {self.cfg.table_name} ADD COLUMN updated_at REAL")
                con.execute(f"UPDATE {self.cfg.table_name} SET updated_at = ? WHERE updated_at IS NULL", (time.time(),))
                con.commit()

    def _resolve_payload_col(self) -> str:
        if self.cfg.payload_col:
            return self.cfg.payload_col

        with self._connect() as con:
            cols = set(self._get_columns(con))

        # Prefer newer name if present
        if "tags_json" in cols:
            return "tags_json"
        if "payload_json" in cols:
            return "payload_json"

        # If neither exists, schema is broken/unexpected
        raise RuntimeError(
            f"Tag store table '{self.cfg.table_name}' exists but has neither 'tags_json' nor 'payload_json'. "
            f"Columns found: {sorted(cols)}"
        )

    def _norm(self, track_id: str) -> str:
        return (track_id or "").strip().lower()

    def put_bundle(self, bundle: TrackTagBundle) -> None:
        """
        Upsert one bundle.
        Writes into whichever payload column exists.
        """
        tid = self._norm(bundle.track_id)
        if not tid:
            return
        payload = json.dumps(bundle.to_dict(), ensure_ascii=False)
        now = time.time()

        with self._connect() as con:
            con.execute(
                f"""
                INSERT INTO {self.cfg.table_name} (track_id, {self._payload_col}, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(track_id) DO UPDATE SET
                  {self._payload_col} = excluded.{self._payload_col},
                  updated_at = excluded.updated_at
                """,
                (tid, payload, now),
            )
            con.commit()

    def get(self, track_id: str) -> Optional[TrackTagBundle]:
        tid = self._norm(track_id)
        if not tid:
            return None

        with self._connect() as con:
            row = con.execute(
                f"SELECT track_id, {self._payload_col} FROM {self.cfg.table_name} WHERE track_id = ?",
                (tid,),
            ).fetchone()

        if not row:
            return None

        payload = row[1]
        try:
            d = json.loads(payload) if payload else {}
            return TrackTagBundle.from_dict(d)
        except Exception:
            return None

    def batch_get(self, track_ids: Iterable[str]) -> Dict[str, TrackTagBundle]:
        """
        Returns a map: track_id_variant -> TrackTagBundle for those present.

        We return multiple keys for convenience:
          - stored id
          - stored id lower
          - stored id upper
        so callers that accidentally mix casing still get a hit.
        """
        raw = [((x or "").strip()) for x in (track_ids or [])]
        raw = [x for x in raw if x]
        if not raw:
            return {}

        # De-dupe by normalized id for query size reduction
        norm_ids: List[str] = []
        seen = set()
        for x in raw:
            nx = self._norm(x)
            if nx and nx not in seen:
                seen.add(nx)
                norm_ids.append(nx)

        if not norm_ids:
            return {}

        out: Dict[str, TrackTagBundle] = {}

        def chunks(xs: list[str], n: int):
            for i in range(0, len(xs), n):
                yield xs[i : i + n]

        with self._connect() as con:
            for part in chunks(norm_ids, 800):
                qmarks = ",".join(["?"] * len(part))
                rows = con.execute(
                    f"SELECT track_id, {self._payload_col} FROM {self.cfg.table_name} WHERE track_id IN ({qmarks})",
                    tuple(part),
                ).fetchall()

                for track_id_db, payload in rows:
                    try:
                        d = json.loads(payload) if payload else {}
                        bundle = TrackTagBundle.from_dict(d)
                    except Exception:
                        continue

                    # Store under multiple variants
                    tid = str(track_id_db)
                    out[tid] = bundle
                    out[tid.lower()] = bundle
                    out[tid.upper()] = bundle

        return out

    def stats(self) -> Dict[str, object]:
        with self._connect() as con:
            row = con.execute(
                f"SELECT COUNT(*) as rows, MAX(updated_at) as last_updated_at FROM {self.cfg.table_name}"
            ).fetchone()

        rows = int(row[0] or 0) if row else 0
        last_updated_at = float(row[1] or 0.0) if row else 0.0
        return {"rows": rows, "last_updated_at": last_updated_at}