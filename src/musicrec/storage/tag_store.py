from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union


@dataclass(frozen=True)
class TagRow:
    track_id: str
    provider: str
    tags: Dict[str, Any]
    updated_at: float


@dataclass(frozen=True)
class TagStoreConfig:
    """
    db_path can be a Path or a str (tests pass a str).
    """
    db_path: Optional[Union[str, Path]] = None
    table_name: str = "tags"


class TagStore:
    def __init__(self, cfg: Optional[TagStoreConfig] = None) -> None:
        self.cfg = cfg or TagStoreConfig()

        raw = self.cfg.db_path
        if raw is None:
            db_path = self._repo_root() / "data" / "tags.sqlite3"
        else:
            db_path = raw if isinstance(raw, Path) else Path(str(raw))

        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path: Path = db_path

        self._init_db()

    # -------------------------
    # Write
    # -------------------------
    def upsert(self, row: TagRow) -> None:
        self.upsert_many([row])

    def upsert_many(self, rows: Sequence[TagRow]) -> None:
        if not rows:
            return
        q = f"""
        INSERT INTO {self.cfg.table_name} (track_id, provider, tags_json, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(track_id) DO UPDATE SET
          provider=excluded.provider,
          tags_json=excluded.tags_json,
          updated_at=excluded.updated_at
        """
        payload = [
            (r.track_id, r.provider, json.dumps(r.tags, ensure_ascii=False), float(r.updated_at))
            for r in rows
        ]
        with self._connect() as conn:
            conn.executemany(q, payload)
            conn.commit()

    # -------------------------
    # Read
    # -------------------------
    def get(self, track_id: str) -> Optional[TagRow]:
        tid = (track_id or "").strip()
        if not tid:
            return None
        q = f"SELECT track_id, provider, tags_json, updated_at FROM {self.cfg.table_name} WHERE track_id=?"
        with self._connect() as conn:
            cur = conn.execute(q, (tid,))
            row = cur.fetchone()
        return self._row_to_tagrow(row) if row else None

    # Back-compat
    def get_by_track_id(self, track_id: str) -> Optional[TagRow]:
        return self.get(track_id)

    def batch_get(self, track_ids: Sequence[str]) -> List[TagRow]:
        ids = [str(t).strip() for t in (track_ids or []) if str(t).strip()]
        if not ids:
            return []

        placeholders = ",".join(["?"] * len(ids))
        q = f"""
        SELECT track_id, provider, tags_json, updated_at
        FROM {self.cfg.table_name}
        WHERE track_id IN ({placeholders})
        """
        with self._connect() as conn:
            cur = conn.execute(q, ids)
            rows = cur.fetchall()

        found: Dict[str, TagRow] = {}
        for r in rows:
            tr = self._row_to_tagrow(r)
            if tr:
                found[tr.track_id] = tr

        # preserve input order
        out: List[TagRow] = []
        for tid in ids:
            tr = found.get(tid)
            if tr:
                out.append(tr)
        return out

    def stats(self) -> Dict[str, Any]:
        q_count = f"SELECT COUNT(*) FROM {self.cfg.table_name}"
        q_max = f"SELECT MAX(updated_at) FROM {self.cfg.table_name}"
        with self._connect() as conn:
            n = int(conn.execute(q_count).fetchone()[0] or 0)
            mx = conn.execute(q_max).fetchone()[0]
        last_updated_at = float(mx) if mx is not None else 0.0
        return {"rows": n, "last_updated_at": last_updated_at}

    # -------------------------
    # Internals
    # -------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        q = f"""
        CREATE TABLE IF NOT EXISTS {self.cfg.table_name} (
          track_id TEXT PRIMARY KEY,
          provider TEXT NOT NULL,
          tags_json TEXT NOT NULL,
          updated_at REAL NOT NULL
        )
        """
        with self._connect() as conn:
            conn.execute(q)
            conn.commit()

    def _row_to_tagrow(self, row: Any) -> Optional[TagRow]:
        if row is None:
            return None
        track_id = str(row["track_id"])
        provider = str(row["provider"])
        tags_json = row["tags_json"]
        updated_at = float(row["updated_at"])
        try:
            tags = json.loads(tags_json) if isinstance(tags_json, str) else dict(tags_json)
        except Exception:
            tags = {}
        return TagRow(track_id=track_id, provider=provider, tags=tags, updated_at=updated_at)

    def _repo_root(self) -> Path:
        here = Path(__file__).resolve()
        for p in [here] + list(here.parents):
            if (p / "pyproject.toml").exists() or (p / ".git").exists():
                return p
        return here.parents[4]