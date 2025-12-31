from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union


@dataclass(frozen=True)
class TagStoreConfig:
    """
    Storage config for tags.
    db_path can be a string path (tests pass a temp path string) or a Path.
    """
    db_path: Union[str, Path, None] = None
    table_name: str = "tags"


def _repo_root() -> Path:
    """
    Best-effort repo root discovery.
    Falls back to 3 levels above this file (src/musicrec/storage).
    """
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    return here.parents[3] if len(here.parents) >= 4 else here.parent


def _bundle_to_dict(bundle: Any) -> Dict[str, Any]:
    """
    Convert TrackTagBundle-like objects to dict for persistence.
    Supports:
      - .to_dict()
      - .dict() (pydantic)
      - dataclass asdict-compatible via __dict__ fallback
      - already-a-dict
    """
    if bundle is None:
        return {}

    if isinstance(bundle, dict):
        return bundle

    to_dict = getattr(bundle, "to_dict", None)
    if callable(to_dict):
        return to_dict()

    pyd_dict = getattr(bundle, "dict", None)
    if callable(pyd_dict):
        return pyd_dict()

    # Fallback (works for simple dataclasses / objects)
    d = getattr(bundle, "__dict__", None)
    if isinstance(d, dict):
        return dict(d)

    raise TypeError(f"Unsupported tag bundle type: {type(bundle)}")


class TagStore:
    """
    SQLite-backed store for TrackTagBundle JSON payloads.

    Schema:
      track_id TEXT PRIMARY KEY
      provider TEXT
      payload_json TEXT
      updated_at REAL
    """

    def __init__(self, cfg: Optional[TagStoreConfig] = None) -> None:
        self.cfg = cfg or TagStoreConfig()

        if self.cfg.db_path is None:
            db_path = _repo_root() / "data" / "tags.sqlite3"
        else:
            db_path = Path(self.cfg.db_path) if not isinstance(self.cfg.db_path, Path) else self.cfg.db_path

        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path: Path = db_path

        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.db_path))
        con.row_factory = sqlite3.Row
        return con

    def _ensure_schema(self) -> None:
        with self._connect() as con:
            con.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.cfg.table_name} (
                    track_id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            con.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.cfg.table_name}_updated_at ON {self.cfg.table_name}(updated_at)")
            con.commit()

    def stats(self) -> Dict[str, Any]:
        with self._connect() as con:
            row = con.execute(
                f"SELECT COUNT(*) AS n, COALESCE(MAX(updated_at), 0.0) AS last FROM {self.cfg.table_name}"
            ).fetchone()
        return {"rows": int(row["n"]), "last_updated_at": float(row["last"])}

    def upsert(self, bundle: Any, provider: str = "heuristic", updated_at: Optional[float] = None) -> None:
        self.upsert_many([bundle], provider=provider, updated_at=updated_at)

    def upsert_many(
        self,
        bundles: Iterable[Any],
        provider: str = "heuristic",
        updated_at: Optional[float] = None,
    ) -> None:
        ts = float(updated_at) if updated_at is not None else float(time.time())

        payload: List[tuple] = []
        for b in bundles:
            d = _bundle_to_dict(b)
            track_id = str(d.get("track_id", "")).strip()
            if not track_id:
                # skip invalid
                continue

            # Prefer provider embedded in bundle if present, else passed provider
            prov = d.get("provider") or getattr(b, "provider", None) or provider
            prov = str(prov)

            payload_json = json.dumps(d, ensure_ascii=False)
            # updated_at: prefer bundle updated_at if present and numeric
            b_ts = d.get("updated_at") or getattr(b, "updated_at", None)
            try:
                row_ts = float(b_ts) if b_ts is not None else ts
            except Exception:
                row_ts = ts

            payload.append((track_id, prov, payload_json, row_ts))

        if not payload:
            return

        with self._connect() as con:
            con.executemany(
                f"""
                INSERT INTO {self.cfg.table_name} (track_id, provider, payload_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(track_id) DO UPDATE SET
                    provider=excluded.provider,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                payload,
            )
            con.commit()

    def get(self, track_id: str) -> Optional[Any]:
        tid = str(track_id).strip()
        if not tid:
            return None

        with self._connect() as con:
            row = con.execute(
                f"SELECT payload_json FROM {self.cfg.table_name} WHERE track_id = ?",
                (tid,),
            ).fetchone()

        if row is None:
            return None

        data = json.loads(row["payload_json"])

        # Return TrackTagBundle instance if available (tests expect .instrumental_label etc.)
        try:
            from musicrec.tagging.schema import TrackTagBundle  # type: ignore
            return TrackTagBundle.from_dict(data)
        except Exception:
            # Fallback: return raw dict if schema import isn't available
            return data

    def batch_get(self, track_ids: List[str]) -> Dict[str, Any]:
        ids = [str(t).strip() for t in (track_ids or []) if str(t).strip()]
        if not ids:
            return {}

        placeholders = ",".join(["?"] * len(ids))
        with self._connect() as con:
            rows = con.execute(
                f"SELECT track_id, payload_json FROM {self.cfg.table_name} WHERE track_id IN ({placeholders})",
                tuple(ids),
            ).fetchall()

        out: Dict[str, Any] = {}
        for r in rows:
            tid = r["track_id"]
            data = json.loads(r["payload_json"])
            try:
                from musicrec.tagging.schema import TrackTagBundle  # type: ignore
                out[tid] = TrackTagBundle.from_dict(data)
            except Exception:
                out[tid] = data
        return out