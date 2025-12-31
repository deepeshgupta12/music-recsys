from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class TagStoreConfig:
    db_path: Optional[Union[str, Path]] = None
    table_name: str = "tags"


@dataclass(frozen=True)
class TagRow:
    track_id: str
    provider: str
    tags: Dict[str, Any]
    updated_at: float


class TagStore:
    """
    SQLite-backed store for per-track tagging bundles.

    This store is intentionally permissive about input types:
    - TagRow
    - "TrackTagBundle"-like objects (from heuristic tagger)
    - dict payloads

    It normalizes into TagRow and persists:
      track_id (PK), provider, tags_json, updated_at
    """

    def __init__(self, cfg: Optional[TagStoreConfig] = None) -> None:
        self.cfg = cfg or TagStoreConfig()
        db_path: Path

        if self.cfg.db_path is None:
            db_path = self._repo_root() / "data" / "tags.sqlite3"
        else:
            db_path = Path(self.cfg.db_path)  # IMPORTANT: accept string paths from tests

        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._ensure_schema()

    def _repo_root(self) -> Path:
        """
        Resolve repo root by walking upwards until we find .git or pyproject.toml.
        Falls back to 4 levels up from this file (src/musicrec/storage/tag_store.py -> repo).
        """
        here = Path(__file__).resolve()
        for p in [here] + list(here.parents):
            if (p / ".git").exists() or (p / "pyproject.toml").exists():
                return p
        # fallback: .../src/musicrec/storage/tag_store.py -> parents[4] ~ repo
        try:
            return here.parents[4]
        except Exception:
            return Path.cwd()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.cfg.table_name} (
                    track_id   TEXT PRIMARY KEY,
                    provider   TEXT NOT NULL,
                    tags_json  TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                """
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.cfg.table_name}_updated_at ON {self.cfg.table_name}(updated_at);"
            )
            conn.commit()

    # -------------------------
    # Public API
    # -------------------------

    def stats(self) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) AS n, COALESCE(MAX(updated_at), 0.0) AS mx FROM {self.cfg.table_name};"
            ).fetchone()
        return {"rows": int(row["n"]), "last_updated_at": float(row["mx"])}

    def get(self, track_id: str) -> Optional[TagRow]:
        tid = str(track_id).strip()
        if not tid:
            return None
        with self._connect() as conn:
            r = conn.execute(
                f"SELECT track_id, provider, tags_json, updated_at FROM {self.cfg.table_name} WHERE track_id = ?;",
                (tid,),
            ).fetchone()
        if not r:
            return None
        return TagRow(
            track_id=str(r["track_id"]),
            provider=str(r["provider"]),
            tags=json.loads(r["tags_json"]) if r["tags_json"] else {},
            updated_at=float(r["updated_at"]),
        )

    def batch_get(self, track_ids: Sequence[str]) -> Dict[str, TagRow]:
        ids = [str(x).strip() for x in (track_ids or []) if str(x).strip()]
        if not ids:
            return {}
        qmarks = ",".join(["?"] * len(ids))
        out: Dict[str, TagRow] = {}
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT track_id, provider, tags_json, updated_at FROM {self.cfg.table_name} WHERE track_id IN ({qmarks});",
                tuple(ids),
            ).fetchall()
        for r in rows:
            tr = TagRow(
                track_id=str(r["track_id"]),
                provider=str(r["provider"]),
                tags=json.loads(r["tags_json"]) if r["tags_json"] else {},
                updated_at=float(r["updated_at"]),
            )
            out[tr.track_id] = tr
        return out

    def upsert(self, row: Any) -> None:
        self.upsert_many([row])

    def upsert_many(self, rows: Iterable[Any]) -> None:
        normalized: List[TagRow] = [self._coerce_to_tag_row(r) for r in rows]
        payload: List[Tuple[str, str, str, float]] = [
            (
                r.track_id,
                r.provider,
                json.dumps(r.tags, ensure_ascii=False),
                float(r.updated_at),
            )
            for r in normalized
        ]
        with self._connect() as conn:
            conn.executemany(
                f"""
                INSERT INTO {self.cfg.table_name} (track_id, provider, tags_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(track_id) DO UPDATE SET
                    provider   = excluded.provider,
                    tags_json  = excluded.tags_json,
                    updated_at = excluded.updated_at;
                """,
                payload,
            )
            conn.commit()

    # -------------------------
    # Normalization helpers
    # -------------------------

    def _coerce_to_tag_row(self, obj: Any) -> TagRow:
        """
        Accept:
          - TagRow
          - dict
          - TrackTagBundle-like objects

        Expected minimum: track_id must exist.
        """
        if isinstance(obj, TagRow):
            return obj

        # dict input
        if isinstance(obj, dict):
            track_id = str(obj.get("track_id", "")).strip()
            if not track_id:
                raise ValueError("TagStore.upsert: dict row missing track_id")

            provider = (
                obj.get("provider")
                or obj.get("source")
                or obj.get("model_id")
                or obj.get("tagger")
                or "heuristic"
            )
            provider = str(provider).strip() or "heuristic"

            tags_val = obj.get("tags")
            if tags_val is None:
                # If dict is itself a "bundle", store full dict as tags
                tags_val = obj
            if not isinstance(tags_val, dict):
                # last resort: wrap
                tags_val = {"value": tags_val}

            updated_at = obj.get("updated_at") or obj.get("created_at") or time.time()
            return TagRow(track_id=track_id, provider=provider, tags=tags_val, updated_at=float(updated_at))

        # object input
        track_id = str(getattr(obj, "track_id", "")).strip()
        if not track_id:
            raise ValueError("TagStore.upsert: row missing track_id")

        provider = getattr(obj, "provider", None)
        if provider is None:
            provider = getattr(obj, "source", None)
        if provider is None:
            provider = getattr(obj, "model_id", None)
        if provider is None:
            provider = getattr(obj, "tagger", None)
        provider = str(provider).strip() if provider is not None else "heuristic"
        if not provider:
            provider = "heuristic"

        tags_val = getattr(obj, "tags", None)
        if tags_val is None:
            # many bundles expose to_dict()
            to_dict = getattr(obj, "to_dict", None)
            if callable(to_dict):
                try:
                    tags_val = to_dict()
                except Exception:
                    tags_val = {"track_id": track_id}
            else:
                # last resort: best-effort repr
                tags_val = {"track_id": track_id}

        if not isinstance(tags_val, dict):
            tags_val = {"value": tags_val}

        updated_at = getattr(obj, "updated_at", None)
        if updated_at is None:
            updated_at = getattr(obj, "created_at", None)
        if updated_at is None:
            updated_at = time.time()

        return TagRow(track_id=track_id, provider=provider, tags=tags_val, updated_at=float(updated_at))