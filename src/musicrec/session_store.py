"""Simple on-disk session event store.

We keep V0 intentionally lightweight: one JSONL file per session_id under base_dir.

Public API (used by FastAPI):
- append_event/add_event/log_event(...)
- get_events(session_id, limit=None) -> list[dict]
- events_count(session_id) -> int

We also expose typed helpers for internal use:
- read(session_id, limit=None) -> list[SessionEvent]
- seen_track_ids(session_id) -> set[str]
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Union
import json


@dataclass(frozen=True, init=False)
class SessionEvent:
    """A single user event inside a session.

    Tests (and our JSONL storage) use the key `ts` for the timestamp.
    Internally we keep it as `timestamp: datetime`.
    """

    session_id: str
    track_id: str
    event_type: str
    timestamp: datetime

    def __init__(
        self,
        session_id: str,
        track_id: str,
        event_type: str,
        ts: Union[str, datetime, None] = None,
        timestamp: Union[str, datetime, None] = None,
    ) -> None:
        # Accept both `ts` (preferred) and `timestamp` (legacy/internal).
        ts_val = ts if ts is not None else timestamp
        if ts_val is None:
            dt = datetime.now(timezone.utc)
        elif isinstance(ts_val, datetime):
            dt = ts_val
        else:
            s = str(ts_val).replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        object.__setattr__(self, "session_id", str(session_id))
        object.__setattr__(self, "track_id", str(track_id))
        object.__setattr__(self, "event_type", str(event_type))
        object.__setattr__(self, "timestamp", dt)

    def to_record(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "track_id": self.track_id,
            "event_type": self.event_type,
            "ts": self.timestamp.isoformat(),
        }

    @staticmethod
    def from_record(rec: Dict[str, Any]) -> "SessionEvent":
        # Support both legacy keys (timestamp) and newer key (ts)
        ts_val = rec.get("ts", rec.get("timestamp"))
        return SessionEvent(
            session_id=str(rec.get("session_id", "")),
            track_id=str(rec.get("track_id", "")),
            event_type=str(rec.get("event_type", "")),
            timestamp=ts_val,
        )


class SessionStore:
    """JSONL-backed store of (session_id -> ordered events)."""

    def __init__(self, base_dir: Union[str, Path]):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ---------- low-level helpers ----------

    def _path(self, session_id: str) -> Path:
        # keep filenames safe & simple
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in session_id)
        return self.base_dir / f"{safe}.jsonl"

    def _append_line(self, session_id: str, rec: Dict[str, Any]) -> None:
        p = self._path(session_id)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---------- public API used by FastAPI ----------

    def append_event(
        self,
        session_id: str,
        track_id: str,
        event_type: str,
        ts: Optional[datetime] = None,
    ) -> int:
        if not session_id or not track_id or not event_type:
            raise ValueError("session_id, track_id, event_type are required")

        ts = ts or datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        ev = SessionEvent(session_id=session_id, track_id=track_id, event_type=event_type, timestamp=ts)
        self._append_line(session_id, ev.to_record())
        return self.events_count(session_id)

    # Aliases — earlier iterations used different names
    def add_event(self, session_id: str, track_id: str, event_type: str, ts: Optional[datetime] = None) -> int:
        return self.append_event(session_id=session_id, track_id=track_id, event_type=event_type, ts=ts)

    def log_event(self, session_id: str, track_id: str, event_type: str, ts: Optional[datetime] = None) -> int:
        return self.append_event(session_id=session_id, track_id=track_id, event_type=event_type, ts=ts)

    def get_events(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        return [e.to_record() for e in self.read(session_id, limit=limit)]

    def events_count(self, session_id: str) -> int:
        return len(self.read(session_id))

    # ---------- typed convenience helpers ----------

    def read(self, session_id: str, limit: Optional[int] = None) -> List[SessionEvent]:
        p = self._path(session_id)
        if not p.exists():
            return []

        events: List[SessionEvent] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    # ensure session_id is present
                    if "session_id" not in rec:
                        rec["session_id"] = session_id
                    events.append(SessionEvent.from_record(rec))
                except Exception:
                    # best-effort: skip malformed line
                    continue

        if limit is None or limit <= 0:
            return events
        return events[-limit:]

    def seen_track_ids(self, session_id: str) -> Set[str]:
        return {e.track_id for e in self.read(session_id)}

    def write(self, session_id: str, events: Sequence[SessionEvent]) -> None:
        """Overwrite a session file with provided events."""
        p = self._path(session_id)
        with p.open("w", encoding="utf-8") as f:
            for e in events:
                f.write(json.dumps(e.to_record(), ensure_ascii=False) + "\n")
