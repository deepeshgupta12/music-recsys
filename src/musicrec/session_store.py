from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_ts(val: Union[str, datetime, None]) -> datetime:
    """Parse an ISO timestamp into a timezone-aware UTC datetime."""
    if val is None:
        return datetime.now(timezone.utc)
    if isinstance(val, datetime):
        return _ensure_utc(val)
    s = str(val)
    dt = datetime.fromisoformat(s)
    return _ensure_utc(dt)


@dataclass(frozen=True, init=False)
class SessionEvent:
    """A single session interaction event.

    Compatibility:
    - Tests and some older code pass `ts` (ISO string).
    - Internal code may pass `timestamp` (datetime or ISO string).
    - The recommender reads `event.ts`.
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
        *,
        ts: Union[str, datetime, None] = None,
        timestamp: Union[str, datetime, None] = None,
    ) -> None:
        chosen = timestamp if timestamp is not None else ts
        dt = _parse_ts(chosen)

        object.__setattr__(self, "session_id", str(session_id))
        object.__setattr__(self, "track_id", str(track_id))
        object.__setattr__(self, "event_type", str(event_type))
        object.__setattr__(self, "timestamp", dt)

    @property
    def ts(self) -> str:
        return _ensure_utc(self.timestamp).isoformat()

    @staticmethod
    def from_record(rec: Dict[str, Any]) -> "SessionEvent":
        ts_val = rec.get("ts", rec.get("timestamp"))
        return SessionEvent(
            session_id=str(rec.get("session_id", "")),
            track_id=str(rec.get("track_id", "")),
            event_type=str(rec.get("event_type", "")),
            ts=ts_val,
        )

    def to_record(self) -> Dict[str, Any]:
        # Write both keys for backward compatibility with any existing JSONL.
        iso = self.ts
        return {
            "session_id": self.session_id,
            "track_id": self.track_id,
            "event_type": self.event_type,
            "ts": iso,
            "timestamp": iso,
        }


class SessionStore:
    """JSONL-backed store of (session_id -> ordered events)."""

    def __init__(self, base_dir: Union[str, Path] = "data/sessions") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        safe = str(session_id).strip()
        return self.base_dir / f"{safe}.jsonl"

    def read_events(self, session_id: str) -> List[SessionEvent]:
        path = self._session_path(session_id)
        if not path.exists():
            return []
        events: List[SessionEvent] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                events.append(SessionEvent.from_record(rec))
        return events

    def append_event(
        self,
        session_id: str,
        track_id: str,
        event_type: str,
        ts: Optional[datetime] = None,
    ) -> int:
        event = SessionEvent(
            session_id=session_id,
            track_id=track_id,
            event_type=event_type,
            timestamp=_ensure_utc(ts) if ts is not None else datetime.now(timezone.utc),
        )
        path = self._session_path(session_id)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_record(), ensure_ascii=False) + "\n")
        return self.count_events(session_id)

    # Aliases expected by API / older code
    def add_event(self, session_id: str, track_id: str, event_type: str, ts: Optional[datetime] = None) -> int:
        return self.append_event(session_id=session_id, track_id=track_id, event_type=event_type, ts=ts)

    def log_event(self, session_id: str, track_id: str, event_type: str, ts: Optional[datetime] = None) -> int:
        return self.append_event(session_id=session_id, track_id=track_id, event_type=event_type, ts=ts)

    def get_events(self, session_id: str, limit: Optional[int] = None) -> List[SessionEvent]:
        return self.read_events(session_id=session_id, limit=limit)

    def count_events(self, session_id: str) -> int:
        return len(self.read_events(session_id))

    def get_last_seed_track_id(self, session_id: str) -> Optional[str]:
        events = self.read_events(session_id)
        if not events:
            return None
        return str(events[-1].track_id)
