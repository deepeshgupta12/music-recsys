from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class SessionEvent:
    session_id: str
    track_id: str
    event_type: str  # play, like, skip, dislike, add_to_playlist
    ts: str          # ISO8601 UTC string


class SessionStore:
    """
    Simple file-based store:
      data/processed/sessions/<session_id>.jsonl
    Each line is a JSON dict with keys: session_id, track_id, event_type, ts
    """

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.sessions_dir = self.base_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, session_id: str) -> Path:
        safe = session_id.strip()
        if not safe:
            raise ValueError("session_id cannot be empty")
        return self.sessions_dir / f"{safe}.jsonl"

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def append(self, session_id: str, track_id: str, event_type: str, ts: Optional[str] = None) -> None:
        event_type = event_type.strip().lower()
        if event_type not in {"play", "like", "skip", "dislike", "add_to_playlist"}:
            raise ValueError("event_type must be one of: play, like, skip, dislike, add_to_playlist")
        track_id = str(track_id).strip()
        if not track_id:
            raise ValueError("track_id cannot be empty")

        payload = {
            "session_id": session_id,
            "track_id": track_id,
            "event_type": event_type,
            "ts": ts or self._utc_now_iso(),
        }

        p = self._path_for(session_id)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def read(self, session_id: str) -> List[SessionEvent]:
        p = self._path_for(session_id)
        if not p.exists():
            return []
        events: List[SessionEvent] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                events.append(
                    SessionEvent(
                        session_id=str(obj["session_id"]),
                        track_id=str(obj["track_id"]),
                        event_type=str(obj["event_type"]),
                        ts=str(obj["ts"]),
                    )
                )
        return events

    @staticmethod
    def seen_track_ids(events: Iterable[SessionEvent]) -> List[str]:
        seen = []
        s = set()
        for e in events:
            tid = str(e.track_id)
            if tid not in s:
                s.add(tid)
                seen.append(tid)
        return seen