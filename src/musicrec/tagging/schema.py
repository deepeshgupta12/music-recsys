from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class TrackTagBundle:
    """
    Structured tags for UX + filtering.
    Keep this stable and additive (don’t rename keys lightly).
    """
    track_id: str

    # Human labels
    moods: List[str]
    scenes: List[str]
    activities: List[str]
    energy_label: str  # low / medium / high
    danceability_label: str  # low / medium / high
    instrumental_label: str  # instrumental / mostly_instrumental / vocal
    tempo_label: str  # slow / mid / fast
    descriptors: List[str]  # short style adjectives
    subgenre_hints: List[str]  # lightweight; may be empty

    # Metadata passthrough (optional)
    country: Optional[str] = None
    genre: Optional[str] = None
    explicit: Optional[bool] = None
    artist_name: Optional[str] = None
    track_name: Optional[str] = None

    # Provenance
    source: str = "heuristic"  # heuristic | openai
    model_id: str = "v1.5.1-heuristic"
    created_at_ts: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TrackTagBundle":
        return TrackTagBundle(
            track_id=str(d.get("track_id", "")).strip(),
            moods=list(d.get("moods") or []),
            scenes=list(d.get("scenes") or []),
            activities=list(d.get("activities") or []),
            energy_label=str(d.get("energy_label") or "medium"),
            danceability_label=str(d.get("danceability_label") or "medium"),
            instrumental_label=str(d.get("instrumental_label") or "vocal"),
            tempo_label=str(d.get("tempo_label") or "mid"),
            descriptors=list(d.get("descriptors") or []),
            subgenre_hints=list(d.get("subgenre_hints") or []),
            country=d.get("country"),
            genre=d.get("genre"),
            explicit=d.get("explicit"),
            artist_name=d.get("artist_name"),
            track_name=d.get("track_name"),
            source=str(d.get("source") or "heuristic"),
            model_id=str(d.get("model_id") or "v1.5.1-heuristic"),
            created_at_ts=d.get("created_at_ts"),
        )