from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from musicrec.tagging.heuristic import heuristic_tags_from_features
from musicrec.tagging.schema import TrackTagBundle


@dataclass(frozen=True)
class TaggerConfig:
    provider: str = "heuristic"  # heuristic | openai
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4.1-mini"  # placeholder; only used if provider=openai


class Tagger:
    def __init__(self, cfg: Optional[TaggerConfig] = None) -> None:
        self.cfg = cfg or TaggerConfig()

    def tag_track(self, track_id: str, features: Dict[str, Any], meta: Dict[str, Any]) -> TrackTagBundle:
        if self.cfg.provider == "heuristic":
            return heuristic_tags_from_features(track_id=track_id, features=features, meta=meta)

        # Optional provider stub (kept strict so tests never depend on it)
        if self.cfg.provider == "openai":
            return self._tag_with_openai(track_id=track_id, features=features, meta=meta)

        raise ValueError(f"Unknown provider: {self.cfg.provider}")

    def _tag_with_openai(self, track_id: str, features: Dict[str, Any], meta: Dict[str, Any]) -> TrackTagBundle:
        """
        Intentionally not wired to an SDK to keep repo minimal.
        We’ll implement actual HTTP call when you’re ready (V1.5.1.1 / follow-up step).
        """
        raise NotImplementedError(
            "OpenAI provider not wired yet. Use provider=heuristic for now. "
            "We’ll add the real OpenAI call in the next sub-step once you confirm."
        )