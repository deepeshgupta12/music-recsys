from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


__all__ = [
    "PersonalizationConfig",
    "reorder_section_reorder_only",
    "apply_reorder_only_personalization",
    "reorder_only_personalize_sections",
]


@dataclass(frozen=True)
class PersonalizationConfig:
    """
    V1.5.6 Step 1.4: Reorder-only personalization

    - Never add/remove items from a rail
    - Only reorder items inside rails
    - Uses recent feedback events to create per-track boosts
    """

    # Item field names
    track_id_key: str = "track_id"
    score_key: str = "score"

    # Boost weights derived from feedback events
    like_boost: float = 5.0
    play_boost: float = 2.0

    # Which rails to apply reorder to (default: typical home rails)
    # NOTE: we intentionally do NOT reorder "for_you" here.
    default_section_names: Tuple[str, ...] = ("top", "rising", "new_releases", "instrumental", "explicit_safe")

    # Stable ordering for ties
    stable: bool = True


def _safe_track_id(it: Any, track_id_key: str) -> str:
    if isinstance(it, dict):
        v = it.get(track_id_key)
        return "" if v is None else str(v).strip()
    return ""


def _safe_score(it: Any, score_key: str) -> float:
    if not isinstance(it, dict):
        return 0.0
    v = it.get(score_key, 0.0)
    try:
        return float(v)
    except Exception:
        return 0.0


def reorder_section_reorder_only(
    items: List[Any],
    *,
    track_boosts: Optional[Dict[str, float]] = None,
    cfg: Optional[PersonalizationConfig] = None,
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Reorder a single rail using track_boosts.

    Tests expect:
      - Accepts keyword 'track_boosts'
      - Returns dbg containing 'boosted_items' and 'moved'
      - When no boosts, list remains unchanged
    """
    cfg = cfg or PersonalizationConfig()
    boosts = track_boosts or {}

    dbg: Dict[str, Any] = {
        "boosted_items": 0,
        "applied": False,
        "moved": 0,
    }

    if not isinstance(items, list) or len(items) <= 1:
        return items, dbg

    # Count boosted items present in this section
    if boosts:
        present_ids = [_safe_track_id(x, cfg.track_id_key) for x in items]
        dbg["boosted_items"] = sum(1 for tid in present_ids if tid and float(boosts.get(tid, 0.0)) != 0.0)
    else:
        dbg["boosted_items"] = 0
        return items, dbg  # stable when no boosts

    # If nothing in this rail is boosted, return unchanged
    if dbg["boosted_items"] == 0:
        return items, dbg

    # Score = base_score + boost, ties stable by original index
    scored: List[Tuple[float, int, Any]] = []
    for idx, it in enumerate(items):
        tid = _safe_track_id(it, cfg.track_id_key)
        base = _safe_score(it, cfg.score_key)
        boost = 0.0
        if tid:
            try:
                boost = float(boosts.get(tid, 0.0))
            except Exception:
                boost = 0.0
        scored.append((base + boost, idx, it))

    scored.sort(key=lambda t: (-t[0], t[1]))
    out = [t[2] for t in scored]

    before_ids = [_safe_track_id(x, cfg.track_id_key) for x in items]
    after_ids = [_safe_track_id(x, cfg.track_id_key) for x in out]

    dbg["applied"] = bool(before_ids != after_ids)
    dbg["moved"] = sum(1 for i, (b, a) in enumerate(zip(before_ids, after_ids)) if b != a)

    return out, dbg


def apply_reorder_only_personalization(
    sections: Dict[str, list],
    *,
    track_boosts: Optional[Dict[str, float]] = None,
    section_names: Optional[List[str]] = None,
    cfg: Optional[PersonalizationConfig] = None,
) -> Tuple[Dict[str, list], Dict[str, Any]]:
    """
    Apply reorder-only personalization across multiple rails.

    Tests expect dbg contains:
      dbg["personalization_reorder_only_length_deltas"][section_name] == 0
    """
    cfg = cfg or PersonalizationConfig()
    boosts = track_boosts or {}

    out: Dict[str, list] = dict(sections) if isinstance(sections, dict) else sections

    # default rails if not provided
    if section_names is None:
        section_names = [s for s in cfg.default_section_names if isinstance(out.get(s), list)]

    length_deltas: Dict[str, int] = {}
    per_section_dbg: Dict[str, Any] = {}

    for name in section_names:
        items = out.get(name)
        if not isinstance(items, list):
            continue

        before_n = len(items)
        reordered, sdbg = reorder_section_reorder_only(items, track_boosts=boosts, cfg=cfg)
        out[name] = reordered
        after_n = len(reordered)

        length_deltas[name] = int(after_n - before_n)
        per_section_dbg[name] = sdbg

    dbg: Dict[str, Any] = {
        "personalization_reorder_only_length_deltas": length_deltas,
        "personalization_reorder_only_per_section": per_section_dbg,
        "personalization_reorder_only_sections": list(section_names),
    }
    return out, dbg


def _build_track_boosts_from_recent_events(
    recent_events: List[Dict[str, Any]],
    *,
    cfg: PersonalizationConfig,
) -> Dict[str, float]:
    """
    Convert recent feedback events into per-track boosts.
    Only uses positive signals (play/like). Negative signals are handled via suppression elsewhere.
    """
    boosts: Dict[str, float] = {}

    if not isinstance(recent_events, list) or not recent_events:
        return boosts

    for e in recent_events:
        if not isinstance(e, dict):
            continue
        tid = str(e.get("track_id") or "").strip()
        et = str(e.get("event_type") or "").strip().lower()
        if not tid or not et:
            continue

        if et == "like":
            boosts[tid] = float(boosts.get(tid, 0.0)) + float(cfg.like_boost)
        elif et == "play":
            boosts[tid] = float(boosts.get(tid, 0.0)) + float(cfg.play_boost)

    return boosts


def reorder_only_personalize_sections(
    *,
    sections: Dict[str, list],
    recent_events: List[Dict[str, Any]],
    user_id: str,
    debug: bool,
    config: Optional[PersonalizationConfig] = None,
) -> Tuple[Dict[str, list], Dict[str, Any]]:
    """
    API used by feed_routes.py for v1.5.6 Step 1.4
    Signature must accept:
      sections, recent_events, user_id, debug, config
    """
    cfg = config or PersonalizationConfig()

    track_boosts = _build_track_boosts_from_recent_events(recent_events, cfg=cfg)

    # Apply reorder-only to default rails only (NOT for_you)
    section_names = [s for s in cfg.default_section_names if isinstance(sections.get(s), list)]
    out, apply_dbg = apply_reorder_only_personalization(
        sections,
        track_boosts=track_boosts,
        section_names=section_names,
        cfg=cfg,
    )

    personalization_dbg: Dict[str, Any] = {
        "reorder_only": True,
        "user_id": user_id,
        "recent_events_considered": int(len(recent_events)) if isinstance(recent_events, list) else 0,
        "track_boosts_n": int(len(track_boosts)),
        "track_boosts_sample": dict(list(track_boosts.items())[:10]) if debug else None,
    }
    personalization_dbg.update(apply_dbg)

    return out, personalization_dbg