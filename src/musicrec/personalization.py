from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

# Positive-only signals for reorder-only personalization
_EVENT_WEIGHT_POSITIVE: Dict[str, float] = {
    "play": 1.0,
    "like": 3.0,
    "add_to_playlist": 4.0,
}

# Negative signals are NOT handled here (suppression is handled elsewhere)
_NEGATIVE_EVENT_TYPES = {"dislike", "skip"}


@dataclass(frozen=True)
class PersonalizationConfig:
    half_life_days: float = 7.0
    max_events: int = 500

    # How strongly we let boosts influence order.
    # We do NOT change scores in the payload; we only reorder by (boost, score, original_index).
    boost_rank_weight: float = 1.0

    # When true, only consider positive events for boosts
    positive_only: bool = True


def _parse_ts_iso(ts: str) -> datetime:
    """
    Parse ISO timestamp and ensure tz-aware UTC.
    Accepts strings like:
      - 2025-12-30T08:48:21+00:00
      - 2025-12-30T08:48:21.123456+00:00
      - 2025-12-30T08:48:21Z  (normalize 'Z' -> '+00:00')
    """
    s = (ts or "").strip()
    if not s:
        return datetime.now(timezone.utc)
    s = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def build_track_boosts_from_events(
    events: Iterable[Any],
    *,
    cfg: Optional[PersonalizationConfig] = None,
    now: Optional[datetime] = None,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Convert recent feedback events into per-track boost weights with recency decay.

    Expected event shape: either dict-like with keys:
      - track_id
      - event_type
      - ts (ISO string)
    or an object with attrs:
      - track_id
      - event_type
      - ts

    Returns: (track_boosts, debug)
    """
    cfg = cfg or PersonalizationConfig()
    now = now or datetime.now(timezone.utc)

    boosts: Dict[str, float] = {}
    considered = 0
    used = 0
    skipped_negative = 0
    skipped_invalid = 0

    for e in list(events)[: cfg.max_events]:
        considered += 1

        # tolerate dicts or objects
        if isinstance(e, dict):
            tid = str((e.get("track_id") or "")).strip()
            et = str((e.get("event_type") or "")).strip().lower()
            ts = str((e.get("ts") or "")).strip()
        else:
            tid = str(getattr(e, "track_id", "") or "").strip()
            et = str(getattr(e, "event_type", "") or "").strip().lower()
            ts = str(getattr(e, "ts", "") or "").strip()

        if not tid or not et:
            skipped_invalid += 1
            continue

        if et in _NEGATIVE_EVENT_TYPES and cfg.positive_only:
            skipped_negative += 1
            continue

        w0 = _EVENT_WEIGHT_POSITIVE.get(et, 0.0)
        if w0 <= 0.0:
            skipped_invalid += 1
            continue

        dt = _parse_ts_iso(ts)
        age_days = max(0.0, (now - dt).total_seconds() / 86400.0)
        decay = 0.5 ** (age_days / float(cfg.half_life_days)) if cfg.half_life_days > 0 else 1.0
        w = float(w0) * float(decay)

        boosts[tid] = boosts.get(tid, 0.0) + w
        used += 1

    dbg = {
        "events_considered": int(considered),
        "events_used_for_boosts": int(used),
        "events_skipped_negative": int(skipped_negative),
        "events_skipped_invalid": int(skipped_invalid),
        "boosted_track_ids_n": int(len(boosts)),
    }
    return boosts, dbg


def _minmax_dict(d: Mapping[str, float]) -> Dict[str, float]:
    if not d:
        return {}
    vals = np.array(list(d.values()), dtype="float64")
    mn = float(np.min(vals))
    mx = float(np.max(vals))
    if abs(mx - mn) < 1e-12:
        return {k: 0.5 for k in d.keys()}
    return {k: float((v - mn) / (mx - mn)) for k, v in d.items()}


def reorder_section_reorder_only(
    items: List[Dict[str, Any]],
    *,
    track_boosts: Mapping[str, float],
    cfg: Optional[PersonalizationConfig] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Reorder-only:
      - keeps same items
      - returns a new list with boosted items earlier (stable)

    Sort key:
      - boost_norm desc
      - item["score"] desc (fallback 0.0)
      - original_index asc
    """
    cfg = cfg or PersonalizationConfig()
    if not items:
        return items, {"boosted_items": 0, "moved": 0}

    boost_norm = _minmax_dict(track_boosts)

    orig_ids = [str(it.get("track_id") or "") for it in items]
    idx_by_id = {tid: i for i, tid in enumerate(orig_ids)}

    def score_of(it: Dict[str, Any]) -> float:
        s = it.get("score")
        try:
            return float(s)
        except Exception:
            return 0.0

    decorated = []
    boosted_items = 0
    for i, it in enumerate(items):
        tid = str(it.get("track_id") or "").strip()
        b = float(boost_norm.get(tid, 0.0)) * float(cfg.boost_rank_weight)
        if b > 0.0:
            boosted_items += 1
        decorated.append((it, b, score_of(it), i))

    decorated.sort(key=lambda t: (-t[1], -t[2], t[3]))
    out = [t[0] for t in decorated]

    # moved count (how many positions changed)
    moved = 0
    for new_i, it in enumerate(out):
        tid = str(it.get("track_id") or "")
        old_i = idx_by_id.get(tid, new_i)
        if old_i != new_i:
            moved += 1

    dbg = {"boosted_items": int(boosted_items), "moved": int(moved)}
    return out, dbg


def apply_reorder_only_personalization(
    sections: Dict[str, List[Dict[str, Any]]],
    *,
    track_boosts: Mapping[str, float],
    section_names: Optional[List[str]] = None,
    cfg: Optional[PersonalizationConfig] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """
    Apply reorder-only personalization to selected sections.
    Keeps section lengths exactly the same.

    Returns (new_sections, debug)
    """
    cfg = cfg or PersonalizationConfig()
    section_names = section_names or list(sections.keys())

    out: Dict[str, List[Dict[str, Any]]] = {}
    per_section_dbg: Dict[str, Any] = {}
    length_deltas: Dict[str, int] = {}

    for name, items in sections.items():
        if name not in section_names:
            out[name] = items
            continue

        before_n = len(items)
        new_items, sdbg = reorder_section_reorder_only(items, track_boosts=track_boosts, cfg=cfg)
        after_n = len(new_items)

        out[name] = new_items
        per_section_dbg[name] = sdbg
        length_deltas[name] = int(after_n - before_n)

    dbg = {
        "reorder_sections": section_names,
        "personalization_reorder_only_length_deltas": length_deltas,
        "reorder_debug_by_section": per_section_dbg,
    }
    return out, dbg

reorder_only_personalize_sections = PersonalizationConfig