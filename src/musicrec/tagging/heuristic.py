from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from musicrec.tagging.schema import TrackTagBundle


def _bucket(v: Optional[float], lo: float, hi: float) -> str:
    if v is None:
        return "medium"
    if v <= lo:
        return "low"
    if v >= hi:
        return "high"
    return "medium"


def _tempo_label(tempo: Optional[float]) -> str:
    if tempo is None:
        return "mid"
    if tempo < 90:
        return "slow"
    if tempo > 130:
        return "fast"
    return "mid"


def _instrumental_label(instrumentalness: Optional[float]) -> str:
    if instrumentalness is None:
        return "vocal"
    if instrumentalness >= 0.80:
        return "instrumental"
    if instrumentalness >= 0.40:
        return "mostly_instrumental"
    return "vocal"


def _dedup_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in xs:
        x2 = (x or "").strip()
        if not x2 or x2 in seen:
            continue
        seen.add(x2)
        out.append(x2)
    return out


def heuristic_tags_from_features(
    track_id: str,
    features: Dict[str, Any],
    meta: Dict[str, Any],
) -> TrackTagBundle:
    """
    Deterministic tag generator from numeric/audio + metadata.
    This is the default for tests and offline runs.

    Expected feature keys (best-effort):
      - energy, danceability, tempo, instrumentalness, loudness
    Metadata (best-effort):
      - genre, country, explicit, artist_name, track_name
    """
    tid = (track_id or "").strip()
    if not tid:
        raise ValueError("track_id required")

    energy = _safe_float(features.get("energy"))
    dance = _safe_float(features.get("danceability"))
    tempo = _safe_float(features.get("tempo"))
    instr = _safe_float(features.get("instrumentalness"))
    loud = _safe_float(features.get("loudness"))

    energy_label = _bucket(energy, lo=0.35, hi=0.70)
    dance_label = _bucket(dance, lo=0.35, hi=0.70)
    tempo_label = _tempo_label(tempo)
    instrumental_label = _instrumental_label(instr)

    moods: List[str] = []
    scenes: List[str] = []
    activities: List[str] = []
    descriptors: List[str] = []
    subgenre_hints: List[str] = []

    # Scene/activity heuristics (simple but useful)
    if energy_label == "high" and dance_label in ("medium", "high"):
        moods += ["energetic"]
        scenes += ["party"]
        activities += ["workout", "dance"]
        descriptors += ["upbeat", "driving"]
    if energy_label == "low" and tempo_label in ("slow", "mid"):
        moods += ["chill"]
        scenes += ["late-night", "study"]
        activities += ["focus", "relax"]
        descriptors += ["soft", "smooth"]
    if tempo_label == "fast" and energy_label == "high":
        scenes += ["gym"]
        activities += ["run", "workout"]
        descriptors += ["fast-paced"]

    # Instrumentalness impacts focus labels
    if instrumental_label in ("instrumental", "mostly_instrumental"):
        scenes += ["focus"]
        activities += ["work", "study"]
        descriptors += ["instrumental"]
        if instrumental_label == "instrumental":
            moods += ["calm"]

    # Loudness hint (if present)
    if loud is not None and loud > -6.0:
        descriptors += ["punchy"]
    if loud is not None and loud < -14.0:
        descriptors += ["airy"]

    # Metadata light hints
    genre = _safe_str(meta.get("genre"))
    if genre:
        g = genre.lower()
        if "hip hop" in g or "rap" in g:
            descriptors += ["rhythmic"]
            scenes += ["street"]
        if "edm" in g or "electronic" in g:
            scenes += ["club"]
            descriptors += ["electronic"]
        if "classical" in g:
            scenes += ["focus"]
            moods += ["calm"]
            descriptors += ["classical"]
        if "rock" in g:
            descriptors += ["guitar-driven"]
        if "jazz" in g:
            moods += ["smooth"]
            descriptors += ["jazzy"]

        # Subgenre hints: keep extremely light and not over-claiming
        if "lofi" in g or "lo-fi" in g:
            subgenre_hints += ["lofi"]
            scenes += ["study"]
            moods += ["chill"]

    moods = _dedup_keep_order(moods)
    scenes = _dedup_keep_order(scenes)
    activities = _dedup_keep_order(activities)
    descriptors = _dedup_keep_order(descriptors)
    subgenre_hints = _dedup_keep_order(subgenre_hints)

    # Ensure non-empty minimal defaults
    if not moods:
        moods = ["balanced"]
    if not scenes:
        scenes = ["everyday"]
    if not activities:
        activities = ["listen"]

    return TrackTagBundle(
        track_id=tid,
        moods=moods,
        scenes=scenes,
        activities=activities,
        energy_label=energy_label,
        danceability_label=dance_label,
        instrumental_label=instrumental_label,
        tempo_label=tempo_label,
        descriptors=descriptors,
        subgenre_hints=subgenre_hints,
        country=_safe_str(meta.get("country")),
        genre=genre,
        explicit=_safe_bool(meta.get("explicit")),
        artist_name=_safe_str(meta.get("artist_name")),
        track_name=_safe_str(meta.get("track_name")),
        source="heuristic",
        model_id="v1.5.1-heuristic",
        created_at_ts=None,
    )


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _safe_str(x: Any) -> Optional[str]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        return s or None
    except Exception:
        return None


def _safe_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("true", "1", "yes", "y"):
        return True
    if s in ("false", "0", "no", "n"):
        return False
    return None