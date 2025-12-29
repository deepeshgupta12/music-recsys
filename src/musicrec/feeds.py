from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _safe_str(x: object) -> str:
    if x is None:
        return ""
    return str(x)


@dataclass(frozen=True)
class FeedSection:
    name: str
    items: List[Dict[str, object]]


class SegmentFeeds:
    """
    Build non-personalized segment feeds (V1.4):
      - /feed/home?country=IN
      - /feed/genre?country=IN&genre=Pop

    Sections:
      - top: by stream_count/popularity
      - rising: by momentum (log1p(stream_count)/(days_since_release+1))
      - new_releases: by recency (lower days_since_release)
      - instrumental: instrumental-heavy flag
      - explicit_safe: explicit = 0

    NOTE (Step 1.4.4):
      - We deduplicate artists WITHIN EACH SECTION (per-section artist cap = 1),
        but we allow an artist to appear in multiple sections.
    """

    def __init__(self, feature_table: pd.DataFrame):
        if "track_id" not in feature_table.columns:
            raise ValueError("feature_table must include track_id")

        self.ft = feature_table.copy()

    def _row_to_item(self, row: pd.Series, score: float) -> Dict[str, object]:
        return {
            "track_id": _safe_str(row.get("track_id")),
            "score": float(score),
            "track_name": _safe_str(row.get("track_name")),
            "artist_name": _safe_str(row.get("artist_name")),
            "country": _safe_str(row.get("country")),
            "genre": _safe_str(row.get("genre")),
            "popularity": int(row.get("popularity")) if row.get("popularity") is not None else 0,
            "stream_count": int(row.get("stream_count")) if row.get("stream_count") is not None else 0,
            "release_date": _safe_str(row.get("release_date")),
        }

    def _dedup_by_artist(self, items: List[Dict[str, object]], n: int) -> List[Dict[str, object]]:
        """Deduplicate items by artist_name within a single section.

        - Stable/order-preserving: keeps the first appearance.
        - Case-insensitive: normalizes artist_name to lower-case.
        - If artist_name is missing, uses track_id as a fallback unique key.
        """
        if n <= 0:
            return []

        seen: set[str] = set()
        out: List[Dict[str, object]] = []

        for it in items:
            artist = _safe_str(it.get("artist_name", "")).strip().lower()
            if not artist:
                artist = f'__missing__::{_safe_str(it.get("track_id", ""))}'

            if artist in seen:
                continue

            seen.add(artist)
            out.append(it)

            if len(out) >= n:
                break

        return out

    def home_feed(
        self,
        country: str,
        n: int = 5,
        genre: Optional[str] = None,
        explicit_ok: bool = True,
        debug: bool = False,
    ) -> Dict[str, object]:
        if n <= 0:
            n = 5

        df = self.ft
        df = df[df["country"].astype(str) == str(country)]

        if genre:
            df = df[df["genre"].astype(str) == str(genre)]

        base_rows = int(len(df))

        sections: Dict[str, List[Dict[str, object]]] = {}
        sections["top"] = self._top(df, n=n, explicit_ok=explicit_ok)
        sections["rising"] = self._rising(df, n=n, explicit_ok=explicit_ok)
        sections["new_releases"] = self._new_releases(df, n=n, explicit_ok=explicit_ok)
        sections["instrumental"] = self._instrumental(df, n=n, explicit_ok=explicit_ok)
        sections["explicit_safe"] = self._explicit_safe(df, n=n)

        payload: Dict[str, object] = {
            "ok": True,
            "country": country,
            "genre": genre,
            "n": n,
            "sections": sections,
        }

        if debug:
            payload["debug"] = {
                "country": country,
                "genre": genre,
                "explicit_ok": explicit_ok,
                "base_rows": base_rows,
                "sections_returned": {k: len(v) for k, v in sections.items()},
            }

        return payload

    def genre_feed(
        self,
        country: str,
        genre: str,
        n: int = 5,
        explicit_ok: bool = True,
        debug: bool = False,
    ) -> Dict[str, object]:
        return self.home_feed(country=country, n=n, genre=genre, explicit_ok=explicit_ok, debug=debug)

    def _top(self, df: pd.DataFrame, n: int, explicit_ok: bool) -> List[Dict[str, object]]:
        if len(df) == 0:
            return []

        d = df
        if not explicit_ok and "explicit" in d.columns:
            d = d[d["explicit"].astype(int) == 0]

        # oversample then per-section artist dedup
        d = d.sort_values(["stream_count", "popularity"], ascending=[False, False]).head(max(n * 5, n))

        out: List[Dict[str, object]] = []
        for _, row in d.iterrows():
            out.append(self._row_to_item(row, score=float(row.get("stream_count", 0))))
        return self._dedup_by_artist(out, n)

    def _rising(self, df: pd.DataFrame, n: int, explicit_ok: bool) -> List[Dict[str, object]]:
        if len(df) == 0:
            return []

        d = df
        if not explicit_ok and "explicit" in d.columns:
            d = d[d["explicit"].astype(int) == 0]

        if "momentum" not in d.columns:
            # fallback compute if not present
            sc = d["stream_count"].fillna(0).astype(float)
            dsr = d["days_since_release"].fillna(0).astype(float) if "days_since_release" in d.columns else 0.0
            d = d.copy()
            d["momentum"] = np.log1p(sc) / (dsr + 1.0)

        d = d.sort_values(["momentum", "popularity"], ascending=[False, False]).head(max(n * 5, n))

        out: List[Dict[str, object]] = []
        for _, row in d.iterrows():
            out.append(self._row_to_item(row, score=float(row.get("momentum", 0.0))))
        return self._dedup_by_artist(out, n)

    def _new_releases(self, df: pd.DataFrame, n: int, explicit_ok: bool) -> List[Dict[str, object]]:
        if len(df) == 0:
            return []

        d = df
        if not explicit_ok and "explicit" in d.columns:
            d = d[d["explicit"].astype(int) == 0]

        if "days_since_release" in d.columns:
            d = d.sort_values(["days_since_release", "popularity"], ascending=[True, False]).head(max(n * 5, n))
        else:
            # fallback to release_date if days_since_release missing
            d = d.sort_values(["release_date", "popularity"], ascending=[False, False]).head(max(n * 5, n))

        out: List[Dict[str, object]] = []
        for _, row in d.iterrows():
            dsr = float(row.get("days_since_release", 0.0) or 0.0)
            score = 1.0 / (dsr + 1.0)
            out.append(self._row_to_item(row, score=score))
        return self._dedup_by_artist(out, n)

    def _instrumental(self, df: pd.DataFrame, n: int, explicit_ok: bool) -> List[Dict[str, object]]:
        if len(df) == 0:
            return []

        d = df
        if not explicit_ok and "explicit" in d.columns:
            d = d[d["explicit"].astype(int) == 0]

        if "instrumental_heavy" in d.columns:
            d = d[d["instrumental_heavy"].astype(int) == 1]
        elif "instrumentalness" in d.columns:
            d = d[d["instrumentalness"].fillna(0).astype(float) >= 0.80]

        if len(d) == 0:
            return []

        d = d.sort_values(["popularity", "stream_count"], ascending=[False, False]).head(max(n * 5, n))

        out: List[Dict[str, object]] = []
        for _, row in d.iterrows():
            # constant score (we’re filtering by category here)
            out.append(self._row_to_item(row, score=0.8))
        return self._dedup_by_artist(out, n)

    def _explicit_safe(self, df: pd.DataFrame, n: int) -> List[Dict[str, object]]:
        d = df
        if "explicit" in d.columns:
            d = d[d["explicit"].astype(int) == 0]
        # reuse _top (already includes per-section artist dedup)
        return self._top(d, n=n, explicit_ok=True)