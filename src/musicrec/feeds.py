from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


class FeedQuery(BaseModel):
    """
    Query model used by feed_routes.py.
    Keep this exported symbol stable to avoid import errors.
    """
    country: str = Field(..., min_length=1)
    genre: Optional[str] = None
    n: int = Field(default=5, ge=1, le=100)
    explicit_ok: bool = True
    debug: bool = False


class SegmentFeeds:
    """
    Segment feed builder (V1.4):
      - home feed: country-based
      - genre feed: country + genre-based

    Sections:
      - top: stream_count/popularity
      - rising: momentum (log1p(stream_count)/(days_since_release+1))
      - new_releases: recency (lower days_since_release)
      - instrumental: instrumental-heavy filter
      - explicit_safe: explicit == 0

    Step 1.4.4 (sub-sub-step 1):
      - Per-section artist de-dup: within each section only.
      - Same artist may appear across different sections.
    """

    def __init__(self, feature_table: pd.DataFrame):
        if "track_id" not in feature_table.columns:
            raise ValueError("feature_table must include 'track_id'")
        self.ft = feature_table.copy()

    # ----------------------------
    # Public helpers for routes
    # ----------------------------
    def build(self, q: FeedQuery) -> Dict[str, Any]:
        if q.genre:
            return self.genre_feed(country=q.country, genre=q.genre, n=q.n, explicit_ok=q.explicit_ok, debug=q.debug)
        return self.home_feed(country=q.country, n=q.n, explicit_ok=q.explicit_ok, debug=q.debug)

    def home_feed(
        self,
        country: str,
        n: int = 5,
        explicit_ok: bool = True,
        debug: bool = False,
    ) -> Dict[str, Any]:
        return self._build_feed(country=country, genre=None, n=n, explicit_ok=explicit_ok, debug=debug)

    def genre_feed(
        self,
        country: str,
        genre: str,
        n: int = 5,
        explicit_ok: bool = True,
        debug: bool = False,
    ) -> Dict[str, Any]:
        return self._build_feed(country=country, genre=genre, n=n, explicit_ok=explicit_ok, debug=debug)

    # ----------------------------
    # Core build
    # ----------------------------
    def _build_feed(
        self,
        country: str,
        genre: Optional[str],
        n: int,
        explicit_ok: bool,
        debug: bool,
    ) -> Dict[str, Any]:
        # base filter
        df = self.ft
        df = df[df["country"].astype(str) == str(country)]
        if genre:
            df = df[df["genre"].astype(str) == str(genre)]

        base_rows = int(len(df))

        sections: Dict[str, List[Dict[str, Any]]] = {}
        sections["top"] = self._top(df, n=n, explicit_ok=explicit_ok)
        sections["rising"] = self._rising(df, n=n, explicit_ok=explicit_ok)
        sections["new_releases"] = self._new_releases(df, n=n, explicit_ok=explicit_ok)
        sections["instrumental"] = self._instrumental(df, n=n, explicit_ok=explicit_ok)
        sections["explicit_safe"] = self._explicit_safe(df, n=n)

        payload: Dict[str, Any] = {
            "ok": True,
            "country": country,
            "genre": genre,
            "n": int(n),
            "sections": sections,
        }

        if debug:
            payload["debug"] = {
                "country": country,
                "genre": genre,
                "explicit_ok": bool(explicit_ok),
                "base_rows": base_rows,
                "sections_returned": {k: len(v) for k, v in sections.items()},
            }

        return payload

    # ----------------------------
    # Section utilities
    # ----------------------------
    def _row_to_item(self, row: pd.Series, score: float) -> Dict[str, Any]:
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

    def _dedup_by_artist(self, items: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
        """
        Per-section artist de-dup (Step 1.4.4):
          - stable, order-preserving
          - case-insensitive artist_name normalization
          - if artist missing, fallback to unique by track_id
        """
        if n <= 0:
            return []

        seen: Set[str] = set()
        out: List[Dict[str, Any]] = []

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

    def _apply_explicit_filter(self, df: pd.DataFrame, explicit_ok: bool) -> pd.DataFrame:
        if explicit_ok:
            return df
        if "explicit" not in df.columns:
            return df
        return df[df["explicit"].astype(int) == 0]

    def _oversample_head(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        # oversample so dedup doesn't shrink the section too much
        return df.head(max(n * 5, n))

    # ----------------------------
    # Sections
    # ----------------------------
    def _top(self, df: pd.DataFrame, n: int, explicit_ok: bool) -> List[Dict[str, Any]]:
        if len(df) == 0:
            return []

        d = self._apply_explicit_filter(df, explicit_ok)
        if len(d) == 0:
            return []

        d = d.sort_values(["stream_count", "popularity"], ascending=[False, False])
        d = self._oversample_head(d, n)

        items: List[Dict[str, Any]] = []
        for _, row in d.iterrows():
            items.append(self._row_to_item(row, score=float(row.get("stream_count", 0) or 0)))
        return self._dedup_by_artist(items, n)

    def _rising(self, df: pd.DataFrame, n: int, explicit_ok: bool) -> List[Dict[str, Any]]:
        if len(df) == 0:
            return []

        d = self._apply_explicit_filter(df, explicit_ok)
        if len(d) == 0:
            return []

        if "momentum" not in d.columns:
            sc = d["stream_count"].fillna(0).astype(float)
            if "days_since_release" in d.columns:
                dsr = d["days_since_release"].fillna(0).astype(float)
            else:
                dsr = 0.0
            d = d.copy()
            d["momentum"] = np.log1p(sc) / (dsr + 1.0)

        d = d.sort_values(["momentum", "popularity"], ascending=[False, False])
        d = self._oversample_head(d, n)

        items: List[Dict[str, Any]] = []
        for _, row in d.iterrows():
            items.append(self._row_to_item(row, score=float(row.get("momentum", 0.0) or 0.0)))
        return self._dedup_by_artist(items, n)

    def _new_releases(self, df: pd.DataFrame, n: int, explicit_ok: bool) -> List[Dict[str, Any]]:
        if len(df) == 0:
            return []

        d = self._apply_explicit_filter(df, explicit_ok)
        if len(d) == 0:
            return []

        if "days_since_release" in d.columns:
            d = d.sort_values(["days_since_release", "popularity"], ascending=[True, False])
        else:
            d = d.sort_values(["release_date", "popularity"], ascending=[False, False])

        d = self._oversample_head(d, n)

        items: List[Dict[str, Any]] = []
        for _, row in d.iterrows():
            dsr = float(row.get("days_since_release", 0.0) or 0.0)
            score = 1.0 / (dsr + 1.0)
            items.append(self._row_to_item(row, score=score))
        return self._dedup_by_artist(items, n)

    def _instrumental(self, df: pd.DataFrame, n: int, explicit_ok: bool) -> List[Dict[str, Any]]:
        if len(df) == 0:
            return []

        d = self._apply_explicit_filter(df, explicit_ok)
        if len(d) == 0:
            return []

        if "instrumental_heavy" in d.columns:
            d = d[d["instrumental_heavy"].astype(int) == 1]
        elif "instrumentalness" in d.columns:
            d = d[d["instrumentalness"].fillna(0).astype(float) >= 0.80]

        if len(d) == 0:
            return []

        d = d.sort_values(["popularity", "stream_count"], ascending=[False, False])
        d = self._oversample_head(d, n)

        items: List[Dict[str, Any]] = []
        for _, row in d.iterrows():
            items.append(self._row_to_item(row, score=0.8))
        return self._dedup_by_artist(items, n)

    def _explicit_safe(self, df: pd.DataFrame, n: int) -> List[Dict[str, Any]]:
        d = df
        if "explicit" in d.columns:
            d = d[d["explicit"].astype(int) == 0]
        return self._top(d, n=n, explicit_ok=True)