from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


class FeedQuery(BaseModel):
    """
    Query model imported by src/musicrec/api/feed_routes.py.

    feed_routes expects:
      q = FeedQuery(country=..., genre=..., n=..., explicit_ok=..., debug=...)
      sections, dbg = feeds.home_feed(q)
      sections, dbg = feeds.genre_feed(q)
    """
    country: str = Field(..., min_length=1)
    genre: Optional[str] = None
    n: int = Field(default=25, ge=1, le=200)
    explicit_ok: bool = True
    debug: bool = False


class SegmentFeeds:
    """
    Segment feeds engine (V1.4).

    IMPORTANT: Keep API contract compatible with feed_routes.py:
      - home_feed(q: FeedQuery) -> (sections: dict, dbg: dict)
      - genre_feed(q: FeedQuery) -> (sections: dict, dbg: dict)

    Step 1.4.4: Per-section artist de-dup
      - De-dup happens WITHIN each section only.
      - Same artist may appear across different sections.
    """

    def __init__(self, feature_table: pd.DataFrame):
        if "track_id" not in feature_table.columns:
            raise ValueError("feature_table must include 'track_id'")
        self.ft = feature_table.copy()

    # ----------------------------
    # Public methods used by routes
    # ----------------------------
    def home_feed(self, q: FeedQuery) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
        return self._build_sections(
            country=q.country,
            genre=None,
            n=int(q.n),
            explicit_ok=bool(q.explicit_ok),
            debug=bool(q.debug),
        )

    def genre_feed(self, q: FeedQuery) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
        g = (q.genre or "").strip()
        return self._build_sections(
            country=q.country,
            genre=g,
            n=int(q.n),
            explicit_ok=bool(q.explicit_ok),
            debug=bool(q.debug),
        )

    # ----------------------------
    # Core builder (returns tuple for feed_routes)
    # ----------------------------
    def _build_sections(
        self,
        country: str,
        genre: Optional[str],
        n: int,
        explicit_ok: bool,
        debug: bool,
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
        df = self.ft
        df = df[df["country"].astype(str) == str(country)]
        if genre:
            df = df[df["genre"].astype(str) == str(genre)]

        base_rows = int(len(df))

        sections: Dict[str, List[Dict[str, Any]]] = {
            "top": self._top(df, n=n, explicit_ok=explicit_ok),
            "rising": self._rising(df, n=n, explicit_ok=explicit_ok),
            "new_releases": self._new_releases(df, n=n, explicit_ok=explicit_ok),
            "instrumental": self._instrumental(df, n=n, explicit_ok=explicit_ok),
            "explicit_safe": self._explicit_safe(df, n=n),
        }

        dbg: Dict[str, Any] = {}
        if debug:
            dbg = {
                "country": country,
                "genre": genre,
                "explicit_ok": bool(explicit_ok),
                "base_rows": base_rows,
                "sections_returned": {k: len(v) for k, v in sections.items()},
            }

        return sections, dbg

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
        Step 1.4.4: Per-section artist de-dup (order-preserving).
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
        # oversample so dedup still yields ~n results
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