from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class FeedQuery:
    country: str
    genre: Optional[str] = None
    n: int = 25
    explicit_ok: bool = True
    debug: bool = False


class SegmentFeeds:
    """
    Feed builder that produces multiple sections (rails) for:
      - Home feed: country-only
      - Genre feed: country + genre

    V1.4 behavior:
      - Always returns the same section keys.
      - Returns (sections, debug_dict)
      - Per-section artist de-dup is applied inside each section selection.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # ----------------------------
    # Base filtering + derived cols
    # ----------------------------

    def _apply_base_filters(self, q: FeedQuery, *, include_genre: bool = True) -> pd.DataFrame:
        """Return a filtered copy of the catalog for this request.

        By default, applies country + (optional) genre + explicit filter.
        For section-level fallbacks (e.g., genre feed missing instrumentals),
        callers can set include_genre=False to get a country-only slice.
        """
        d = self.df.copy()

        # Country (required)
        if q.country:
            d = d[d["country"].astype(str) == q.country]

        # Genre (optional)
        if include_genre and q.genre:
            # case-insensitive exact match
            g = str(q.genre).strip().lower()
            d = d[d["genre"].astype(str).str.lower() == g]

        # Explicit filter
        if not q.explicit_ok and "explicit" in d.columns:
            d = d[d["explicit"].fillna(False) == False]  # noqa: E712

        return d

    def _add_derived(self, d: pd.DataFrame) -> pd.DataFrame:
        """Add derived columns used by section scoring."""
        if len(d) == 0:
            return d

        out = d.copy()

        # release_date -> datetime
        if "_release_dt" not in out.columns:
            if "release_date" in out.columns:
                out["_release_dt"] = pd.to_datetime(out["release_date"], errors="coerce")
            else:
                out["_release_dt"] = pd.NaT

        # days since release (used in rising/new releases)
        if "_days_since_release" not in out.columns:
            now = datetime.utcnow()
            rel = out["_release_dt"]
            days = (now - rel).dt.days
            out["_days_since_release"] = days.fillna(3650).clip(lower=0).astype(int)

        return out

    # ----------------------------
    # Row -> API item
    # ----------------------------

    def _row_to_item(self, r: pd.Series, score: float) -> Dict[str, object]:
        return {
            "track_id": r.get("track_id"),
            "score": float(score),
            "track_name": r.get("track_name"),
            "artist_name": r.get("artist_name"),
            "country": r.get("country"),
            "genre": r.get("genre"),
            "popularity": int(r.get("popularity")) if pd.notna(r.get("popularity")) else None,
            "stream_count": int(r.get("stream_count")) if pd.notna(r.get("stream_count")) else None,
            "release_date": str(r.get("release_date")),
        }

    # ----------------------------
    # Section builders (each returns list[items])
    # ----------------------------

    def _top(self, df: pd.DataFrame, n: int) -> List[Dict[str, object]]:
        if len(df) == 0:
            return []
        if "stream_count" not in df.columns:
            return []
        d = df.copy()
        d = d[d["stream_count"].notna()]
        if len(d) == 0:
            return []
        d = d.sort_values(["stream_count"], ascending=[False])

        # Per-section artist de-dup
        if "artist_name" in d.columns:
            d = d.assign(_artist_key=d["artist_name"].fillna(d["track_id"]).astype(str))
            d = d.drop_duplicates(subset=["_artist_key"], keep="first").drop(columns=["_artist_key"])

        d = d.head(n)
        out: List[Dict[str, object]] = []
        for _, r in d.iterrows():
            score = float(r.get("stream_count", 0) or 0)
            out.append(self._row_to_item(r, score))
        return out

    def _rising(self, df: pd.DataFrame, n: int) -> List[Dict[str, object]]:
        """Rising = high momentum relative to popularity.

        Score heuristic:
          - reward high momentum (stream_count / (popularity+1))
          - lightly reward recency (1/(days_since_release+1))
        """
        if len(df) == 0:
            return []
        if "stream_count" not in df.columns or "popularity" not in df.columns:
            return []
        d = df.copy()
        d = d[d["stream_count"].notna() & d["popularity"].notna()]
        if len(d) == 0:
            return []
        if "_days_since_release" not in d.columns:
            d = self._add_derived(d)

        d = d.assign(
            _mom=d["stream_count"].astype(float) / (d["popularity"].astype(float) + 1.0),
            _rec=1.0 / (d["_days_since_release"].astype(float) + 1.0),
        )
        d = d.assign(_score=d["_mom"] * 0.7 + d["_rec"] * 0.3)
        d = d.sort_values(["_score"], ascending=[False])

        # Per-section artist de-dup
        if "artist_name" in d.columns:
            d = d.assign(_artist_key=d["artist_name"].fillna(d["track_id"]).astype(str))
            d = d.drop_duplicates(subset=["_artist_key"], keep="first").drop(columns=["_artist_key"])

        d = d.head(n)
        out: List[Dict[str, object]] = []
        for _, r in d.iterrows():
            out.append(self._row_to_item(r, float(r.get("_score", 0.0) or 0.0)))
        return out

    def _new_releases(self, df: pd.DataFrame, n: int) -> List[Dict[str, object]]:
        if len(df) == 0:
            return []
        if "_release_dt" not in df.columns:
            return []
        d = df.copy()
        d = d[d["_release_dt"].notna()]
        if len(d) == 0:
            return []
        d = d.sort_values(["_release_dt"], ascending=[False])

        # Per-section artist de-dup
        if "artist_name" in d.columns:
            d = d.assign(_artist_key=d["artist_name"].fillna(d["track_id"]).astype(str))
            d = d.drop_duplicates(subset=["_artist_key"], keep="first").drop(columns=["_artist_key"])

        d = d.head(n)
        out: List[Dict[str, object]] = []
        for _, r in d.iterrows():
            dsr = int(r.get("_days_since_release", 0) or 0)
            score = float(1.0 / float(dsr + 1))
            out.append(self._row_to_item(r, score))
        return out

    def _instrumental(self, df: pd.DataFrame, n: int) -> List[Dict[str, object]]:
        if len(df) == 0:
            return []
        if "instrumentalness" not in df.columns:
            return []
        d = df.copy()
        d = d[d["instrumentalness"].notna()]
        if len(d) == 0:
            return []
        d = d[d["instrumentalness"].astype(float) >= 0.9]
        if len(d) == 0:
            return []
        d = d.sort_values(["instrumentalness", "stream_count"], ascending=[False, False])

        # Per-section artist de-dup
        if "artist_name" in d.columns:
            d = d.assign(_artist_key=d["artist_name"].fillna(d["track_id"]).astype(str))
            d = d.drop_duplicates(subset=["_artist_key"], keep="first").drop(columns=["_artist_key"])

        d = d.head(n)
        out: List[Dict[str, object]] = []
        for _, r in d.iterrows():
            score = float(r.get("instrumentalness", 0.0) or 0.0)
            out.append(self._row_to_item(r, score))
        return out

    def _explicit_safe(self, df: pd.DataFrame, n: int) -> List[Dict[str, object]]:
        if len(df) == 0:
            return []
        if "explicit" not in df.columns:
            return []
        d = df.copy()
        d = d[d["explicit"].fillna(False) == False]  # noqa: E712
        if len(d) == 0:
            return []
        d = d.sort_values(["stream_count"], ascending=[False])

        # Per-section artist de-dup
        if "artist_name" in d.columns:
            d = d.assign(_artist_key=d["artist_name"].fillna(d["track_id"]).astype(str))
            d = d.drop_duplicates(subset=["_artist_key"], keep="first").drop(columns=["_artist_key"])

        d = d.head(n)
        out: List[Dict[str, object]] = []
        for _, r in d.iterrows():
            score = float(r.get("stream_count", 0) or 0)
            out.append(self._row_to_item(r, score))
        return out

    # ----------------------------
    # Public API
    # ----------------------------

    def home_feed(self, q: FeedQuery) -> Tuple[Dict[str, List[Dict[str, object]]], Dict[str, object]]:
        """Build feed sections.

        If q.genre is provided (genre feed), selectors run on the genre-sliced
        dataframe. For any section that comes back empty, we apply a targeted
        fallback by dropping the genre constraint (country-only) for that
        section, so the UI doesn't render empty rails.
        """
        # Country-only slice (for fallbacks)
        base_country = self._apply_base_filters(q, include_genre=False)
        base_country = self._add_derived(base_country)

        # Primary slice (country + optional genre)
        base = base_country
        if q.genre:
            g = str(q.genre).strip().lower()
            base = base[base["genre"].astype(str).str.lower() == g]

        fallback_used: Dict[str, str] = {}

        def _build_with_fallback(name: str, fn, primary_df: pd.DataFrame) -> List[Dict[str, object]]:
            items = fn(primary_df, q.n)
            if q.genre and len(items) == 0:
                fb = fn(base_country, q.n)
                if len(fb) > 0:
                    fallback_used[name] = "dropped_genre"
                    return fb
            return items

        sections: Dict[str, List[Dict[str, object]]] = {
            "top": _build_with_fallback("top", self._top, base),
            "rising": _build_with_fallback("rising", self._rising, base),
            "new_releases": _build_with_fallback("new_releases", self._new_releases, base),
            "instrumental": _build_with_fallback("instrumental", self._instrumental, base),
            "explicit_safe": _build_with_fallback("explicit_safe", self._explicit_safe, base),
        }

        dbg: Dict[str, object] = {}
        if q.debug:
            dbg = {
                "country": q.country,
                "genre": q.genre,
                "explicit_ok": q.explicit_ok,
                "base_rows": int(len(base)),
                "sections_returned": {k: int(len(v)) for k, v in sections.items()},
                "fallback_used": fallback_used,  # always present; may be empty
            }
        return sections, dbg

    def genre_feed(self, q: FeedQuery) -> Tuple[Dict[str, List[Dict[str, object]]], Dict[str, object]]:
        # For now, genre feed uses the same engine; behavior differs via q.genre
        return self.home_feed(q)