from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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
    Feed builder for:
      - Home feed: country-only
      - Genre feed: country + genre

    V1.4 behavior:
      - Always returns the same section keys
      - Returns (sections, debug_dict)
      - Per-section artist de-dup
      - V1.4.5: targeted fallback so sections don't come back empty where possible
      - V1.4.6: cross-section track de-dup so the same track doesn't show in multiple rails
    """

    SECTION_ORDER = ["top", "rising", "new_releases", "instrumental", "explicit_safe"]

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # ----------------------------
    # Helpers
    # ----------------------------

    @staticmethod
    def _norm_artist_name(name: Any) -> str:
        s = "" if name is None else str(name)
        return s.strip().lower()

    @classmethod
    def _artist_key_for_row(cls, artist_name: Any, track_id: Any) -> str:
        a = cls._norm_artist_name(artist_name)
        if a:
            return a
        return "" if track_id is None else str(track_id)

    def _dedup_by_artist(self, items: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
        """
        Stable, case-insensitive, space-trim de-dup by artist_name.
        If artist_name missing/blank -> treat as unique by track_id.
        """
        out: List[Dict[str, Any]] = []
        seen: set[str] = set()

        for it in items:
            key = self._artist_key_for_row(it.get("artist_name"), it.get("track_id"))
            if key in seen:
                continue
            seen.add(key)
            out.append(it)
            if len(out) >= n:
                break

        return out

    def _dedup_df_by_artist(self, d: pd.DataFrame) -> pd.DataFrame:
        if len(d) == 0:
            return d
        if "artist_name" not in d.columns:
            return d

        artist_norm = d["artist_name"].fillna("").astype(str).str.strip().str.lower()
        track_id = d["track_id"].fillna("").astype(str)
        artist_key = artist_norm.where(artist_norm != "", track_id)

        dd = d.copy()
        dd["_artist_key"] = artist_key
        dd = dd.drop_duplicates(subset=["_artist_key"], keep="first").drop(columns=["_artist_key"])
        return dd

    @staticmethod
    def _norm_track_key(item: Dict[str, Any]) -> str:
        """
        Canonical cross-rail key. Prefer track_id; if missing, use a composite fallback.
        """
        tid = item.get("track_id")
        if tid is not None and str(tid).strip():
            return str(tid).strip()
        # fallback key (rare)
        an = (item.get("artist_name") or "")
        tn = (item.get("track_name") or "")
        return f"__fallback__:{str(an).strip().lower()}:{str(tn).strip().lower()}"

    def _cross_section_dedup(
        self,
        sections: Dict[str, List[Dict[str, Any]]],
        *,
        n_final: int,
        order: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, int]]:
        """
        Remove duplicate tracks across sections in a stable way.
        Earlier sections keep items; later sections lose duplicates.
        Returns (deduped_sections, removed_counts_per_section).
        """
        order = order or self.SECTION_ORDER
        seen: set[str] = set()
        removed: Dict[str, int] = {}

        out: Dict[str, List[Dict[str, Any]]] = {}

        # process in priority order
        for name in order:
            items = sections.get(name, [])
            kept: List[Dict[str, Any]] = []
            rem = 0
            for it in items:
                k = self._norm_track_key(it)
                if k in seen:
                    rem += 1
                    continue
                seen.add(k)
                kept.append(it)
                if len(kept) >= n_final:
                    break
            out[name] = kept
            removed[name] = rem

        # carry over any other keys not in order (just in case)
        for k, v in sections.items():
            if k not in out:
                out[k] = v[:n_final]
                removed[k] = 0

        return out, removed

    # ----------------------------
    # Filtering + derived columns
    # ----------------------------

    def _apply_base_filters(self, q: FeedQuery, *, include_genre: bool = True) -> pd.DataFrame:
        d = self.df.copy()

        # Country (required)
        if q.country:
            d = d[d["country"].astype(str) == q.country]

        # Genre (optional)
        if include_genre and q.genre:
            g = str(q.genre).strip().lower()
            d = d[d["genre"].astype(str).str.lower() == g]

        # Explicit filter
        if not q.explicit_ok and "explicit" in d.columns:
            d = d[d["explicit"].fillna(False) == False]  # noqa: E712

        return d

    def _add_derived(self, d: pd.DataFrame) -> pd.DataFrame:
        if len(d) == 0:
            return d

        out = d.copy()

        if "_release_dt" not in out.columns:
            if "release_date" in out.columns:
                out["_release_dt"] = pd.to_datetime(out["release_date"], errors="coerce")
            else:
                out["_release_dt"] = pd.NaT

        if "_days_since_release" not in out.columns:
            now = datetime.utcnow()
            rel = out["_release_dt"]
            days = (now - rel).dt.days
            out["_days_since_release"] = days.fillna(3650).clip(lower=0).astype(int)

        return out

    # ----------------------------
    # Row -> API item
    # ----------------------------

    def _row_to_item(self, r: pd.Series, score: float) -> Dict[str, Any]:
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
    # Section builders (return up to n items)
    # ----------------------------

    def _top(self, df: pd.DataFrame, n: int) -> List[Dict[str, Any]]:
        if len(df) == 0 or "stream_count" not in df.columns:
            return []
        d = df.copy()
        d = d[d["stream_count"].notna()]
        if len(d) == 0:
            return []
        d = d.sort_values(["stream_count"], ascending=[False])
        d = self._dedup_df_by_artist(d)
        d = d.head(n)

        items: List[Dict[str, Any]] = []
        for _, r in d.iterrows():
            items.append(self._row_to_item(r, float(r.get("stream_count", 0) or 0)))
        return items

    def _rising(self, df: pd.DataFrame, n: int) -> List[Dict[str, Any]]:
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
        d = self._dedup_df_by_artist(d)
        d = d.head(n)

        items: List[Dict[str, Any]] = []
        for _, r in d.iterrows():
            items.append(self._row_to_item(r, float(r.get("_score", 0.0) or 0.0)))
        return items

    def _new_releases(self, df: pd.DataFrame, n: int) -> List[Dict[str, Any]]:
        if len(df) == 0:
            return []
        if "_release_dt" not in df.columns:
            return []

        d = df.copy()
        d = d[d["_release_dt"].notna()]
        if len(d) == 0:
            return []

        d = d.sort_values(["_release_dt"], ascending=[False])
        d = self._dedup_df_by_artist(d)
        d = d.head(n)

        items: List[Dict[str, Any]] = []
        for _, r in d.iterrows():
            dsr = int(r.get("_days_since_release", 0) or 0)
            score = float(1.0 / float(dsr + 1))
            items.append(self._row_to_item(r, score))
        return items

    def _instrumental(self, df: pd.DataFrame, n: int) -> List[Dict[str, Any]]:
        """
        Instrumental rail strategy:
          - Prefer true instrumental tracks: instrumentalness >= 0.9
          - If none exist in the slice, return top-N most-instrumental tracks
          - If instrumentalness column missing or all null, fall back to top by stream_count
        """
        if len(df) == 0:
            return []

        if "instrumentalness" not in df.columns:
            return self._top(df, n)

        d = df.copy()
        d = d[d["instrumentalness"].notna()]
        if len(d) == 0:
            return self._top(df, n)

        d = d.assign(_inst=d["instrumentalness"].astype(float))
        d = d.sort_values(["_inst", "stream_count"], ascending=[False, False])

        strict = d[d["_inst"] >= 0.9]
        picked = strict if len(strict) > 0 else d

        picked = self._dedup_df_by_artist(picked)
        picked = picked.head(n)

        items: List[Dict[str, Any]] = []
        for _, r in picked.iterrows():
            items.append(self._row_to_item(r, float(r.get("_inst", 0.0) or 0.0)))
        return items

    def _explicit_safe(self, df: pd.DataFrame, n: int) -> List[Dict[str, Any]]:
        if len(df) == 0:
            return []
        if "explicit" not in df.columns:
            return []

        d = df.copy()
        d = d[d["explicit"].fillna(False) == False]  # noqa: E712
        if len(d) == 0:
            return []
        d = d.sort_values(["stream_count"], ascending=[False])
        d = self._dedup_df_by_artist(d)
        d = d.head(n)

        items: List[Dict[str, Any]] = []
        for _, r in d.iterrows():
            items.append(self._row_to_item(r, float(r.get("stream_count", 0) or 0)))
        return items

    # ----------------------------
    # Public API
    # ----------------------------

    def home_feed(self, q: FeedQuery) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
        """
        Build feed sections.

        If q.genre exists (genre feed), primary slice is country+genre.
        For any section that comes back empty, do a targeted fallback by dropping genre
        for that section only.

        V1.4.6: After sections are built, we do cross-section dedup by track_id
        so the same track doesn't appear in multiple rails.
        """
        n_final = int(q.n)
        n_search = min(200, max(n_final, n_final * 5))

        # Country-only slice (fallback universe)
        base_country = self._apply_base_filters(q, include_genre=False)
        base_country = self._add_derived(base_country)

        # Primary slice
        base = base_country
        if q.genre:
            g = str(q.genre).strip().lower()
            base = base[base["genre"].astype(str).str.lower() == g]

        fallback_used: Dict[str, str] = {}

        def _build_with_fallback(name: str, fn, primary_df: pd.DataFrame) -> List[Dict[str, Any]]:
            items = fn(primary_df, n_search)
            if q.genre and len(items) == 0:
                fb = fn(base_country, n_search)
                if len(fb) > 0:
                    fallback_used[name] = "dropped_genre"
                    return fb
            return items

        sections_raw: Dict[str, List[Dict[str, Any]]] = {
            "top": _build_with_fallback("top", self._top, base),
            "rising": _build_with_fallback("rising", self._rising, base),
            "new_releases": _build_with_fallback("new_releases", self._new_releases, base),
            "instrumental": _build_with_fallback("instrumental", self._instrumental, base),
            "explicit_safe": _build_with_fallback("explicit_safe", self._explicit_safe, base),
        }

        sections, removed = self._cross_section_dedup(sections_raw, n_final=n_final, order=self.SECTION_ORDER)

        dbg: Dict[str, Any] = {}
        if q.debug:
            dbg = {
                "country": q.country,
                "genre": q.genre,
                "explicit_ok": q.explicit_ok,
                "base_rows": int(len(base)),
                "sections_returned": {k: int(len(v)) for k, v in sections.items()},
                "fallback_used": fallback_used,
                "cross_section_dedup_removed": removed,
                "n_search": n_search,
            }

        return sections, dbg

    def genre_feed(self, q: FeedQuery) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
        return self.home_feed(q)