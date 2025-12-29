from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _safe_str(x: object) -> str:
    if x is None:
        return ""
    return str(x)


def _parse_release_date(s: object) -> Optional[datetime]:
    """
    Supports strings like:
    - '2015-06-11 00:00:00'
    - '2015-06-11'
    - pandas Timestamp
    """
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    if isinstance(s, pd.Timestamp):
        try:
            return s.to_pydatetime().replace(tzinfo=timezone.utc)
        except Exception:
            return None
    txt = str(s).strip()
    if not txt:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(txt, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue
    try:
        # last-resort: pandas parse
        dt = pd.to_datetime(txt, errors="coerce")
        if pd.isna(dt):
            return None
        if isinstance(dt, pd.Timestamp):
            return dt.to_pydatetime().replace(tzinfo=timezone.utc)
    except Exception:
        pass
    return None


def _days_since(dt: Optional[datetime], now: datetime) -> Optional[int]:
    if dt is None:
        return None
    delta = now - dt
    try:
        return max(int(delta.days), 0)
    except Exception:
        return None


def _compute_momentum(stream_count: int, days_since_release: int) -> float:
    # momentum = log1p(stream_count) / (days_since_release + 1)
    return float(np.log1p(max(int(stream_count), 0)) / float(max(int(days_since_release), 0) + 1))


@dataclass(frozen=True)
class FeedQuery:
    country: str
    n: int = 25
    genre: Optional[str] = None
    explicit_ok: bool = True
    debug: bool = False


class SegmentFeeds:
    """
    Generates non-personalized feeds from the real catalog feature table.
    Feeds (home):
      - top
      - rising (momentum)
      - new_releases
      - instrumental
      - explicit_safe (only if explicit_ok=False, else still provided as a section)
    """

    def __init__(self, feature_table: pd.DataFrame):
        if "track_id" not in feature_table.columns:
            raise ValueError("feature_table must contain 'track_id'")

        self.ft = feature_table.copy()

        # Normalize key columns defensively
        for col in ["track_id", "track_name", "artist_name", "country", "genre", "label"]:
            if col in self.ft.columns:
                self.ft[col] = self.ft[col].astype("string").fillna("")

        if "popularity" in self.ft.columns:
            self.ft["popularity"] = pd.to_numeric(self.ft["popularity"], errors="coerce").fillna(0).astype(int)
        if "stream_count" in self.ft.columns:
            self.ft["stream_count"] = pd.to_numeric(self.ft["stream_count"], errors="coerce").fillna(0).astype(int)

        # explicit can be bool-ish; normalize if present
        if "explicit" in self.ft.columns:
            self.ft["explicit"] = self.ft["explicit"].astype(bool)

    def _apply_base_filters(self, q: FeedQuery) -> pd.DataFrame:
        df = self.ft

        # country filter (required in our API)
        country = (q.country or "").strip()
        if country:
            if "country" in df.columns:
                df = df[df["country"].astype("string") == country]
            else:
                df = df.iloc[0:0]  # no country column => empty

        # genre filter (optional)
        if q.genre:
            g = q.genre.strip()
            if g and "genre" in df.columns:
                # case-insensitive match
                df = df[df["genre"].str.lower() == g.lower()]

        # explicit filter
        if not q.explicit_ok and "explicit" in df.columns:
            df = df[~df["explicit"].astype(bool)]

        return df

    def _add_derived(self, df: pd.DataFrame) -> pd.DataFrame:
        now = datetime.now(timezone.utc)

        out = df.copy()

        # derive days_since_release
        if "days_since_release" in out.columns:
            dsr = pd.to_numeric(out["days_since_release"], errors="coerce").fillna(0).astype(int)
            out["_days_since_release"] = dsr
        else:
            rel = out["release_date"] if "release_date" in out.columns else pd.Series([""] * len(out))
            parsed = rel.apply(_parse_release_date)
            out["_days_since_release"] = parsed.apply(lambda dt: _days_since(dt, now) or 0).astype(int)

        # momentum
        if "stream_count" in out.columns:
            out["_momentum"] = [
                _compute_momentum(sc, dsr)
                for sc, dsr in zip(out["stream_count"].astype(int).tolist(), out["_days_since_release"].astype(int).tolist())
            ]
        else:
            out["_momentum"] = 0.0

        # release datetime for sorting new releases
        if "release_date" in out.columns:
            out["_release_dt"] = out["release_date"].apply(_parse_release_date)
        else:
            out["_release_dt"] = None

        return out

    def _row_to_item(self, row: pd.Series, score: float) -> Dict[str, object]:
        return {
            "track_id": _safe_str(row.get("track_id", "")),
            "score": float(score),
            "track_name": _safe_str(row.get("track_name", "")),
            "artist_name": _safe_str(row.get("artist_name", "")),
            "country": _safe_str(row.get("country", "")),
            "genre": _safe_str(row.get("genre", "")),
            "popularity": int(row.get("popularity", 0) or 0),
            "stream_count": int(row.get("stream_count", 0) or 0),
            "release_date": _safe_str(row.get("release_date", "")),
        }

    def _top(self, df: pd.DataFrame, n: int) -> List[Dict[str, object]]:
        if len(df) == 0:
            return []
        # Prefer stream_count, then popularity
        sort_cols = []
        if "stream_count" in df.columns:
            sort_cols.append("stream_count")
        if "popularity" in df.columns:
            sort_cols.append("popularity")
        if not sort_cols:
            sort_cols = ["track_id"]

        d = df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).head(n)
        out = []
        for _, r in d.iterrows():
            # score = normalized proxy (not used for ranking in UI necessarily)
            score = float(r.get("stream_count", 0))
            out.append(self._row_to_item(r, score))
        return out

    def _rising(self, df: pd.DataFrame, n: int) -> List[Dict[str, object]]:
        if len(df) == 0:
            return []
        if "_momentum" not in df.columns:
            return []
        d = df.sort_values(["_momentum"], ascending=[False]).head(n)
        out = []
        for _, r in d.iterrows():
            out.append(self._row_to_item(r, float(r.get("_momentum", 0.0))))
        return out

    def _new_releases(self, df: pd.DataFrame, n: int) -> List[Dict[str, object]]:
        if len(df) == 0:
            return []
        if "_release_dt" not in df.columns:
            return []
        # drop nulls for sorting
        d = df.copy()
        d = d[d["_release_dt"].notna()]
        if len(d) == 0:
            return []
        d = d.sort_values(["_release_dt"], ascending=[False]).head(n)
        out = []
        for _, r in d.iterrows():
            # score proxy: newer => higher (inverse of days_since)
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
        d["instrumentalness"] = pd.to_numeric(d["instrumentalness"], errors="coerce").fillna(0.0).astype(float)
        d = d[d["instrumentalness"] >= 0.60]
        if len(d) == 0:
            return []
        d = d.sort_values(["instrumentalness"], ascending=[False]).head(n)
        out = []
        for _, r in d.iterrows():
            out.append(self._row_to_item(r, float(r.get("instrumentalness", 0.0))))
        return out

    def _explicit_safe(self, df: pd.DataFrame, n: int) -> List[Dict[str, object]]:
        if len(df) == 0:
            return []
        if "explicit" not in df.columns:
            return []
        d = df[~df["explicit"].astype(bool)]
        return self._top(d, n)

    def home_feed(self, q: FeedQuery) -> Tuple[Dict[str, List[Dict[str, object]]], Dict[str, object]]:
        if q.n <= 0 or q.n > 200:
            raise ValueError("n must be between 1 and 200")

        base = self._apply_base_filters(q)
        base = self._add_derived(base)

        sections: Dict[str, List[Dict[str, object]]] = {
            "top": self._top(base, q.n),
            "rising": self._rising(base, q.n),
            "new_releases": self._new_releases(base, q.n),
            "instrumental": self._instrumental(base, q.n),
            "explicit_safe": self._explicit_safe(base, q.n),
        }

        dbg: Dict[str, object] = {}
        if q.debug:
            dbg = {
                "country": q.country,
                "genre": q.genre,
                "explicit_ok": q.explicit_ok,
                "base_rows": int(len(base)),
                "sections_returned": {k: int(len(v)) for k, v in sections.items()},
            }
        return sections, dbg

    def genre_feed(self, q: FeedQuery) -> Tuple[Dict[str, List[Dict[str, object]]], Dict[str, object]]:
        if not q.genre:
            raise ValueError("genre is required for genre_feed()")
        return self.home_feed(q)