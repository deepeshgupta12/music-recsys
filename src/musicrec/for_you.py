from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from musicrec.session_store import SessionEvent


_EVENT_WEIGHT = {
    "play": 1.0,
    "like": 3.0,
    "add_to_playlist": 4.0,
    "skip": -2.0,
    "dislike": -4.0,
}


@dataclass(frozen=True)
class ForYouQuery:
    session_id: str
    n: int = 30
    candidate_k: int = 1200

    same_country_only: bool = False
    country: Optional[str] = None
    explicit_ok: bool = True

    unique_artist: bool = True
    max_per_genre: int = 10

    lambda_relevance: float = 0.75  # MMR: relevance vs diversity
    debug: bool = False


@dataclass(frozen=True)
class ForYouItem:
    track_id: str
    track_name: str
    artist_name: str
    genre: str
    country: str
    popularity: int
    stream_count: int
    release_date: str

    score: float
    sim_to_profile: float
    redundancy_penalty: float
    mmr_score: float


class ForYouRecommender:
    """
    Build a session taste vector from events, then recommend with:
      - profile cosine similarity as primary relevance
      - optional small boosts using popularity/stream_count/freshness
      - MMR diversification + constraints (unique artist, max per genre)
      - exclude already seen track_ids in the session
    """

    def __init__(self, feature_table: pd.DataFrame, X_scaled: np.ndarray):
        if "track_id" not in feature_table.columns:
            raise ValueError("feature_table must contain track_id")
        if len(feature_table) != X_scaled.shape[0]:
            raise ValueError("feature_table rows must match X_scaled rows")

        self.ft = feature_table.reset_index(drop=True).copy()
        self.X = X_scaled.astype("float64", copy=False)

        self.norms = np.linalg.norm(self.X, axis=1)
        self.norms[self.norms == 0.0] = 1e-12

        self.id_to_idx: Dict[str, int] = {}
        tids = self.ft["track_id"].astype("string").fillna("").tolist()
        for i, tid in enumerate(tids):
            self.id_to_idx[str(tid)] = i

    @staticmethod
    def _parse_ts(ts: str) -> datetime:
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return datetime.now(timezone.utc)

    def build_profile_vector(self, events: List[SessionEvent]) -> Tuple[np.ndarray, Dict[str, object]]:
        if not events:
            raise ValueError("No events: cannot build profile")

        now = datetime.now(timezone.utc)
        vecs = []
        ws = []

        used = 0
        skipped_unknown = 0

        for e in events:
            tid = str(e.track_id)
            if tid not in self.id_to_idx:
                skipped_unknown += 1
                continue

            base_w = _EVENT_WEIGHT.get(str(e.event_type).lower(), 0.0)
            if base_w == 0.0:
                continue

            dt = self._parse_ts(str(e.ts))
            age_days = max(0.0, (now - dt).total_seconds() / 86400.0)

            # recency decay: half-life ~ 7 days
            decay = 0.5 ** (age_days / 7.0)

            w = base_w * decay
            idx = self.id_to_idx[tid]
            vecs.append(self.X[idx])
            ws.append(w)
            used += 1

        if used == 0:
            raise ValueError("No usable events (unknown track_ids or zero-weight events)")

        W = np.array(ws, dtype="float64")
        V = np.vstack(vecs).astype("float64")

        profile = (W[:, None] * V).sum(axis=0)
        norm = float(np.linalg.norm(profile))
        if norm == 0.0:
            profile = profile + 1e-9
            norm = float(np.linalg.norm(profile))
        profile = profile / norm

        dbg = {
            "events_total": len(events),
            "events_used": used,
            "events_skipped_unknown_track": skipped_unknown,
        }
        return profile, dbg

    def _cos_to_profile(self, profile: np.ndarray) -> np.ndarray:
        dots = self.X @ profile
        return dots / self.norms

    def _apply_filters_mask(self, q: ForYouQuery, seed_country: Optional[str] = None) -> np.ndarray:
        mask = np.ones(len(self.ft), dtype=bool)

        if not q.explicit_ok and "explicit" in self.ft.columns:
            mask &= (~self.ft["explicit"].astype(bool).to_numpy())

        if q.same_country_only:
            target = q.country or seed_country
            if target is not None:
                mask &= (self.ft["country"].astype("string").to_numpy() == target)

        return mask

    @staticmethod
    def _minmax(x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        mn = float(np.min(x))
        mx = float(np.max(x))
        if abs(mx - mn) < 1e-12:
            return np.ones_like(x) * 0.5
        return (x - mn) / (mx - mn)

    def recommend(
        self,
        q: ForYouQuery,
        events: List[SessionEvent],
    ) -> Tuple[List[ForYouItem], Dict[str, object]]:
        if q.n < 5 or q.n > 200:
            raise ValueError("n must be between 5 and 200")
        if q.candidate_k < max(300, q.n * 10):
            raise ValueError("candidate_k too small; use at least max(300, n*10)")
        if not (0.0 <= q.lambda_relevance <= 1.0):
            raise ValueError("lambda_relevance must be between 0 and 1")

        seen = {str(e.track_id) for e in events}
        profile, prof_dbg = self.build_profile_vector(events)

        seed_country = None
        if q.same_country_only and q.country is None:
            for e in reversed(events):
                tid = str(e.track_id)
                if tid in self.id_to_idx:
                    seed_country = str(self.ft.loc[self.id_to_idx[tid], "country"])
                    break

        sim = self._cos_to_profile(profile)

        mask = self._apply_filters_mask(q, seed_country=seed_country)
        if seen:
            tids = self.ft["track_id"].astype("string").to_numpy()
            mask &= (~np.isin(tids, list(seen)))

        sim_f = np.where(mask, sim, -np.inf)

        available = int(np.isfinite(sim_f).sum())
        k = min(q.candidate_k, available)
        if k <= 0:
            return [], {"available_candidates": available} if q.debug else {}

        cand_idx = np.argpartition(-sim_f, kth=min(k, len(sim_f) - 1))[:k]
        cand_idx = cand_idx[np.argsort(-sim_f[cand_idx])]

        pop = self.ft.loc[cand_idx, "popularity"].to_numpy(dtype="float64")
        sc = self.ft.loc[cand_idx, "stream_count"].to_numpy(dtype="float64")

        if "days_since_release" in self.ft.columns:
            dsr = self.ft.loc[cand_idx, "days_since_release"].to_numpy(dtype="float64")
            freshness = 1.0 - self._minmax(dsr)
        else:
            freshness = np.ones_like(pop) * 0.5

        pop_n = self._minmax(pop)
        sc_n = self._minmax(np.log1p(sc))
        sim_n = self._minmax(sim_f[cand_idx])
        score = 0.80 * sim_n + 0.10 * pop_n + 0.07 * sc_n + 0.03 * freshness

        selected: List[int] = []
        used_artists = set() if q.unique_artist else set()
        genre_counts: Dict[str, int] = {}

        cand_vectors = self.X[cand_idx]
        cand_norms = self.norms[cand_idx]

        def max_sim_to_selected(candidate_pos: int) -> float:
            if not selected:
                return 0.0
            v = cand_vectors[candidate_pos]
            dots = cand_vectors[selected] @ v
            sims = dots / (cand_norms[selected] * cand_norms[candidate_pos])
            return float(np.max(sims))

        out: List[ForYouItem] = []
        for _ in range(q.n):
            best_mmr = -1e18
            best_pos = None
            best_rel = None
            best_red = None

            for pos in range(len(cand_idx)):
                if pos in selected:
                    continue

                ridx = int(cand_idx[pos])
                row = self.ft.iloc[ridx]

                artist = str(row.get("artist_name") or "").strip().lower()
                genre = str(row.get("genre") or "").strip().lower()

                if q.unique_artist and artist:
                    if artist in used_artists:
                        continue

                if q.max_per_genre > 0 and genre:
                    if genre_counts.get(genre, 0) >= q.max_per_genre:
                        continue

                rel = float(score[pos])
                red = max_sim_to_selected(pos)
                mmr = q.lambda_relevance * rel - (1.0 - q.lambda_relevance) * red

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_pos = pos
                    best_rel = rel
                    best_red = red

            if best_pos is None:
                break

            selected.append(best_pos)

            ridx = int(cand_idx[best_pos])
            row = self.ft.iloc[ridx]

            artist = str(row.get("artist_name") or "").strip().lower()
            genre = str(row.get("genre") or "").strip().lower()
            if q.unique_artist and artist:
                used_artists.add(artist)
            if genre:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1

            item = ForYouItem(
                track_id=str(row.get("track_id")),
                track_name=str(row.get("track_name")),
                artist_name=str(row.get("artist_name")),
                genre=str(row.get("genre")),
                country=str(row.get("country")),
                popularity=int(row.get("popularity")) if pd.notna(row.get("popularity")) else 0,
                stream_count=int(row.get("stream_count")) if pd.notna(row.get("stream_count")) else 0,
                release_date=str(row.get("release_date")),
                score=float(best_rel if best_rel is not None else 0.0),
                sim_to_profile=float(sim_f[ridx]) if np.isfinite(sim_f[ridx]) else 0.0,
                redundancy_penalty=float(best_red if best_red is not None else 0.0),
                mmr_score=float(best_mmr),
            )
            out.append(item)

        dbg: Dict[str, object] = {}
        if q.debug:
            dbg = {
                "available_candidates": available,
                "candidate_k": int(q.candidate_k),
                "n_requested": int(q.n),
                "n_returned": int(len(out)),
                # keep both keys: tests expect profile_debug
                "profile_debug": prof_dbg,
                "profile": prof_dbg,
                "constraints": {
                    "same_country_only": bool(q.same_country_only),
                    "country": q.country,
                    "unique_artist": bool(q.unique_artist),
                    "max_per_genre": int(q.max_per_genre),
                },
            }

        return out, dbg


# ----------------------------
# Build vectors from feature table (no sklearn)
# ----------------------------

def _days_since_release_from_release_date(release_date_series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(release_date_series, errors="coerce", utc=True)
    now = datetime.now(timezone.utc)
    days = (now - dt).dt.days
    return days.fillna(3650).clip(lower=0).astype(int)


def build_for_you_recommender_from_feature_table(feature_table: pd.DataFrame) -> ForYouRecommender:
    """
    Create a ForYouRecommender using only the existing feature table.
    Dense matrix:
      - numeric: popularity, log1p(stream_count), days_since_release, explicit (if present)
      - one-hot: genre, country
    Then z-score normalize columns.
    """
    ft = feature_table.copy()

    if "track_id" not in ft.columns:
        raise ValueError("feature_table missing track_id")
    for col in ["track_name", "artist_name", "genre", "country", "popularity", "stream_count", "release_date"]:
        if col not in ft.columns:
            ft[col] = None

    if "days_since_release" not in ft.columns:
        ft["days_since_release"] = _days_since_release_from_release_date(ft["release_date"])

    popularity = pd.to_numeric(ft["popularity"], errors="coerce").fillna(0.0).astype("float64")
    stream = pd.to_numeric(ft["stream_count"], errors="coerce").fillna(0.0).astype("float64")
    days = pd.to_numeric(ft["days_since_release"], errors="coerce").fillna(3650.0).astype("float64")

    explicit = None
    if "explicit" in ft.columns:
        explicit = ft["explicit"].fillna(False).astype(bool).astype("float64")

    num = pd.DataFrame(
        {
            "popularity": popularity,
            "log_stream_count": np.log1p(stream),
            "days_since_release": days,
        }
    )
    if explicit is not None:
        num["explicit"] = explicit

    genre_oh = pd.get_dummies(ft["genre"].fillna("").astype(str).str.strip().str.lower(), prefix="g", dtype="float64")
    country_oh = pd.get_dummies(ft["country"].fillna("").astype(str).str.strip(), prefix="c", dtype="float64")

    X_df = pd.concat([num, genre_oh, country_oh], axis=1)

    X = X_df.to_numpy(dtype="float64", copy=True)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-12] = 1.0
    X_scaled = (X - mu) / sigma

    return ForYouRecommender(ft, X_scaled)


def for_you_item_to_api_dict(it: ForYouItem) -> Dict[str, Any]:
    return {
        "track_id": it.track_id,
        "score": float(it.mmr_score),
        "track_name": it.track_name,
        "artist_name": it.artist_name,
        "country": it.country,
        "genre": it.genre,
        "popularity": int(it.popularity),
        "stream_count": int(it.stream_count),
        "release_date": it.release_date,
    }