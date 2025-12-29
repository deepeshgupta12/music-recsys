from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

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
        # expects ISO; tolerant fallback
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return datetime.now(timezone.utc)

    def build_profile_vector(self, events: List[SessionEvent]) -> Tuple[np.ndarray, Dict[str, object]]:
        """
        Weighted average of track vectors based on event weights + recency decay.
        """
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

            dt = self._parse_ts(e.ts)
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
            # fallback: small epsilon to avoid NaNs
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
        # cosine(profile, X_i) = (X_i · profile) / ||X_i|| since ||profile||=1
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

        # if same_country_only and no explicit country provided, use the last seen playable track's country as seed_country
        seed_country = None
        if q.same_country_only and q.country is None:
            for e in reversed(events):
                tid = str(e.track_id)
                if tid in self.id_to_idx:
                    seed_country = str(self.ft.loc[self.id_to_idx[tid], "country"])
                    break

        sim = self._cos_to_profile(profile)

        # base mask + exclude seen
        mask = self._apply_filters_mask(q, seed_country=seed_country)
        if seen:
            tids = self.ft["track_id"].astype("string").to_numpy()
            mask &= (~np.isin(tids, list(seen)))

        sim_f = np.where(mask, sim, -np.inf)

        # candidate top-k by similarity
        available = int(np.isfinite(sim_f).sum())
        k = min(q.candidate_k, available)
        if k <= 0:
            return [], {"available_candidates": available} if q.debug else {}

        cand_idx = np.argpartition(-sim_f, kth=min(k, len(sim_f) - 1))[:k]
        cand_idx = cand_idx[np.argsort(-sim_f[cand_idx])]

        # small additional scoring signals (normalized within candidate set)
        pop = self.ft.loc[cand_idx, "popularity"].to_numpy(dtype="float64")
        sc = self.ft.loc[cand_idx, "stream_count"].to_numpy(dtype="float64")
        # prefer fresher => invert days_since_release if present
        if "days_since_release" in self.ft.columns:
            dsr = self.ft.loc[cand_idx, "days_since_release"].to_numpy(dtype="float64")
            freshness = 1.0 - self._minmax(dsr)  # higher = fresher
        else:
            freshness = np.ones_like(pop) * 0.5

        pop_n = self._minmax(pop)
        # log-scale stream_count then normalize
        sc_n = self._minmax(np.log1p(sc))

        # final relevance score (keep similarity dominant)
        sim_n = self._minmax(sim_f[cand_idx])
        score = 0.80 * sim_n + 0.10 * pop_n + 0.07 * sc_n + 0.03 * freshness

        # MMR selection with constraints
        selected: List[int] = []
        used_artists = set() if q.unique_artist else set()
        genre_counts: Dict[str, int] = {}

        # precompute norms in candidate subset for redundancy calculation
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
            best = None  # (mmr, rel, tid)
            best_pos = None
            best_red = None

            for pos in range(len(cand_idx)):
                if pos in selected:
                    continue

                idx = int(cand_idx[pos])
                row = self.ft.loc[idx]

                artist = str(row.get("artist_name", ""))
                genre = str(row.get("genre", ""))

                if q.unique_artist and artist in used_artists:
                    continue
                if q.max_per_genre and genre_counts.get(genre, 0) >= q.max_per_genre:
                    continue

                redundancy = max_sim_to_selected(pos)
                mmr = q.lambda_relevance * float(score[pos]) - (1.0 - q.lambda_relevance) * float(redundancy)

                tid = str(row.get("track_id", ""))
                key = (mmr, float(score[pos]), tid)
                if best is None or key > best:
                    best = key
                    best_pos = pos
                    best_red = redundancy

            if best_pos is None:
                break

            selected.append(best_pos)

            idx = int(cand_idx[best_pos])
            row = self.ft.loc[idx]
            artist = str(row.get("artist_name", ""))
            genre = str(row.get("genre", ""))

            if q.unique_artist:
                used_artists.add(artist)
            genre_counts[genre] = genre_counts.get(genre, 0) + 1

            out.append(
                ForYouItem(
                    track_id=str(row.get("track_id", "")),
                    track_name=str(row.get("track_name", "")),
                    artist_name=artist,
                    genre=genre,
                    country=str(row.get("country", "")),
                    popularity=int(row.get("popularity", 0)),
                    stream_count=int(row.get("stream_count", 0)),
                    release_date=str(row.get("release_date", "")),
                    score=float(score[best_pos]),
                    sim_to_profile=float(sim_f[idx]),
                    redundancy_penalty=float(best_red if best_red is not None else 0.0),
                    mmr_score=float(best[0]),
                )
            )

        debug = {}
        if q.debug:
            debug = {
                "session_id": q.session_id,
                "available_candidates": available,
                "candidate_k": q.candidate_k,
                "returned": len(out),
                "lambda_relevance": q.lambda_relevance,
                "unique_artist": q.unique_artist,
                "max_per_genre": q.max_per_genre,
                "seed_country": seed_country,
                "profile_debug": prof_dbg,
                "genre_counts": dict(sorted(genre_counts.items(), key=lambda x: (-x[1], x[0]))),
            }

        return out, debug