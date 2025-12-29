# Music RecSys API (V1)

Base URL (local): `http://127.0.0.1:8000`

This API provides:
- Session event capture (play/like)
- “For You” recommendations from session context
- Similar tracks (KNN)
- Hybrid similar tracks (KNN + rerank blend)
- Playlist generation from a seed (diversified)

## Conventions

### Track ID
- `track_id` is a string like `TRK-BEBD53DA84E1`

### Boolean query params
FastAPI accepts booleans as:
- `true/false` (recommended)
- Also works with Python bools when using TestClient.

### Debug
When `debug=true`, responses include a `debug` object with internal details useful for QA.

---

## POST `/session/event`

Record a user event in a session.

### Body (JSON)
- `session_id` (string, required)
- `track_id` (string, required)
- `event_type` (string, required)  
  Expected events for V1:
  - `play`
  - `like`

### Response (200)
```json
{
  "ok": true,
  "session_id": "ses-xxx",
  "events_count": 2
}
```

### Errors
- 400: invalid payload (missing fields / bad event type)
- 500: storage error (SessionStore failure)

---

## GET `/for_you`

Return recommendations personalized to a session’s events.

### Query params
- `session_id` (string, required)
- `n` (int, default: 25) — number of tracks to return
- `candidate_k` (int, default: 1200) — candidate set size used internally
- `same_country_only` (bool, default: false)
- `unique_artist` (bool, default: false)
- `max_per_genre` (int, default: 10)
- `explicit_ok` (bool, default: true)
- `debug` (bool, default: false)

### Response (200)
```json
{
  "session_id": "ses-xxx",
  "events_count": 2,
  "seed_track_id": "TRK-...",
  "n": 25,
  "returned": 25,
  "results": [
    {
      "track_id": "TRK-...",
      "score": 0.91,
      "track_name": "…",
      "artist_name": "…",
      "country": "Brazil",
      "genre": "Rock",
      "popularity": 50,
      "stream_count": 3000,
      "release_date": "2015-04-02 00:00:00"
    }
  ],
  "debug": {
    "seed_idx": 0,
    "seed_track_name": "…",
    "seed_artist_name": "…",
    "filters": { "...": "..." },
    "available_candidates": 8479,
    "candidate_k": 1200,
    "weights": {
      "w_sim": 0.7,
      "w_momentum": 0.15,
      "w_popularity": 0.1,
      "w_freshness": 0.05
    }
  }
}
```

### Errors
- 404: session not found / no events recorded for session (implementation-dependent)
- 400: invalid params (e.g., candidate_k too small/large, invalid constraints)
- 500: internal error

---

## GET `/recommend/similar`

Return KNN-based similar tracks for a seed.

### Query params
- `seed_track_id` (string, required)
- `k` (int, default: 5)
- `same_country_only` (bool, default: false)
- `country` (string, optional) — if provided, filter to this country
- `exclude_same_artist` (bool, default: false)
- `explicit_ok` (bool, default: true)
- `debug` (bool, default: false)

### Response (200)
```json
{
  "seed_track_id": "TRK-...",
  "k": 5,
  "returned": 5,
  "results": [
    {
      "track_id": "TRK-...",
      "score": 0.96,
      "track_name": "…",
      "artist_name": "…",
      "country": "Brazil",
      "genre": "Classical",
      "popularity": 60,
      "stream_count": 57000,
      "release_date": "2015-06-11 00:00:00"
    }
  ],
  "debug": {
    "seed_idx": 0,
    "seed_track_name": "…",
    "seed_artist_name": "…",
    "filters": { "...": "..." },
    "available_candidates": 8479
  }
}
```

### Errors
- 404: seed_track_id not found
- 400: invalid k / invalid constraints

---

## GET `/recommend/similar_hybrid`

Return similar tracks using:
1) KNN retrieval
2) weighted rerank over multiple signals

### Query params
All params from `/recommend/similar`, plus:
- `candidate_k` (int, default: 200) — candidate pool for hybrid rerank
- `w_sim` (float, default: 0.7)
- `w_momentum` (float, default: 0.15)
- `w_popularity` (float, default: 0.1)
- `w_freshness` (float, default: 0.05)

### Weight guard
At least one weight must be > 0 (sum must be > 0). Otherwise API returns 400.

### Response (200)
```json
{
  "seed_track_id": "TRK-...",
  "k": 5,
  "candidate_k": 200,
  "returned": 5,
  "weights": {
    "w_sim": 0.7,
    "w_momentum": 0.15,
    "w_popularity": 0.1,
    "w_freshness": 0.05
  },
  "results": [ { "...": "..." } ],
  "debug": {
    "available_candidates": 8479,
    "candidate_k": 200,
    "weights": { "...": "..." },
    "hybrid_top_scores": [0.77, 0.63, 0.59]
  }
}
```

### Errors
- 404: seed_track_id not found
- 400: invalid candidate_k (too small / too large), invalid weights

---

## GET `/playlist/from_seed`

Generate a diversified playlist from a seed.

### Query params
- `seed_track_id` (string, required)
- `n_tracks` (int, default: 25)
- `candidate_k` (int, default: 800)
- `same_country_only` (bool, default: false)
- `unique_artist` (bool, default: false)
- `max_per_genre` (int, default: 8)
- `explicit_ok` (bool, default: true)
- `debug` (bool, default: false)

Hybrid weights (same as `/recommend/similar_hybrid`):
- `w_sim`, `w_momentum`, `w_popularity`, `w_freshness` (defaults as above)

### Response (200)
```json
{
  "seed_track_id": "TRK-...",
  "n_tracks": 25,
  "candidate_k": 800,
  "returned": 25,
  "weights": { "...": "..." },
  "playlist": [
    {
      "track_id": "TRK-...",
      "track_name": "…",
      "artist_name": "…",
      "genre": "…",
      "country": "…",
      "popularity": 60,
      "stream_count": 57000,
      "release_date": "2015-06-11 00:00:00",
      "relevance_score": 1.0,
      "redundancy_penalty": 0.96,
      "mmr_score": 0.50
    }
  ],
  "debug": {
    "requested_n": 25,
    "returned_n": 25,
    "candidate_k": 800,
    "lambda_relevance": 0.75,
    "unique_artist": true,
    "max_per_genre": 8,
    "genre_counts": {
      "Rock": 5,
      "Classical": 3
    }
  }
}
```

### Errors
- 404: seed_track_id not found
- 400: invalid candidate_k / invalid weights / invalid constraints
