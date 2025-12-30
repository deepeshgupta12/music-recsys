# music-recsys API (V1.4.7)

Base URL (local dev)
- `http://127.0.0.1:8000`

All endpoints return JSON. Unless stated otherwise, errors follow:
- `400` for invalid/missing parameters
- `404` for unknown resources (e.g., unknown track_id)
- `500` for unexpected server errors

---

## Common object: TrackItem

A track item returned inside `sections` / `results`.

| Field | Type | Notes |
|---|---|---|
| `track_id` | string | Stable identifier, e.g. `TRK-80416E23DD93` |
| `score` | number | Section-specific ranking score |
| `track_name` | string | Track title |
| `artist_name` | string | Primary artist display name |
| `country` | string | Country label used for filtering (example: `Brazil`) |
| `genre` | string | Genre label (example: `Rock`) |
| `popularity` | integer | 0–100 (dataset-derived) |
| `stream_count` | integer | Dataset-derived |
| `release_date` | string | ISO-like string (dataset-derived) |

---

# Feeds (V1.4.x)

Feeds are “rail/section” style endpoints that return multiple curated lists under `sections`.

**Guaranteed behavior added in V1.4.5+**
- No empty rails: if a rail is empty in strict mode (e.g., `country+genre`), the API falls back to a broader filter (typically `country-only`) for that rail.
- Artist de-duplication: within each rail, results are de-duplicated by artist name (case-insensitive, whitespace-trimmed). If `artist_name` is missing/blank, uniqueness falls back to `track_id`.
- Cross-section de-duplication (V1.4.6): the same `track_id` will not appear in multiple rails; duplicates are removed deterministically while preserving section order as much as possible.

---

## GET `/feed/home`

Home feed for a country.

### Query params

| Param | Type | Required | Default | Notes |
|---|---|---:|---:|---|
| `country` | string | Yes | – | Example: `Brazil` |
| `n` | int | No | 25 | 1–200, number of items per rail |
| `explicit_ok` | bool | No | true | If false, filters out explicit tracks |
| `debug` | bool | No | false | If true, includes a `debug` object |

### Response (200)

```json
{
  "ok": true,
  "country": "Brazil",
  "n": 5,
  "sections": {
    "top": [ { "track_id": "…", "score": 20000000.0, "track_name": "…", "artist_name": "…", "country": "Brazil", "genre": "Rock", "popularity": 100, "stream_count": 20000000, "release_date": "2015-10-07 00:00:00" } ],
    "rising": [ { "track_id": "…", "score": 186666.66, "track_name": "…", "artist_name": "…", "country": "Brazil", "genre": "Classical", "popularity": 74, "stream_count": 20000000, "release_date": "2016-07-12 00:00:00" } ],
    "new_releases": [ { "track_id": "…", "score": 0.2, "track_name": "…", "artist_name": "…", "country": "Brazil", "genre": "Indie", "popularity": 57, "stream_count": 2000, "release_date": "2025-12-26 00:00:00" } ],
    "instrumental": [ { "track_id": "…", "score": 0.8, "track_name": "…", "artist_name": "…", "country": "Brazil", "genre": "Jazz", "popularity": 49, "stream_count": 2000, "release_date": "2015-01-01 00:00:00" } ],
    "explicit_safe": [ { "track_id": "…", "score": 20000000.0, "track_name": "…", "artist_name": "…", "country": "Brazil", "genre": "EDM", "popularity": 100, "stream_count": 20000000, "release_date": "2015-05-20 00:00:00" } ]
  },
  "debug": {
    "country": "Brazil",
    "genre": null,
    "explicit_ok": true,
    "base_rows": 8480,
    "sections_returned": { "top": 5, "rising": 5, "new_releases": 5, "instrumental": 5, "explicit_safe": 5 },
    "fallback_used": {},
    "cross_section_dedup_removed": { "top": 0, "rising": 1, "new_releases": 0, "instrumental": 0, "explicit_safe": 0 },
    "n_search": 25
  }
}
```

Notes:
- `sections` is always a dict. Each known rail key maps to a list (possibly empty only if the entire dataset is empty for the filter).
- With `debug=true`, `debug.fallback_used` tells which rails used fallback sourcing.

---

## GET `/feed/genre`

Genre feed for a country + genre.

### Query params

| Param | Type | Required | Default | Notes |
|---|---|---:|---:|---|
| `country` | string | Yes | – | Example: `Brazil` |
| `genre` | string | Yes | – | Example: `Rock` (case-insensitive) |
| `n` | int | No | 25 | 1–200, number of items per rail |
| `explicit_ok` | bool | No | true | If false, filters out explicit tracks |
| `debug` | bool | No | false | If true, includes a `debug` object |

### Response (200)

Same shape as `/feed/home`, plus `genre` at top level:

```json
{
  "ok": true,
  "country": "Brazil",
  "genre": "Rock",
  "n": 5,
  "sections": { "...": [] },
  "debug": { "...": "..." }
}
```

Fallback behavior:
- If a rail (commonly `instrumental`) is empty under strict `country+genre`, the API falls back to a broader query for **that rail only** (typically `country-only`) so the rail is non-empty.
- `debug.fallback_used` will record any rails where fallback was applied.

---

# Other endpoints (legacy; may exist depending on your current branch)

These were present in earlier V1.3.x snapshots and may still exist:

- `GET /health`
- `GET /recommend/similar`
- `GET /recommend/similar_hybrid`
- `GET /playlist/from_seed`
- `POST /session/event`
- `GET /for_you`
- `GET /for_you/from_session`

If you want this `api.md` to include the exact **current** parameters + response schemas for these endpoints too, point me to your latest `src/musicrec/api/main.py` (or paste it), and I’ll regenerate the doc to match your current code precisely.
