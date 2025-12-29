# Music RecSys API (V1.3.x) — Runbook

## Prereqs
- Python venv activated
- Required data exists:
  - data/processed/catalog_features.parquet
  - (any other artifact files your API loads at startup)

## Run locally
Run from repo root:

uvicorn musicrec.api.main:app --reload --port 8000

Server: http://127.0.0.1:8000  
Swagger docs: http://127.0.0.1:8000/docs

## Endpoints

### 1) Similar tracks (KNN)
GET /recommend/similar

Query params:
- seed_track_id (str)
- k (int)
- same_country_only (bool)
- exclude_same_artist (bool)
- explicit_ok (bool)
- debug (bool)

Expected:
- 200 with list of tracks
- 404 for invalid seed_track_id

### 2) Similar tracks (Hybrid rerank)
GET /recommend/similar_hybrid

Query params:
- seed_track_id (str)
- k (int)
- candidate_k (int)
- same_country_only (bool)
- exclude_same_artist (bool)
- explicit_ok (bool)
- debug (bool)

Optional weights:
- w_sim
- w_momentum
- w_popularity
- w_freshness

Guard:
- if all weights are 0 => 400

### 3) Playlist from seed
GET /playlist/from_seed

Query params:
- seed_track_id (str)
- n_tracks (int)
- candidate_k (int)
- same_country_only (bool)
- unique_artist (bool)
- max_per_genre (int)
- explicit_ok (bool)
- debug (bool)

Optional weights:
- w_sim
- w_momentum
- w_popularity
- w_freshness

Guard:
- if all weights are 0 => 400

### 4) Session events
POST /session/event

JSON body:
- session_id (str)
- track_id (str)
- event_type (str: play|like|skip|...)

Response:
- ok (bool)
- events_count (int)

### 5) For You (from session)
GET /for_you

Query params:
- session_id (str)
- n (int)
- candidate_k (int)
- same_country_only (bool)
- unique_artist (bool)
- max_per_genre (int)
- explicit_ok (bool)
- debug (bool)

Expected:
- 200 with results
- 404 if session_id not found / no events