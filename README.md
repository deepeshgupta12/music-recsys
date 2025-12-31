# Music Recommendation System (V0–V3) — Local MVP

A local-first Music Recommendation System built on a processed catalog (Spotify-like feature table) with a FastAPI backend that exposes recommendation endpoints:
- Similar tracks (KNN)
- Similar tracks (Hybrid rerank)
- Playlist generation (MMR-style diversification)
- Session events logging + “For You” recommendations driven by session signals

This repo is designed to be run on a local machine (Mac M1-friendly) with a stepwise, test-first workflow.

---

## What we have implemented so far

### Step 1.3.1 — API: Session events + For You
Implemented and fully tested:
- `POST /session/event` to record session events (`play`, `like`, etc.)
- `GET /for_you` to produce personalized recommendations based on session signals

### Step 1.3.2 — API Runbook + Smoke Test
Implemented:
- Local runbook docs for running the API and validating endpoints
- A deterministic “API smoke test” script (`scripts/10_api_smoke.py`) that calls:
  1) `POST /session/event (play)`
  2) `POST /session/event (like)`
  3) `GET /for_you`
  4) `GET /recommend/similar`
  5) `GET /recommend/similar_hybrid`
  6) `GET /playlist/from_seed`

### Step 1.3.3 — Documentation packaging
Added documentation files for:
- API surface + response shapes
- Local runbook
- API quickstart

### Current Test Status
- `pytest` suite is green: **27 passed**

---

## Repository Structure (relevant paths)

- `src/musicrec/api/main.py`  
  FastAPI app + endpoints + wiring to recommenders and session store.

- `src/musicrec/recommender_knn.py`  
  KNN-based “similar tracks” recommender.

- `src/musicrec/recommender_hybrid.py`  
  Hybrid reranker (similarity + momentum + popularity + freshness).

- `src/musicrec/playlist.py`  
  Playlist generator (diversification / redundancy control).

- `src/musicrec/session_store.py`  
  Session event store used by `/session/event` and `/for_you`.

- `data/processed/catalog_features.parquet`  
  Processed catalog table used by recommenders.

- `scripts/10_api_smoke.py`  
  Local smoke test script for the API.

- `docs/api.md`  
  API documentation.

- `docs/runbook_local.md`  
  Local runbook (how to run, debug, validate).

---

## Local Setup

### 1) Create and activate venv
- Create `.venv` (Python 3.10 recommended)
- Activate it in terminal

### 2) Install dependencies
Install project deps so that the API + scripts run locally.

If you ever see:
- `ModuleNotFoundError: No module named 'requests'`

it means `requests` is not installed in the active venv. Ensure dependencies include `requests` and re-install.

---

## Running the test suite

From repo root:
- Run `pytest` and confirm everything is green before moving ahead.

---

## Running the API locally

Start the FastAPI server (Uvicorn). Once started, API base is typically:
- `http://127.0.0.1:8000`

Open interactive docs:
- `http://127.0.0.1:8000/docs`

---

## API Quickstart (Smoke Flow)

This is the quickest way to validate the system end-to-end.

### A) Start the API server
Run the FastAPI app with Uvicorn (see local runbook for the exact command).

### B) Run the smoke test script
Run:
- `python scripts/10_api_smoke.py`

Expected outcome:
- Prints successful responses for:
  - Session events
  - For You
  - Similar
  - Similar Hybrid
  - Playlist from seed
- Ends with:
  - `Smoke test complete.`

---

## Implemented Endpoints

### 1) POST `/session/event`
Records a session event.

**Request JSON**
```json
{
  "session_id": "ses-123",
  "track_id": "TRK-...",
  "event_type": "play"
}
```

**Response**
```json
{
  "ok": true,
  "session_id": "ses-123",
  "events_count": 1
}
```

---

### 2) GET `/for_you`
Generates personalized recommendations based on events recorded for a `session_id`.

**Key query params**
- `session_id` (required)
- `n` (default 25)
- `candidate_k` (default 1200 in tests/smoke)
- `same_country_only`
- `unique_artist`
- `max_per_genre`
- `explicit_ok`
- `debug`

**Response (shape)**
- `results`: list of tracks with metadata and `score`
- `debug`: optional debug block when `debug=true`

---

### 3) GET `/recommend/similar`
KNN similar tracks.

**Key query params**
- `seed_track_id` (required)
- `k`
- `same_country_only`
- `exclude_same_artist`
- `explicit_ok`
- `debug`

---

### 4) GET `/recommend/similar_hybrid`
Hybrid rerank on top of candidates from similarity retrieval.

**Key query params**
- `seed_track_id` (required)
- `k`
- `candidate_k`
- `same_country_only`
- `exclude_same_artist`
- `explicit_ok`
- `debug`
- Weight knobs:
  - `w_sim`
  - `w_momentum`
  - `w_popularity`
  - `w_freshness`

**Weight guard behavior**
- All weights cannot be zero (guarded by tests).

---

### 5) GET `/playlist/from_seed`
Generates a diversified playlist from a seed.

**Key query params**
- `seed_track_id` (required)
- `n_tracks` (playlist size)
- `candidate_k`
- `same_country_only`
- `unique_artist`
- `max_per_genre`
- `explicit_ok`
- `debug`
- Weight knobs:
  - `w_sim`
  - `w_momentum`
  - `w_popularity`
  - `w_freshness`

**Response (shape)**
- `playlist`: ordered list of tracks including:
  - `relevance_score`
  - `redundancy_penalty`
  - `mmr_score`
- `debug`: includes genre counts, constraints, etc.

---

## Debugging Notes

### 1) Always confirm you are using the correct venv
If imports behave weirdly or packages are missing, confirm:
- `.venv` is activated
- you are running from repo root

### 2) If the API works but tests fail intermittently
Ensure:
- processed data exists at `data/processed/catalog_features.parquet`
- server is not required for tests (FastAPI TestClient runs in-process)
- no stale cached artifacts are being used

---

## Documentation

- `docs/api.md` — full API spec and examples
- `docs/runbook_local.md` — local runbook for running + validating
- `README.md` — this file (high-level entry point)

---

## Scope and Roadmap (V0–V3)

We are building version-wise with explicit sub-steps and test gates.

- V0: Local baseline recommenders + processed catalog
- V1: API layer + session events + “For You” + runbook/smoke
- V2: Evaluation harness + offline metrics + better ranking strategies
- V3: Production hardening patterns (config, caching, observability, deployment story)

Only the parts explicitly implemented and tested are considered “done”.

---

## Workflow Constraints (non-negotiables)

- Visual Studio Code for coding edits
- Terminal for:
  - running tests
  - running scripts
  - running the server
  - git operations
- Stepwise implementation with explicit confirmations
- Full files when updating code (not patch snippets)
- Keep synthetic/local data separate but linked to the “Spotify 2015–2025 catalog” concept
- GitHub workflow with branch-per-step and clear commit messages

---

## Next Step

Proceed to **Step 1.3.3** completion checklist:
- docs are present
- smoke script runs successfully
- tests are green
- changes are pushed to the branch

After this, we move to the next sub-step in the roadmap.
