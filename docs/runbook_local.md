# Local Runbook — Music RecSys API (V1)

This runbook is for local development + smoke verification.

## Prereqs
- Python 3.10.x
- Virtual environment already created (`.venv`)
- Processed feature table exists:
  - `data/processed/catalog_features.parquet`

## Start the API (local)
Run the FastAPI app via Uvicorn (host/port must match smoke script defaults):

- App import path: `musicrec.api.main:app`
- Host: `127.0.0.1`
- Port: `8000`

## Test suite
Run all tests:

- `python -m pytest -q`

Expected: all green.

## Smoke test
Run:

- `python scripts/10_api_smoke.py`

Expected behavior (high level):
1) POST `/session/event` (play) increments `events_count`
2) POST `/session/event` (like) increments `events_count`
3) GET `/for_you` returns `returned == n`, includes `results[]`, and `debug` when enabled
4) GET `/recommend/similar` returns `returned == k`
5) GET `/recommend/similar_hybrid` returns `returned == k` and `weights`
6) GET `/playlist/from_seed` returns `returned == n_tracks` and `playlist[]`

## Where session events are stored
Session events are persisted through `SessionStore`.

To locate the exact directory used:
- Open `src/musicrec/api/main.py`
- Find `_get_session_store()` and the `SessionStore(base_dir=...)` wiring.
- That value is the source of truth for local storage.

## Troubleshooting

### `ModuleNotFoundError: No module named 'requests'`
The smoke script depends on `requests`.

Fix:
- install `requests` in the active `.venv`
- then ensure it’s also recorded in the project dependency file used by this repo (pyproject / requirements).

### Port already in use
- Stop the process holding port `8000`, or run Uvicorn on a different port and also update the smoke script base URL accordingly.

### 404 Not Found from an endpoint
- Verify the server is running.
- Verify the endpoint path matches the docs.
- Re-run `python -m pytest -q` to confirm the API wiring is intact.
