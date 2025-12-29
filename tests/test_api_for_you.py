from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from musicrec.api.main import app


def test_api_session_event_and_for_you_roundtrip():
    client = TestClient(app)

    session_id = f"ses-test-{uuid.uuid4().hex[:10]}"
    seed_track_id = "TRK-BEBD53DA84E1"

    # 1) Add events
    r1 = client.post(
        "/session/event",
        json={"session_id": session_id, "track_id": seed_track_id, "event_type": "play"},
    )
    assert r1.status_code == 200, r1.text
    assert r1.json()["ok"] is True
    assert r1.json()["events_count"] >= 1

    r2 = client.post(
        "/session/event",
        json={"session_id": session_id, "track_id": seed_track_id, "event_type": "like"},
    )
    assert r2.status_code == 200, r2.text
    assert r2.json()["events_count"] >= 2

    # 2) Fetch For You
    r3 = client.get(
        "/for_you",
        params={
            "session_id": session_id,
            "n": 25,
            "candidate_k": 1200,
            "same_country_only": "true",
            "unique_artist": "true",
            "max_per_genre": 10,
            "explicit_ok": "true",
            "debug": "true",
        },
    )
    assert r3.status_code == 200, r3.text
    payload = r3.json()
    assert payload["session_id"] == session_id
    assert payload["returned"] >= 10

    # Should not recommend already seen track_id
    rec_ids = [x["track_id"] for x in payload["results"]]
    assert seed_track_id not in rec_ids