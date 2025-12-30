import time

from fastapi.testclient import TestClient

from musicrec.api.main import app


def test_for_you_requires_user_id_header():
    client = TestClient(app)
    r = client.get("/feed/for-you", params={"country": "Brazil", "n": 10})
    assert r.status_code == 400
    body = r.json()
    assert "X-User-Id" in body.get("detail", "")


def test_for_you_returns_sections_and_debug():
    client = TestClient(app)

    r = client.get(
        "/feed/for-you",
        params={"country": "Brazil", "n": 10, "debug": "true"},
        headers={"X-User-Id": "user_fy"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["country"] == "Brazil"
    assert body["n"] == 10

    assert "sections" in body
    for key in ["top", "rising", "new_releases", "instrumental", "explicit_safe"]:
        assert key in body["sections"]
        assert isinstance(body["sections"][key], list)

    assert "debug" in body
    assert "sections_returned" in body["debug"]
    assert "fallback_used" in body["debug"]
    assert "suppressed_event_types" in body["debug"]
    assert "personalization" in body["debug"]


def test_for_you_suppresses_disliked_tracks():
    client = TestClient(app)
    user = "user_fy_suppress"

    # Get a track_id from home feed to dislike
    r0 = client.get("/feed/home", params={"country": "Brazil", "n": 10})
    assert r0.status_code == 200, r0.text
    body0 = r0.json()
    tid = body0["sections"]["top"][0]["track_id"]
    assert tid

    # Dislike it
    now = time.time()
    r1 = client.post(
        "/events/feedback",
        headers={"X-User-Id": user},
        json={"track_id": tid, "event_type": "dislike", "ts": now},
    )
    assert r1.status_code == 200, r1.text
    assert r1.json()["ok"] is True

    # For-you should not contain that track_id anywhere
    r2 = client.get(
        "/feed/for-you",
        params={"country": "Brazil", "n": 20, "debug": "true"},
        headers={"X-User-Id": user},
    )
    assert r2.status_code == 200, r2.text
    body2 = r2.json()
    assert body2["ok"] is True

    seen = set()
    for sec, items in body2["sections"].items():
        for it in items:
            if isinstance(it, dict) and it.get("track_id"):
                seen.add(it["track_id"])

    assert tid not in seen
    assert body2["debug"]["suppressed_ids_n"] >= 1