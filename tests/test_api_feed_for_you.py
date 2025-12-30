import time

from fastapi.testclient import TestClient

from musicrec.api.main import app


def test_for_you_requires_user_id_header():
    client = TestClient(app)
    r = client.get("/feed/for-you", params={"country": "Brazil", "n": 10})
    assert r.status_code == 400
    body = r.json()
    assert "X-User-Id" in body.get("detail", "")


def test_for_you_returns_sections_and_debug_and_has_for_you_key():
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
    # Always present
    assert "for_you" in body["sections"]
    assert isinstance(body["sections"]["for_you"], list)

    # Base rails still present
    for key in ["top", "rising", "new_releases", "instrumental", "explicit_safe"]:
        assert key in body["sections"]
        assert isinstance(body["sections"][key], list)

    assert "debug" in body
    assert "sections_returned" in body["debug"]
    assert "fallback_used" in body["debug"]
    assert "suppressed_event_types" in body["debug"]
    assert "personalization" in body["debug"]


def test_for_you_nonempty_after_play_event():
    client = TestClient(app)
    user = "user_fy_play"

    # Pick a track to play
    r0 = client.get("/feed/home", params={"country": "Brazil", "n": 10})
    assert r0.status_code == 200, r0.text
    tid = r0.json()["sections"]["top"][0]["track_id"]
    assert tid

    # Send a play event
    r1 = client.post(
        "/events/feedback",
        headers={"X-User-Id": user},
        json={"track_id": tid, "event_type": "play", "ts": time.time()},
    )
    assert r1.status_code == 200, r1.text
    assert r1.json()["ok"] is True

    # Now for_you should usually produce candidates (may exclude the played track itself)
    r2 = client.get(
        "/feed/for-you",
        params={"country": "Brazil", "n": 10, "debug": "true"},
        headers={"X-User-Id": user},
    )
    assert r2.status_code == 200, r2.text
    body2 = r2.json()
    assert body2["ok"] is True
    assert "for_you" in body2["sections"]
    assert isinstance(body2["sections"]["for_you"], list)
    assert len(body2["sections"]["for_you"]) > 0


def test_for_you_suppresses_disliked_tracks():
    client = TestClient(app)
    user = "user_fy_suppress"

    # Get a track_id from home feed to dislike
    r0 = client.get("/feed/home", params={"country": "Brazil", "n": 10})
    assert r0.status_code == 200, r0.text
    tid = r0.json()["sections"]["top"][0]["track_id"]
    assert tid

    # Dislike it
    r1 = client.post(
        "/events/feedback",
        headers={"X-User-Id": user},
        json={"track_id": tid, "event_type": "dislike", "ts": time.time()},
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
    seen = set()
    for sec, items in body2["sections"].items():
        for it in items:
            if isinstance(it, dict) and it.get("track_id"):
                seen.add(it["track_id"])

    assert tid not in seen
    assert body2["debug"]["suppressed_ids_n"] >= 1