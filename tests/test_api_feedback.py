import time

import pytest
from fastapi.testclient import TestClient

from musicrec.api.deps import get_feedback_store
from musicrec.api.main import app
from musicrec.storage.feedback_store import FeedbackStore


@pytest.fixture()
def client(tmp_path):
    db_path = tmp_path / "feedback_test.sqlite"
    store = FeedbackStore(str(db_path))

    app.dependency_overrides[get_feedback_store] = lambda: store
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    store.close()


def test_post_feedback_accepts_valid_event(client):
    r = client.post(
        "/events/feedback",
        headers={"X-User-Id": "user_test_1"},
        json={"track_id": "TRK-1", "event_type": "like"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert "event" in body
    assert body["event"]["user_id"] == "user_test_1"
    assert body["event"]["track_id"] == "TRK-1"
    assert body["event"]["event_type"] == "like"


def test_post_feedback_rejects_missing_user_id(client):
    r = client.post("/events/feedback", json={"track_id": "TRK-2", "event_type": "like"})
    assert r.status_code == 400
    assert "X-User-Id" in r.text


def test_post_feedback_rejects_invalid_event_type(client):
    r = client.post(
        "/events/feedback",
        headers={"X-User-Id": "user_test_2"},
        json={"track_id": "TRK-X", "event_type": "something_else"},
    )
    assert r.status_code == 400


def test_feedback_recent_returns_events_desc_ts(client):
    now = time.time()
    # older
    client.post(
        "/events/feedback",
        headers={"X-User-Id": "user_recent"},
        json={"track_id": "TRK-OLD", "event_type": "play", "ts": now - 100},
    )
    # newer
    client.post(
        "/events/feedback",
        headers={"X-User-Id": "user_recent"},
        json={"track_id": "TRK-NEW", "event_type": "like", "ts": now - 10},
    )

    r = client.get(
        "/events/feedback/recent",
        headers={"X-User-Id": "user_recent"},
        params={"limit": 10},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["user_id"] == "user_recent"
    assert len(body["events"]) == 2
    assert body["events"][0]["track_id"] == "TRK-NEW"
    assert body["events"][1]["track_id"] == "TRK-OLD"


def test_feedback_stats_counts_last_n_days(client):
    now = time.time()
    # within window
    client.post(
        "/events/feedback",
        headers={"X-User-Id": "user_stats"},
        json={"track_id": "TRK-1", "event_type": "like", "ts": now - 3600},
    )
    client.post(
        "/events/feedback",
        headers={"X-User-Id": "user_stats"},
        json={"track_id": "TRK-2", "event_type": "like", "ts": now - 1800},
    )
    client.post(
        "/events/feedback",
        headers={"X-User-Id": "user_stats"},
        json={"track_id": "TRK-3", "event_type": "skip", "ts": now - 1200},
    )
    # outside window (2 days ago)
    client.post(
        "/events/feedback",
        headers={"X-User-Id": "user_stats"},
        json={"track_id": "TRK-OLD", "event_type": "dislike", "ts": now - (2 * 86400)},
    )

    r = client.get(
        "/events/feedback/stats",
        headers={"X-User-Id": "user_stats"},
        params={"days": 1},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    counts = body["counts"]
    assert counts.get("like", 0) == 2
    assert counts.get("skip", 0) == 1
    assert counts.get("dislike", 0) == 0