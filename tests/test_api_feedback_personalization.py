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


def _collect_track_ids(sections: dict) -> set[str]:
    out = set()
    for _, items in sections.items():
        for it in items:
            out.add(str(it.get("track_id", "")))
    return out


def test_feed_home_suppresses_disliked_tracks(client):
    # Post a dislike for a track
    client.post(
        "/events/feedback",
        headers={"X-User-Id": "user_p1"},
        json={"track_id": "TRK-DISLIKE-1", "event_type": "dislike", "ts": time.time() - 10},
    )

    # Request feed — should not contain disliked track_id anywhere
    r = client.get("/feed/home", params={"country": "Brazil", "n": 10}, headers={"X-User-Id": "user_p1"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True

    all_ids = _collect_track_ids(body["sections"])
    assert "TRK-DISLIKE-1" not in all_ids