from fastapi.testclient import TestClient

from musicrec.api.main import app


def test_post_feedback_requires_user_header():
    client = TestClient(app)
    r = client.post("/events/feedback", json={"track_id": "TRK-X", "event_type": "like"})
    assert r.status_code == 400
    assert "X-User-Id" in r.text


def test_post_feedback_accepts_valid_event():
    client = TestClient(app)
    r = client.post(
        "/events/feedback",
        headers={"X-User-Id": "user_test_1"},
        json={"track_id": "TRK-X", "event_type": "like"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True


def test_post_feedback_rejects_invalid_event_type():
    client = TestClient(app)
    r = client.post(
        "/events/feedback",
        headers={"X-User-Id": "user_test_2"},
        json={"track_id": "TRK-X", "event_type": "something_else"},
    )
    assert r.status_code == 400