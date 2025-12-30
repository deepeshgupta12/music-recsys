from fastapi.testclient import TestClient

from musicrec.api.main import app


def test_feed_home_has_no_duplicate_track_ids_across_sections():
    client = TestClient(app)
    r = client.get("/feed/home", params={"country": "Brazil", "n": 5, "debug": "true"})
    assert r.status_code == 200, r.text
    body = r.json()
    sections = body["sections"]

    ids = []
    for _, items in sections.items():
        ids.extend([it["track_id"] for it in items if it.get("track_id")])

    assert len(ids) == len(set(ids)), f"Duplicate track_ids across rails: {ids}"


def test_feed_genre_has_no_duplicate_track_ids_across_sections():
    client = TestClient(app)
    r = client.get("/feed/genre", params={"country": "Brazil", "genre": "Rock", "n": 5, "debug": "true"})
    assert r.status_code == 200, r.text
    body = r.json()
    sections = body["sections"]

    ids = []
    for _, items in sections.items():
        ids.extend([it["track_id"] for it in items if it.get("track_id")])

    assert len(ids) == len(set(ids)), f"Duplicate track_ids across rails: {ids}"