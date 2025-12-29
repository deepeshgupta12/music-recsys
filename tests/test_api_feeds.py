from fastapi.testclient import TestClient

from musicrec.api.main import app


def test_feed_home_returns_sections():
    client = TestClient(app)
    r = client.get("/feed/home", params={"country": "Brazil", "n": 10, "debug": "true"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["country"] == "Brazil"
    assert body["n"] == 10
    assert "sections" in body
    assert isinstance(body["sections"], dict)

    # Expected section keys
    for key in ["top", "rising", "new_releases", "instrumental", "explicit_safe"]:
        assert key in body["sections"]
        assert isinstance(body["sections"][key], list)

    # debug exists when debug=true
    assert "debug" in body
    assert "sections_returned" in body["debug"]


def test_feed_genre_requires_genre_and_filters():
    client = TestClient(app)

    r = client.get("/feed/genre", params={"country": "Brazil", "genre": "Rock", "n": 10})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["country"] == "Brazil"
    assert body["genre"] == "Rock"
    assert "sections" in body
    assert isinstance(body["sections"], dict)

    # If results exist in "top", ensure genre matches (case-insensitive)
    top = body["sections"].get("top", [])
    if top:
        assert str(top[0].get("genre", "")).lower() == "rock"