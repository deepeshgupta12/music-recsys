from fastapi.testclient import TestClient

from musicrec.api.main import app


def test_genre_feed_instrumental_section_nonempty_via_fallback():
    """
    In strict (country+genre) mode, some sections can be empty (e.g., instrumental).
    V1.4.5 ensures we do not return empty rails by falling back to country-only
    for that section.
    """
    client = TestClient(app)

    r = client.get(
        "/feed/genre",
        params={"country": "Brazil", "genre": "Rock", "n": 5, "debug": "true"},
    )
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["ok"] is True
    assert "sections" in body
    assert "instrumental" in body["sections"]
    assert isinstance(body["sections"]["instrumental"], list)
    assert len(body["sections"]["instrumental"]) > 0

    # debug=true should include fallback_used (empty dict if not needed)
    assert "debug" in body
    assert "fallback_used" in body["debug"]
    assert isinstance(body["debug"]["fallback_used"], dict)