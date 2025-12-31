def test_feed_mood_basic_shape(client):
    r = client.get("/feed/mood", params={"country": "Brazil", "mood": "late-night-chill", "n": 5})
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] is True
    assert d["country"] == "Brazil"
    assert d["mood"] == "late-night-chill"
    assert "sections" in d
    assert "mood" in d["sections"]
    assert isinstance(d["sections"]["mood"], list)


def test_feed_mood_debug_has_tags_join(client):
    r = client.get(
        "/feed/mood",
        params={"country": "Brazil", "mood": "late-night-chill", "n": 5, "debug": "true"},
    )
    assert r.status_code == 200
    d = r.json()
    assert "debug" in d
    assert "tags_join" in d["debug"]  # mood feed always joins internally
    tj = d["debug"]["tags_join"]
    # stable keys
    for k in ["tracks_requested", "unique_found", "total_items_seen", "tagged_items", "missing_items"]:
        assert k in tj


def test_feed_mood_include_tags_exposes_fields(client):
    r = client.get(
        "/feed/mood",
        params={"country": "Brazil", "mood": "late-night-chill", "n": 5, "include_tags": "true"},
    )
    assert r.status_code == 200
    d = r.json()
    items = d["sections"]["mood"]
    if items:
        it = items[0]
        # tag keys should exist when include_tags=true
        assert "tags" in it
        assert "tags_missing" in it
        assert "tags_provider" in it
        assert "tags_updated_at" in it


def test_feed_mood_without_include_tags_hides_fields(client):
    r = client.get(
        "/feed/mood",
        params={"country": "Brazil", "mood": "late-night-chill", "n": 5, "include_tags": "false"},
    )
    assert r.status_code == 200
    d = r.json()
    items = d["sections"]["mood"]
    if items:
        it = items[0]
        assert "tags" not in it
        assert "tags_missing" not in it
        assert "tags_provider" not in it
        assert "tags_updated_at" not in it