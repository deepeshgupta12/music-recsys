def test_feed_mood_fallback_enabled_returns_non_empty(client):
    r = client.get(
        "/feed/mood",
        params={
            "country": "Brazil",
            "mood": "__no_such_mood__",
            "n": 5,
            "debug": "true",
            "fallback": "true",
        },
    )
    assert r.status_code == 200
    d = r.json()
    assert d.get("ok") is True

    # API returns slugified mood (see _slugify in feed_routes.py)
    assert d.get("mood") == "no-such-mood"

    items = d.get("sections", {}).get("mood", [])
    assert isinstance(items, list)
    assert len(items) > 0  # fallback should populate

    mf = d.get("debug", {}).get("mood_filter", {})
    assert mf.get("fallback_param") is True
    assert mf.get("fallback_used") is True
    assert mf.get("fallback_reason") in ("no_mood_matches", "no_candidates")
    assert mf.get("mood_slug") == "no-such-mood"

    # mood feed always joins internally
    assert "tags_join" in d.get("debug", {})


def test_feed_mood_fallback_disabled_can_be_empty(client):
    r = client.get(
        "/feed/mood",
        params={
            "country": "Brazil",
            "mood": "__no_such_mood__",
            "n": 5,
            "debug": "true",
            "fallback": "false",
        },
    )
    assert r.status_code == 200
    d = r.json()

    assert d.get("mood") == "no-such-mood"

    items = d.get("sections", {}).get("mood", [])
    assert isinstance(items, list)
    assert len(items) == 0

    mf = d.get("debug", {}).get("mood_filter", {})
    assert mf.get("fallback_param") is False
    assert mf.get("fallback_used") is False
    assert mf.get("fallback_reason") == "no_mood_matches"
    assert mf.get("mood_slug") == "no-such-mood"


def test_feed_mood_include_tags_true_has_tag_fields_when_items_exist(client):
    r = client.get(
        "/feed/mood",
        params={
            "country": "Brazil",
            "mood": "__no_such_mood__",
            "n": 5,
            "debug": "true",
            "fallback": "true",
            "include_tags": "true",
        },
    )
    assert r.status_code == 200
    d = r.json()

    assert d.get("mood") == "no-such-mood"

    items = d.get("sections", {}).get("mood", [])
    assert isinstance(items, list)
    assert len(items) > 0

    it = items[0]
    assert isinstance(it, dict)
    for k in ["tags", "tags_missing", "tags_provider", "tags_updated_at"]:
        assert k in it