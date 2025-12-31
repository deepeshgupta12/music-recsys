def _mood_rail_keys(sections: dict) -> list[str]:
    if not isinstance(sections, dict):
        return []
    return [k for k in sections.keys() if isinstance(k, str) and k.startswith("mood__")]


def _has_any_tag_fields(item: dict) -> bool:
    if not isinstance(item, dict):
        return False
    drop = ["tags", "tags_missing", "tags_provider", "tags_updated_at"]
    return any(k in item for k in drop)


def test_feed_home_mood_rails_enabled_returns_non_empty_rails(client):
    r = client.get(
        "/feed/home",
        params={
            "country": "Brazil",
            "n": 5,
            "debug": "true",
            "mood_rails": "true",
            "moods_k": 2,
            "mood_rail_n": 5,
        },
    )
    assert r.status_code == 200
    d = r.json()
    assert d.get("ok") is True

    sections = d.get("sections", {})
    mood_keys = _mood_rail_keys(sections)
    assert len(mood_keys) == 2

    for k in mood_keys:
        items = sections.get(k)
        assert isinstance(items, list)
        assert len(items) >= 1
        assert len(items) <= 5

    dbg = d.get("debug", {})
    assert isinstance(dbg, dict)

    mood_dbg = dbg.get("mood_rails", {})
    assert isinstance(mood_dbg, dict)
    assert mood_dbg.get("enabled") is True
    assert mood_dbg.get("moods_k") == 2
    assert mood_dbg.get("mood_rail_n") == 5

    moods_selected = mood_dbg.get("moods_selected")
    assert isinstance(moods_selected, list)
    assert len(moods_selected) == 2

    sections_added = mood_dbg.get("sections_added")
    assert isinstance(sections_added, dict)
    for m in moods_selected:
        assert m in sections_added
        assert int(sections_added[m]) >= 1


def test_feed_home_mood_rails_include_tags_false_strips_tag_fields(client):
    r = client.get(
        "/feed/home",
        params={
            "country": "Brazil",
            "n": 5,
            "debug": "true",
            "mood_rails": "true",
            "moods_k": 1,
            "mood_rail_n": 3,
            "include_tags": "false",
        },
    )
    assert r.status_code == 200
    d = r.json()
    assert d.get("ok") is True

    sections = d.get("sections", {})
    mood_keys = _mood_rail_keys(sections)
    assert len(mood_keys) == 1

    items = sections.get(mood_keys[0])
    assert isinstance(items, list)
    assert len(items) >= 1
    assert len(items) <= 3

    first = items[0]
    assert isinstance(first, dict)
    assert _has_any_tag_fields(first) is False
