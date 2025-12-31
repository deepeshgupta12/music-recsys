from __future__ import annotations

import urllib.parse


def _assert_item_has_tag_fields(item: dict) -> None:
    assert isinstance(item, dict)
    assert "track_id" in item

    # Tag join fields (v1.5.1 Step 2A)
    assert "tags" in item
    assert "tags_missing" in item
    assert "tags_provider" in item
    assert "tags_updated_at" in item

    assert isinstance(item["tags"], dict)
    assert isinstance(item["tags_missing"], bool)

    # provider/updated_at can be None (missing tags)
    if item["tags_provider"] is not None:
        assert isinstance(item["tags_provider"], str)
    if item["tags_updated_at"] is not None:
        assert isinstance(item["tags_updated_at"], (int, float))


def _pick_any_track_item(sections: dict) -> dict:
    # Prefer top[0], else scan any rail list
    top = sections.get("top")
    if isinstance(top, list) and len(top) > 0 and isinstance(top[0], dict):
        return top[0]

    for _, items in sections.items():
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict) and it.get("track_id"):
                    return it

    raise AssertionError("No dict items with track_id found in sections.")


def test_feed_home_include_tags_adds_fields_and_debug(client):
    r = client.get("/feed/home", params={"country": "Brazil", "n": 8, "debug": "true", "include_tags": "true"})
    assert r.status_code == 200
    d = r.json()

    assert d.get("ok") is True
    assert "sections" in d and isinstance(d["sections"], dict)
    assert "debug" in d and isinstance(d["debug"], dict)

    # debug.tags_join must exist when debug=true and include_tags=true
    assert "tags_join" in d["debug"]
    tj = d["debug"]["tags_join"]
    assert isinstance(tj, dict)
    for k in ("tracks_requested", "total_items_seen", "tagged_items", "missing_items"):
        assert k in tj
        assert isinstance(tj[k], int)

    item = _pick_any_track_item(d["sections"])
    _assert_item_has_tag_fields(item)


def test_feed_home_without_include_tags_does_not_add_tag_fields(client):
    r = client.get("/feed/home", params={"country": "Brazil", "n": 8, "debug": "true"})
    assert r.status_code == 200
    d = r.json()
    item = _pick_any_track_item(d["sections"])

    # tags fields should NOT be present by default (include_tags=false)
    assert "tags" not in item
    assert "tags_missing" not in item
    assert "tags_provider" not in item
    assert "tags_updated_at" not in item

    # and debug.tags_join should not be present
    dbg = d.get("debug", {})
    assert "tags_join" not in dbg


def test_feed_genre_include_tags_adds_fields(client):
    # Get a real genre from home feed to avoid guessing
    base = client.get("/feed/home", params={"country": "Brazil", "n": 10, "debug": "true"})
    assert base.status_code == 200
    bd = base.json()
    item = _pick_any_track_item(bd["sections"])
    genre = (item.get("genre") or "").strip()
    assert genre, "Expected at least one item in home feed to have a genre"

    r = client.get(
        "/feed/genre",
        params={"country": "Brazil", "genre": genre, "n": 10, "debug": "true", "include_tags": "true"},
    )
    assert r.status_code == 200
    d = r.json()

    assert d.get("ok") is True
    assert d.get("genre") == genre
    assert "sections" in d and isinstance(d["sections"], dict)
    assert "debug" in d and isinstance(d["debug"], dict)
    assert "tags_join" in d["debug"]

    item2 = _pick_any_track_item(d["sections"])
    _assert_item_has_tag_fields(item2)


def test_feed_for_you_include_tags_adds_fields_even_if_for_you_empty(client):
    # ForYou requires X-User-Id and (ideally) some events.
    user = "user_test_v151_step2"

    # Grab a track_id from home
    base = client.get("/feed/home", params={"country": "Brazil", "n": 10, "debug": "true"})
    assert base.status_code == 200
    bd = base.json()
    seed = _pick_any_track_item(bd["sections"]).get("track_id")
    assert seed

    # Post a couple of events (play + like) so ForYou has signals
    r1 = client.post("/events/feedback", headers={"X-User-Id": user}, json={"track_id": seed, "event_type": "play", "ts": 1700000000.0})
    assert r1.status_code == 200
    assert r1.json().get("ok") is True

    r2 = client.post("/events/feedback", headers={"X-User-Id": user}, json={"track_id": seed, "event_type": "like", "ts": 1700000001.0})
    assert r2.status_code == 200
    assert r2.json().get("ok") is True

    fy = client.get(
        "/feed/for-you",
        params={"country": "Brazil", "n": 10, "debug": "true", "include_tags": "true"},
        headers={"X-User-Id": user},
    )
    assert fy.status_code == 200
    d = fy.json()

    assert d.get("ok") is True
    assert "sections" in d and isinstance(d["sections"], dict)

    # include_tags should enrich any rails that have dict items
    item = _pick_any_track_item(d["sections"])
    _assert_item_has_tag_fields(item)

    # debug.tags_join should exist
    assert "debug" in d and isinstance(d["debug"], dict)
    assert "tags_join" in d["debug"]