import urllib.parse

def _count_items_and_unique_ids(sections: dict):
    total = 0
    ids = set()
    for _, items in (sections or {}).items():
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            total += 1
            tid = (it.get("track_id") or "").strip()
            if tid:
                ids.add(tid)
    return total, ids


def test_feed_home_tags_join_debug_counters_are_consistent(client):
    r = client.get("/feed/home", params={"country": "Brazil", "n": 5, "debug": "true", "include_tags": "true"})
    assert r.status_code == 200
    d = r.json()
    assert d.get("ok") is True

    sections = d.get("sections", {})
    total_items_seen, ids = _count_items_and_unique_ids(sections)

    # at least one rail item exists (top should exist in normal feed)
    top0 = sections.get("top", [])[0]
    assert all(k in top0 for k in ["tags", "tags_missing", "tags_provider", "tags_updated_at"])

    tj = d.get("debug", {}).get("tags_join")
    assert isinstance(tj, dict)
    assert set(["tracks_requested", "unique_found", "total_items_seen", "tagged_items", "missing_items"]).issubset(set(tj.keys()))

    # key: tracks_requested equals unique ids seen across ALL rails
    assert tj["tracks_requested"] == len(ids)
    assert tj["total_items_seen"] == total_items_seen
    assert tj["tagged_items"] + tj["missing_items"] == total_items_seen


def test_feed_genre_tags_join_debug_counters_are_consistent(client):
    # pick a genre dynamically from home feed to avoid hardcoding
    home = client.get("/feed/home", params={"country": "Brazil", "n": 5, "debug": "false"})
    assert home.status_code == 200
    g = home.json()["sections"]["top"][0]["genre"]
    assert isinstance(g, str) and g.strip()

    r = client.get(
        "/feed/genre",
        params={"country": "Brazil", "genre": g, "n": 10, "debug": "true", "include_tags": "true"},
    )
    assert r.status_code == 200
    d = r.json()
    assert d.get("ok") is True

    sections = d.get("sections", {})
    total_items_seen, ids = _count_items_and_unique_ids(sections)

    top0 = sections.get("top", [])[0]
    assert all(k in top0 for k in ["tags", "tags_missing", "tags_provider", "tags_updated_at"])

    tj = d.get("debug", {}).get("tags_join")
    assert isinstance(tj, dict)
    assert tj["tracks_requested"] == len(ids)
    assert tj["total_items_seen"] == total_items_seen
    assert tj["tagged_items"] + tj["missing_items"] == total_items_seen


def test_feed_for_you_tags_join_debug_counters_are_consistent(client):
    # For-you requires X-User-Id; even with no events, it should return 200 and include rails.
    headers = {"X-User-Id": "test_user_tags_join"}
    r = client.get(
        "/feed/for-you",
        params={"country": "Brazil", "n": 10, "debug": "true", "include_tags": "true"},
        headers=headers,
    )
    assert r.status_code == 200
    d = r.json()
    assert d.get("ok") is True

    sections = d.get("sections", {})
    total_items_seen, ids = _count_items_and_unique_ids(sections)

    # pick any dict item to assert tag fields exist
    any_item = None
    for _, items in sections.items():
        if isinstance(items, list) and items and isinstance(items[0], dict):
            any_item = items[0]
            break
    assert any_item is not None
    assert all(k in any_item for k in ["tags", "tags_missing", "tags_provider", "tags_updated_at"])

    tj = d.get("debug", {}).get("tags_join")
    assert isinstance(tj, dict)
    assert tj["tracks_requested"] == len(ids)
    assert tj["total_items_seen"] == total_items_seen
    assert tj["tagged_items"] + tj["missing_items"] == total_items_seen