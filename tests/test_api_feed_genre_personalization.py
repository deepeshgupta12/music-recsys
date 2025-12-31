import time


def test_feed_genre_debug_contains_reorder_only_personalization(client):
    user_id = "user_genre_p15"

    # 1) Fetch home to discover a real genre string from the dataset
    r_home = client.get("/feed/home", params={"country": "Brazil", "n": 10})
    assert r_home.status_code == 200
    home = r_home.json()

    genre = home["sections"]["top"][0].get("genre")
    assert isinstance(genre, str) and genre.strip()

    # 2) Fetch genre feed baseline (no header) to get track_ids
    r0 = client.get("/feed/genre", params={"country": "Brazil", "genre": genre, "n": 15, "debug": "true"})
    assert r0.status_code == 200
    base = r0.json()

    t0 = base["sections"]["top"][0]["track_id"]
    t1 = base["sections"]["top"][1]["track_id"]

    # 3) Create signals (play/like) for the user
    client.post(
        "/events/feedback",
        headers={"X-User-Id": user_id},
        json={"track_id": t0, "event_type": "play", "ts": time.time() - 5},
    )
    client.post(
        "/events/feedback",
        headers={"X-User-Id": user_id},
        json={"track_id": t0, "event_type": "like", "ts": time.time() - 4},
    )
    client.post(
        "/events/feedback",
        headers={"X-User-Id": user_id},
        json={"track_id": t1, "event_type": "play", "ts": time.time() - 3},
    )

    # 4) Fetch genre feed personalized (header present)
    r1 = client.get(
        "/feed/genre",
        params={"country": "Brazil", "genre": genre, "n": 15, "debug": "true"},
        headers={"X-User-Id": user_id},
    )
    assert r1.status_code == 200
    per = r1.json()

    # 5) Validate personalization debug exists and is reorder-only
    pers = per.get("debug", {}).get("personalization", {})
    assert pers.get("reorder_only") is True
    assert pers.get("user_id") == user_id
    assert pers.get("track_boosts_n", 0) >= 1
    assert isinstance(pers.get("track_boosts_sample", {}), dict)

    # 6) Validate reorder-only property: sets preserved per rail (no dislikes/skips here)
    for rail, items in base["sections"].items():
        if not isinstance(items, list):
            continue
        base_ids = [x.get("track_id") for x in items if isinstance(x, dict)]
        per_ids = [x.get("track_id") for x in per["sections"].get(rail, []) if isinstance(x, dict)]
        assert set(base_ids) == set(per_ids)
        assert len(base_ids) == len(per_ids)