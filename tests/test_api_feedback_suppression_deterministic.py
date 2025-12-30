from fastapi.testclient import TestClient

from musicrec.api.main import app


def _all_track_ids(sections: dict) -> list[str]:
    out: list[str] = []
    for v in sections.values():
        if not isinstance(v, list):
            continue
        for it in v:
            if isinstance(it, dict) and "track_id" in it:
                out.append(it["track_id"])
    return out


def test_dislike_suppresses_track_from_feed_home_deterministically():
    """
    End-to-end:
      1) Fetch feed for user
      2) Pick a track_id returned by the API
      3) Dislike it
      4) Fetch feed again and ensure the disliked track_id is absent across all sections
      5) debug=true should report suppression count >= 1
    """
    client = TestClient(app)
    user_id = "user_det"

    # 1) First feed
    r1 = client.get(
        "/feed/home",
        params={"country": "Brazil", "n": 10, "debug": "true"},
        headers={"X-User-Id": user_id},
    )
    assert r1.status_code == 200, r1.text
    b1 = r1.json()
    assert b1["ok"] is True
    assert "sections" in b1 and isinstance(b1["sections"], dict)

    ids1 = _all_track_ids(b1["sections"])
    assert len(ids1) > 0

    # 2) Pick a deterministic candidate from the response (first ID)
    victim = ids1[0]

    # 3) Dislike it
    r_post = client.post(
        "/events/feedback",
        headers={"X-User-Id": user_id},
        json={"track_id": victim, "event_type": "dislike"},
    )
    assert r_post.status_code == 200, r_post.text
    assert r_post.json().get("ok") is True

    # 4) Fetch feed again (same user) and ensure victim is gone
    r2 = client.get(
        "/feed/home",
        params={"country": "Brazil", "n": 10, "debug": "true"},
        headers={"X-User-Id": user_id},
    )
    assert r2.status_code == 200, r2.text
    b2 = r2.json()
    assert b2["ok"] is True

    ids2 = _all_track_ids(b2["sections"])
    assert victim not in ids2

    # 5) debug signals
    assert "debug" in b2 and isinstance(b2["debug"], dict)
    assert "disliked_suppressed_count" in b2["debug"]
    assert int(b2["debug"]["disliked_suppressed_count"]) >= 1