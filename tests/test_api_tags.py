import time


def test_tags_stats_shape(client):
    r = client.get("/tags/stats")
    assert r.status_code == 200
    d = r.json()
    assert "rows" in d
    assert "last_updated_at" in d


def test_tags_track_404_for_missing(client):
    r = client.get("/tags/track/TRK-DOES-NOT-EXIST")
    assert r.status_code == 404


def test_tags_batch_works(client):
    # This test is intentionally permissive:
    # - If you haven't tagged these IDs, it should still return 200.
    # - It should return a stable structure.
    r = client.get(
        "/tags/batch",
        params=[("track_id", "TRK-DOES-NOT-EXIST-1"), ("track_id", "TRK-DOES-NOT-EXIST-2"), ("include_missing", "true")],
    )
    assert r.status_code == 200
    d = r.json()
    assert "items" in d
    assert "missing" in d
    assert isinstance(d["items"], list)
    assert isinstance(d["missing"], list)