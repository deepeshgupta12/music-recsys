from fastapi.testclient import TestClient

from musicrec.api.main import app


def test_api_ann_endpoint_returns_results():
    client = TestClient(app)

    # Your existing tests already use this ID; keep consistent
    seed_track_id = "TRK-BEBD53DA84E1"

    r = client.get(
        "/recommend/ann",
        params={
            "seed_track_id": seed_track_id,
            "k": 10,
            "same_country_only": "true",
            "explicit_ok": "true",
            "unique_artist": "true",
            "debug": "true",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["seed_track_id"] == seed_track_id
    assert body["k"] == 10
    assert body["returned"] <= 10
    assert "used_ann" in body
    assert "results" in body
    assert isinstance(body["results"], list)

    # If results exist, validate shape
    if body["results"]:
        item0 = body["results"][0]
        assert "track_id" in item0
        assert "score" in item0
        assert "track_name" in item0
        assert "artist_name" in item0
        assert "country" in item0
        assert "genre" in item0
        assert "release_date" in item0

    # debug block should exist when debug=true
    assert "debug" in body
    assert "ann_loaded" in body["debug"]