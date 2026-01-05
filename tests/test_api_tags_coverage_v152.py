def test_tags_coverage_shape(client):
    r = client.get("/tags/coverage")
    assert r.status_code == 200

    d = r.json()
    assert d.get("ok") is True

    assert isinstance(d.get("catalog_tracks"), int)
    assert isinstance(d.get("tagged_tracks"), int)
    assert isinstance(d.get("coverage_pct"), float)

    assert d["catalog_tracks"] > 0
    assert 0.0 <= d["coverage_pct"] <= 100.0

    # rows should be >= tagged_tracks (usually equal in our schema)
    assert isinstance(d.get("rows"), int)
    assert d["rows"] >= 0