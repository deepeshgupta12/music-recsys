def test_tags_filters_shape(client):
    r = client.get("/tags/filters")
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] is True
    assert "moods" in d
    assert isinstance(d["moods"], list)

    for it in d["moods"][:10]:
        assert "key" in it
        assert "label" in it
        assert "count" in it
        assert isinstance(it["key"], str)
        assert isinstance(it["label"], str)
        assert isinstance(it["count"], int)


def test_tags_filters_debug_optional(client):
    r = client.get("/tags/filters", params={"debug": "true"})
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] is True
    assert "debug" in d
    dbg = d["debug"]
    assert "rows_scanned" in dbg
    assert "unique_moods" in dbg