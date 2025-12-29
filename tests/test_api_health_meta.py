from __future__ import annotations

from fastapi.testclient import TestClient
from musicrec.api.main import app


def test_health_endpoint_ok():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["service"] == "music-recsys-api"
    assert "time_utc" in body
    assert "uptime_s" in body


def test_meta_endpoint_has_catalog_stats():
    client = TestClient(app)
    r = client.get("/meta")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True

    cat = body["catalog"]
    assert cat["n_tracks"] > 0
    assert cat["n_ids"] > 0
    assert cat["X_scaled_shape"] is not None
    assert cat["X_scaled_shape"][0] == cat["n_tracks"]