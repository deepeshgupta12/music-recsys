from __future__ import annotations

import json
from fastapi.testclient import TestClient

from musicrec.api.main import app


def _pp(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def main() -> None:
    client = TestClient(app)

    print("1) GET /health")
    r1 = client.get("/health")
    print(_pp(r1.json()))
    assert r1.status_code == 200
    assert r1.json().get("ok") is True

    print("\n2) GET /meta")
    r2 = client.get("/meta")
    print(_pp(r2.json()))
    assert r2.status_code == 200
    assert r2.json().get("ok") is True
    assert r2.json()["catalog"]["n_tracks"] > 0

    print("\nSmoke test complete.")


if __name__ == "__main__":
    main()