import os
import sys
from typing import Any, Dict

import requests


def _base_url() -> str:
    # Default matches our local dev run: uvicorn musicrec.api.main:app --reload
    return os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def _get(path: str, params: Dict[str, Any]) -> None:
    url = _base_url().rstrip("/") + path
    r = requests.get(url, params=params, timeout=15)
    print(r.status_code, r.json())


def main() -> int:
    print("API_BASE_URL =", _base_url())

    # Keep consistent with earlier terminal smoke outputs
    country = "Brazil"
    n = 5

    print("\n1) GET /feed/home")
    _get(
        "/feed/home",
        params={"country": country, "n": n, "explicit_ok": "true", "debug": "true"},
    )

    print("\n2) GET /feed/genre")
    _get(
        "/feed/genre",
        params={"country": country, "genre": "Rock", "n": n, "explicit_ok": "true", "debug": "true"},
    )

    print("\nSmoke test complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())