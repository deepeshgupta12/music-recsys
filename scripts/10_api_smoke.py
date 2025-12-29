import os
import uuid
import json
from urllib import request, parse, error


BASE = os.getenv("BASE", "http://127.0.0.1:8000").rstrip("/")
SEED_TRACK_ID = os.getenv("SEED_TRACK_ID", "TRK-BEBD53DA84E1")
SESSION_ID = os.getenv("SESSION_ID", f"ses-smoke-{uuid.uuid4().hex[:10]}")


def pretty(obj):
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def _build_url(path: str, params: dict | None = None) -> str:
    url = f"{BASE}{path}"
    if params:
        qs = parse.urlencode(params, doseq=True)
        url = f"{url}?{qs}"
    return url


def _read_body(resp) -> str:
    try:
        raw = resp.read()
        if raw is None:
            return ""
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return ""


def http_json(method: str, path: str, params: dict | None = None, body: dict | None = None):
    url = _build_url(path, params=params)
    data = None
    headers = {"Accept": "application/json"}

    if body is not None:
        payload = json.dumps(body).encode("utf-8")
        data = payload
        headers["Content-Type"] = "application/json"

    req = request.Request(url=url, method=method.upper(), data=data, headers=headers)

    try:
        with request.urlopen(req, timeout=30) as resp:
            text = _read_body(resp)
            status = getattr(resp, "status", 200)
            try:
                obj = json.loads(text) if text else None
            except json.JSONDecodeError:
                obj = {"_raw": text}
            return status, obj, text, url
    except error.HTTPError as e:
        text = _read_body(e)
        try:
            obj = json.loads(text) if text else None
        except json.JSONDecodeError:
            obj = {"_raw": text}
        return int(e.code), obj, text, url
    except Exception as e:
        return 0, {"detail": f"Client error: {type(e).__name__}: {e}"}, str(e), url


def assert_status(got_status: int, expected: int, method: str, url: str, text: str):
    if got_status != expected:
        raise SystemExit(
            f"\nFAILED {method} {url}\nExpected {expected}, got {got_status}\nBody: {text}\n"
        )


def main():
    print(f"BASE={BASE}")
    print(f"SEED_TRACK_ID={SEED_TRACK_ID}")
    print(f"SESSION_ID={SESSION_ID}\n")

    # 1) Session events
    print("1) POST /session/event (play)")
    status, obj, text, url = http_json(
        "POST",
        "/session/event",
        body={"session_id": SESSION_ID, "track_id": SEED_TRACK_ID, "event_type": "play"},
    )
    assert_status(status, 200, "POST", url, text)
    pretty(obj)
    print()

    print("2) POST /session/event (like)")
    status, obj, text, url = http_json(
        "POST",
        "/session/event",
        body={"session_id": SESSION_ID, "track_id": SEED_TRACK_ID, "event_type": "like"},
    )
    assert_status(status, 200, "POST", url, text)
    pretty(obj)
    print()

    # 2) For You
    print("3) GET /for_you")
    status, obj, text, url = http_json(
        "GET",
        "/for_you",
        params={
            "session_id": SESSION_ID,
            "n": 25,
            "candidate_k": 1200,
            "same_country_only": "true",
            "unique_artist": "true",
            "max_per_genre": 10,
            "explicit_ok": "true",
            "debug": "true",
        },
    )
    assert_status(status, 200, "GET", url, text)
    pretty(obj)
    print()

    # 3) Similar (KNN)
    print("4) GET /recommend/similar")
    status, obj, text, url = http_json(
        "GET",
        "/recommend/similar",
        params={
            "seed_track_id": SEED_TRACK_ID,
            "k": 5,
            "same_country_only": "true",
            "exclude_same_artist": "true",
            "explicit_ok": "true",
            "debug": "true",
        },
    )
    assert_status(status, 200, "GET", url, text)
    pretty(obj)
    print()

    # 4) Similar (Hybrid)
    print("5) GET /recommend/similar_hybrid")
    status, obj, text, url = http_json(
        "GET",
        "/recommend/similar_hybrid",
        params={
            "seed_track_id": SEED_TRACK_ID,
            "k": 5,
            "candidate_k": 200,
            "same_country_only": "true",
            "exclude_same_artist": "true",
            "explicit_ok": "true",
            "debug": "true",
        },
    )
    assert_status(status, 200, "GET", url, text)
    pretty(obj)
    print()

    # 5) Playlist
    print("6) GET /playlist/from_seed")
    status, obj, text, url = http_json(
        "GET",
        "/playlist/from_seed",
        params={
            "seed_track_id": SEED_TRACK_ID,
            "n_tracks": 25,
            "candidate_k": 800,
            "same_country_only": "true",
            "unique_artist": "true",
            "max_per_genre": 8,
            "explicit_ok": "true",
            "debug": "true",
        },
    )
    assert_status(status, 200, "GET", url, text)
    pretty(obj)
    print()

    print("Smoke test complete.")


if __name__ == "__main__":
    main()