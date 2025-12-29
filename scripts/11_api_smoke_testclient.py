import json
import uuid

import pandas as pd
from fastapi.testclient import TestClient

from musicrec.api.main import app


def _pp(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def main() -> None:
    df = pd.read_parquet("data/processed/catalog_features.parquet")
    seed_track_id = str(df.iloc[0]["track_id"])
    session_id = f"ses-ci-{uuid.uuid4().hex[:10]}"

    client = TestClient(app)

    print("SEED_TRACK_ID=", seed_track_id)
    print("SESSION_ID=", session_id)

    # 1) session events
    r1 = client.post("/session/event", json={"session_id": session_id, "track_id": seed_track_id, "event_type": "play"})
    assert r1.status_code == 200, r1.text
    print("\n1) POST /session/event (play)\n", _pp(r1.json()))

    r2 = client.post("/session/event", json={"session_id": session_id, "track_id": seed_track_id, "event_type": "like"})
    assert r2.status_code == 200, r2.text
    print("\n2) POST /session/event (like)\n", _pp(r2.json()))

    # 2) for_you
    r3 = client.get(
        "/for_you",
        params={
            "session_id": session_id,
            "n": 25,
            "candidate_k": 1200,
            "same_country_only": True,
            "unique_artist": True,
            "max_per_genre": 10,
            "explicit_ok": True,
            "debug": True,
        },
    )
    assert r3.status_code == 200, r3.text
    body3 = r3.json()
    assert body3.get("returned", 0) == 25
    print("\n3) GET /for_you\n", _pp({k: body3[k] for k in body3.keys() if k != "results"}))
    print("   results[0] =", _pp(body3["results"][0]))

    # 3) similar
    r4 = client.get(
        "/recommend/similar",
        params={
            "seed_track_id": seed_track_id,
            "k": 5,
            "same_country_only": True,
            "exclude_same_artist": True,
            "explicit_ok": True,
            "debug": True,
        },
    )
    assert r4.status_code == 200, r4.text
    body4 = r4.json()
    assert body4.get("returned", 0) == 5
    print("\n4) GET /recommend/similar\n", _pp({k: body4[k] for k in body4.keys() if k != "results"}))

    # 4) similar_hybrid
    r5 = client.get(
        "/recommend/similar_hybrid",
        params={
            "seed_track_id": seed_track_id,
            "k": 5,
            "candidate_k": 200,
            "same_country_only": True,
            "exclude_same_artist": True,
            "explicit_ok": True,
            "debug": True,
        },
    )
    assert r5.status_code == 200, r5.text
    body5 = r5.json()
    assert body5.get("returned", 0) == 5
    print("\n5) GET /recommend/similar_hybrid\n", _pp({k: body5[k] for k in body5.keys() if k != "results"}))

    # 5) playlist/from_seed
    r6 = client.get(
        "/playlist/from_seed",
        params={
            "seed_track_id": seed_track_id,
            "n_tracks": 25,
            "candidate_k": 800,
            "same_country_only": True,
            "unique_artist": True,
            "max_per_genre": 8,
            "explicit_ok": True,
            "debug": True,
        },
    )
    assert r6.status_code == 200, r6.text
    body6 = r6.json()
    assert body6.get("returned", 0) == 25
    print("\n6) GET /playlist/from_seed\n", _pp({k: body6[k] for k in body6.keys() if k != "playlist"}))
    print("Smoke test complete.")


if __name__ == "__main__":
    main()