import pandas as pd

from musicrec.feeds import SegmentFeeds


def test_cross_section_dedup_keeps_earlier_sections_and_removes_later_dupes():
    sf = SegmentFeeds(pd.DataFrame({"track_id": ["T0"]}))

    sections = {
        "top": [
            {"track_id": "A", "artist_name": "x", "track_name": "a"},
            {"track_id": "B", "artist_name": "y", "track_name": "b"},
        ],
        "rising": [
            {"track_id": "B", "artist_name": "y", "track_name": "b"},  # dup (should be removed)
            {"track_id": "C", "artist_name": "z", "track_name": "c"},
        ],
        "new_releases": [
            {"track_id": "A", "artist_name": "x", "track_name": "a"},  # dup (should be removed)
            {"track_id": "D", "artist_name": "w", "track_name": "d"},
        ],
        "instrumental": [
            {"track_id": "E", "artist_name": "i", "track_name": "e"},
        ],
        "explicit_safe": [
            {"track_id": "C", "artist_name": "z", "track_name": "c"},  # dup (should be removed)
            {"track_id": "F", "artist_name": "f", "track_name": "f"},
        ],
    }

    out, removed = sf._cross_section_dedup(sections, n_final=10)

    # top keeps A,B
    assert [x["track_id"] for x in out["top"]] == ["A", "B"]

    # rising loses B, keeps C
    assert [x["track_id"] for x in out["rising"]] == ["C"]

    # new_releases loses A, keeps D
    assert [x["track_id"] for x in out["new_releases"]] == ["D"]

    # explicit_safe loses C, keeps F
    assert [x["track_id"] for x in out["explicit_safe"]] == ["F"]

    assert removed["rising"] >= 1
    assert removed["new_releases"] >= 1
    assert removed["explicit_safe"] >= 1