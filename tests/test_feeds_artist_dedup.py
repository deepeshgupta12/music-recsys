import pandas as pd

from musicrec.feeds import SegmentFeeds


def test_dedup_by_artist_is_stable_case_insensitive_and_capped():
    # Minimal feature table needed to construct SegmentFeeds
    sf = SegmentFeeds(pd.DataFrame({"track_id": ["T0"]}))

    items = [
        {"track_id": "T1", "artist_name": "Artist A", "track_name": "x"},
        {"track_id": "T2", "artist_name": "artist a ", "track_name": "y"},  # duplicate (case/space)
        {"track_id": "T3", "artist_name": "Artist B", "track_name": "z"},
        {"track_id": "T4", "artist_name": "", "track_name": "m"},  # missing artist -> unique by track_id
        {"track_id": "T5", "artist_name": "", "track_name": "n"},
    ]

    out = sf._dedup_by_artist(items, n=10)
    assert [x["track_id"] for x in out] == ["T1", "T3", "T4", "T5"]

    out2 = sf._dedup_by_artist(items, n=2)
    assert [x["track_id"] for x in out2] == ["T1", "T3"]