from __future__ import annotations

from musicrec.personalization import (
    PersonalizationConfig,
    apply_reorder_only_personalization,
    reorder_section_reorder_only,
)


def test_reorder_section_reorder_only_moves_boosted_track_up():
    items = [
        {"track_id": "A", "score": 10.0},
        {"track_id": "B", "score": 9.0},
        {"track_id": "C", "score": 8.0},
        {"track_id": "D", "score": 7.0},
    ]
    boosts = {"C": 5.0}  # boosted track

    out, dbg = reorder_section_reorder_only(items, track_boosts=boosts, cfg=PersonalizationConfig())
    assert [x["track_id"] for x in out][0] == "C"
    assert dbg["boosted_items"] == 1
    assert dbg["moved"] >= 1


def test_reorder_section_is_stable_when_no_boosts():
    items = [
        {"track_id": "A", "score": 10.0},
        {"track_id": "B", "score": 9.0},
        {"track_id": "C", "score": 8.0},
    ]
    boosts = {}

    out, dbg = reorder_section_reorder_only(items, track_boosts=boosts, cfg=PersonalizationConfig())
    assert [x["track_id"] for x in out] == ["A", "B", "C"]
    assert dbg["boosted_items"] == 0


def test_apply_reorder_only_personalization_preserves_lengths():
    sections = {
        "top": [{"track_id": "A", "score": 10.0}, {"track_id": "B", "score": 9.0}],
        "rising": [{"track_id": "C", "score": 3.0}],
        "for_you": [{"track_id": "Z", "score": 1.0}],
    }
    boosts = {"B": 2.0}

    out, dbg = apply_reorder_only_personalization(
        sections,
        track_boosts=boosts,
        section_names=["top", "rising"],
        cfg=PersonalizationConfig(),
    )

    assert len(out["top"]) == 2
    assert len(out["rising"]) == 1
    assert len(out["for_you"]) == 1  # untouched
    assert dbg["personalization_reorder_only_length_deltas"]["top"] == 0
    assert dbg["personalization_reorder_only_length_deltas"]["rising"] == 0