"""S03.2 -- what counts as a shot, kept separate from detecting one."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest

from nepal.process.shots import merge_short_scenes, shots_for_recording


def test_a_short_scene_is_absorbed_rather_than_dropped():
    """Discarding it would move the boundary and leave a hole in the recording."""
    assert merge_short_scenes([(0, 4), (4, 4.8), (4.8, 9)]) == [(0, 4.8), (4.8, 9)]


def test_a_short_leading_scene_takes_from_its_successor():
    """It has no predecessor to fold into."""
    assert merge_short_scenes([(0, 0.9), (0.9, 6)]) == [(0, 6)]


def test_the_total_span_is_preserved_by_merging():
    scenes = [(0, 4), (4, 4.5), (4.5, 5.2), (5.2, 11)]
    merged = merge_short_scenes(scenes)
    assert merged[0][0] == scenes[0][0]
    assert merged[-1][1] == scenes[-1][1]


def test_zero_length_scenes_are_discarded():
    assert merge_short_scenes([(0, 3), (3, 3), (3, 7)]) == [(0, 3), (3, 7)]


def test_every_surviving_shot_meets_the_minimum():
    merged = merge_short_scenes([(0, 2), (2, 2.4), (2.4, 2.9), (2.9, 8)], 1.5)
    assert all(b - a >= 1.5 for a, b in merged)


def test_a_single_short_recording_is_still_one_shot():
    """A one-second clip is poor material, but the quality gate rejects it --
    detection should not silently produce nothing."""
    assert merge_short_scenes([(0, 1.0)], 1.5) == [(0, 1.0)]


def test_nothing_in_nothing_out():
    assert merge_short_scenes([]) == []


def test_shot_ids_carry_the_recording_and_the_index():
    rows = shots_for_recording([(0, 4), (4, 9.5)], "camera_20240506_0812")
    assert [r["shot_id"] for r in rows] == ["camera_20240506_0812#0000",
                                           "camera_20240506_0812#0001"]


def test_shot_ids_are_stable_across_runs():
    """Detection is deterministic, and a shot whose id moved would orphan every
    score and vote attached to it."""
    scenes = [(0, 4), (4, 9.5), (9.5, 14)]
    first = [r["shot_id"] for r in shots_for_recording(scenes, "r")]
    second = [r["shot_id"] for r in shots_for_recording(scenes, "r")]
    assert first == second


def test_a_video_shot_belongs_to_a_recording_not_an_asset():
    row = shots_for_recording([(0, 4)], "r")[0]
    assert row["recording_id"] == "r"
    assert row["asset_id"] is None
    assert row["media_kind"] == "video"


# -- long scenes ------------------------------------------------------
def test_a_scene_within_the_limit_is_left_exactly_as_it_is():
    from nepal.process.shots import split_long_scenes
    assert split_long_scenes([(0.0, 12.0)], 20.0) == [(0.0, 12.0)]


def test_a_long_scene_becomes_equal_pieces_covering_the_same_span():
    """A 360 camera on a walking person does not cut. The longest recording in
    this corpus is 29 minutes with no detected boundary at all, and one shot of
    29 minutes is not a unit the timeline can choose between."""
    from nepal.process.shots import split_long_scenes
    out = split_long_scenes([(0.0, 100.0)], 20.0)
    assert len(out) == 5
    assert out[0][0] == 0.0 and out[-1][1] == 100.0
    spans = {round(b - a, 3) for a, b in out}
    assert spans == {20.0}
    assert all(out[i][1] == out[i + 1][0] for i in range(len(out) - 1))


def test_nothing_is_left_over_as_a_stub():
    """Equal pieces, not fixed-length ones: 50 s at a 20 s limit is two 25 s
    shots, never 20 + 20 + 10."""
    from nepal.process.shots import split_long_scenes
    out = split_long_scenes([(0.0, 50.0)], 20.0)
    assert [round(b - a, 1) for a, b in out] == [25.0, 25.0]


def test_the_division_never_produces_a_shot_below_the_minimum():
    from nepal.process.shots import split_long_scenes
    out = split_long_scenes([(0.0, 3.0)], 1.0, min_len_s=1.5)
    assert all(b - a >= 1.5 for a, b in out)


def test_a_zero_limit_disables_the_division():
    from nepal.process.shots import split_long_scenes
    assert split_long_scenes([(0.0, 1800.0)], 0.0) == [(0.0, 1800.0)]


def test_a_recording_with_no_cuts_still_yields_usable_shots():
    from nepal.process.shots import shots_for_recording
    rows = shots_for_recording([(0.0, 120.0)], "camera_x", max_len_s=20.0)
    assert len(rows) == 6
    assert [r["shot_id"] for r in rows][:2] == ["camera_x#0000", "camera_x#0001"]
    assert rows[-1]["end_s"] == 120.0
