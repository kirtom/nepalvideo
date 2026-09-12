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
