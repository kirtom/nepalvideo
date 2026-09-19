"""Scenes and their music targets (Film v2 step 4, task 6) -- pure, no
database, no config reads. See src/nepal/spine/scenes.py for why the
grouping and target rules are what they are."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest

from nepal.spine.scenes import Scene, activity_class, group_scenes, scene_target


def _attrs(*, speed_ms=None, hr_bpm=None, gain_m_per_h=None, alt_m=None,
          hour=None, place_name=None, act=None, levity=False,
          motion_mag=None, under_speech=False):
    return dict(speed_ms=speed_ms, hr_bpm=hr_bpm, gain_m_per_h=gain_m_per_h,
               alt_m=alt_m, hour=hour, place_name=place_name, act=act,
               levity=levity, motion_mag=motion_mag, under_speech=under_speech)


def _slot(i, *, act, t_in, t_out):
    return {"act": act, "t_in": t_in, "t_out": t_out, "shot_id": f"s{i}", "beat_id": None}


def _scene(**kw):
    base = dict(scene_id=1, act=2, t_in=0.0, t_out=60.0, slot_indices=(0,),
               activity="walking", speed_ms=None, hr_bpm=None, gain_m_per_h=None,
               alt_m=None, hour=None, voice_share=0.0, levity=False, picture_energy=None)
    base.update(kw)
    return Scene(**base)


# -- activity_class -- precedence, first match wins ----------------------

def test_activity_class_summit_beats_everything_else():
    a = _attrs(act=4, place_name="Kathmandu", speed_ms=8.0)
    assert activity_class(a, cities=["Kathmandu"]) == "summit"


def test_activity_class_city_by_name():
    a = _attrs(act=1, place_name="Kathmandu", speed_ms=2.0)
    assert activity_class(a, cities=["Kathmandu"]) == "city"


def test_activity_class_city_by_low_altitude():
    a = _attrs(act=5, place_name=None, alt_m=1200.0)
    assert activity_class(a, cities=[]) == "city"


def test_activity_class_transport():
    a = _attrs(act=2, speed_ms=8.0)
    assert activity_class(a, cities=[]) == "transport"


def test_activity_class_resting():
    a = _attrs(act=2, speed_ms=0.1)
    assert activity_class(a, cities=[]) == "resting"


def test_activity_class_crossing_by_keyword():
    a = _attrs(act=3, place_name="the suspension bridge", speed_ms=0.5)
    assert activity_class(a, cities=[]) == "crossing"


def test_activity_class_village():
    a = _attrs(act=2, place_name="Namche", speed_ms=0.5)
    assert activity_class(a, cities=[]) == "village"


def test_activity_class_climbing():
    a = _attrs(act=3, speed_ms=1.0, gain_m_per_h=200.0)
    assert activity_class(a, cities=[]) == "climbing"


def test_activity_class_descending():
    a = _attrs(act=3, speed_ms=1.0, gain_m_per_h=-200.0)
    assert activity_class(a, cities=[]) == "descending"


def test_activity_class_walking_is_the_fallback():
    a = _attrs(act=2, speed_ms=1.5)
    assert activity_class(a, cities=[]) == "walking"


def test_activity_class_none_values_never_match_a_numeric_test():
    a = _attrs(act=2, speed_ms=None, gain_m_per_h=None, alt_m=None, place_name=None)
    assert activity_class(a, cities=[]) == "walking"


# -- group_scenes ----------------------------------------------------------

def test_alternating_activity_yields_at_least_two_scenes():
    """10 slots alternating walking/village never merge across the activity
    boundary, so each stays its own scene."""
    slots, attrs_by_id = [], {}
    t = 0.0
    for i in range(10):
        slots.append(_slot(i, act=2, t_in=t, t_out=t + 15.0))
        village = i % 2 == 1
        attrs_by_id[f"s{i}"] = _attrs(act=2, place_name="Bazaar" if village else None,
                                      speed_ms=0.5 if village else 1.5)
        t += 15.0
    attrs = lambda slot: attrs_by_id[slot["shot_id"]]
    scenes = group_scenes(slots, attrs, min_scene_s=10.0, max_scene_s=1000.0,
                          beats_spans=[], cities=[])
    assert len(scenes) >= 2
    assert len(scenes) == 10


def test_short_scene_merges_into_its_only_same_act_neighbour():
    slots = [_slot(0, act=2, t_in=0.0, t_out=60.0), _slot(1, act=2, t_in=60.0, t_out=90.0)]
    attrs_by_id = {"s0": _attrs(act=2, speed_ms=1.5),                       # walking, 60s
                  "s1": _attrs(act=2, place_name="Bazaar", speed_ms=0.5)}  # village, 30s -- too short
    attrs = lambda slot: attrs_by_id[slot["shot_id"]]
    scenes = group_scenes(slots, attrs, min_scene_s=45.0, max_scene_s=1000.0,
                          beats_spans=[], cities=[])
    assert len(scenes) == 1
    assert scenes[0].t_in == 0.0 and scenes[0].t_out == 90.0


def test_short_scene_takes_the_longer_parts_activity_after_merging():
    slots = [_slot(0, act=2, t_in=0.0, t_out=100.0), _slot(1, act=2, t_in=100.0, t_out=110.0)]
    attrs_by_id = {"s0": _attrs(act=2, speed_ms=1.5),                       # walking, 100s
                  "s1": _attrs(act=2, place_name="Bazaar", speed_ms=0.5)}  # village, 10s
    attrs = lambda slot: attrs_by_id[slot["shot_id"]]
    scenes = group_scenes(slots, attrs, min_scene_s=45.0, max_scene_s=1000.0,
                          beats_spans=[], cities=[])
    assert len(scenes) == 1
    assert scenes[0].activity == "walking"


def test_short_scene_merges_with_the_shorter_of_two_same_act_neighbours():
    slots = [_slot(0, act=2, t_in=0.0, t_out=80.0),      # walking, 80s
            _slot(1, act=2, t_in=80.0, t_out=100.0),    # village, 20s -- too short
            _slot(2, act=2, t_in=100.0, t_out=140.0)]   # resting, 40s -- the shorter neighbour
    attrs_by_id = {"s0": _attrs(act=2, speed_ms=1.5),
                  "s1": _attrs(act=2, place_name="Bazaar", speed_ms=0.5),
                  "s2": _attrs(act=2, speed_ms=0.1)}
    attrs = lambda slot: attrs_by_id[slot["shot_id"]]
    scenes = group_scenes(slots, attrs, min_scene_s=45.0, max_scene_s=1000.0,
                          beats_spans=[], cities=[])
    assert len(scenes) == 2
    assert (scenes[0].t_in, scenes[0].t_out) == (0.0, 80.0)
    assert (scenes[1].t_in, scenes[1].t_out) == (80.0, 140.0)
    assert scenes[1].activity == "resting"          # 40s outweighs the 20s village


def test_long_scene_splits_outside_a_beat_span():
    slots, attrs_by_id = [], {}
    t = 0.0
    for i in range(8):
        slots.append(_slot(i, act=3, t_in=t, t_out=t + 50.0))
        attrs_by_id[f"s{i}"] = _attrs(act=3, speed_ms=1.5)
        t += 50.0
    attrs = lambda slot: attrs_by_id[slot["shot_id"]]
    # 400s total, ceiling 200s -- the boundary nearest the middle (200s) sits
    # inside the beat span, so the split must land somewhere else.
    scenes = group_scenes(slots, attrs, min_scene_s=1.0, max_scene_s=200.0,
                          beats_spans=[(190.0, 210.0)], cities=[])
    assert len(scenes) >= 2
    assert all((s.t_out - s.t_in) <= 200.0 for s in scenes)
    boundaries = sorted({s.t_in for s in scenes} | {s.t_out for s in scenes})
    assert not any(190.0 < b < 210.0 for b in boundaries)


def test_beat_span_covering_every_boundary_leaves_the_scene_unsplit():
    slots, attrs_by_id = [], {}
    t = 0.0
    for i in range(4):
        slots.append(_slot(i, act=2, t_in=t, t_out=t + 100.0))
        attrs_by_id[f"s{i}"] = _attrs(act=2, speed_ms=1.0)
        t += 100.0
    attrs = lambda slot: attrs_by_id[slot["shot_id"]]
    scenes = group_scenes(slots, attrs, min_scene_s=1.0, max_scene_s=200.0,
                          beats_spans=[(0.0, 400.0)], cities=[])
    assert len(scenes) == 1
    assert scenes[0].t_in == 0.0 and scenes[0].t_out == 400.0


def test_empty_slots_yield_no_scenes():
    assert group_scenes([], lambda s: _attrs(), min_scene_s=45.0, max_scene_s=150.0,
                        beats_spans=[], cities=[]) == []


def test_a_single_slot_is_one_scene_even_when_shorter_than_the_floor():
    slots = [_slot(0, act=2, t_in=0.0, t_out=5.0)]
    scenes = group_scenes(slots, lambda s: _attrs(act=2, speed_ms=1.0),
                          min_scene_s=45.0, max_scene_s=150.0, beats_spans=[], cities=[])
    assert len(scenes) == 1
    assert scenes[0].t_in == 0.0 and scenes[0].t_out == 5.0


def test_all_none_attrs_default_to_walking_with_none_aggregates():
    slots = [_slot(0, act=2, t_in=0.0, t_out=10.0)]
    scenes = group_scenes(slots, lambda s: _attrs(act=2), min_scene_s=1.0,
                          max_scene_s=1000.0, beats_spans=[], cities=[])
    scene = scenes[0]
    assert scene.activity == "walking"
    assert scene.speed_ms is None and scene.hr_bpm is None and scene.alt_m is None
    assert scene.hour is None and scene.picture_energy is None
    assert scene.voice_share == 0.0
    assert scene.levity is False


def test_voice_share_is_duration_weighted():
    slots = [_slot(0, act=2, t_in=0.0, t_out=30.0), _slot(1, act=2, t_in=30.0, t_out=90.0)]
    attrs_by_id = {"s0": _attrs(act=2, speed_ms=1.0, under_speech=True),
                  "s1": _attrs(act=2, speed_ms=1.0, under_speech=False)}
    attrs = lambda slot: attrs_by_id[slot["shot_id"]]
    scenes = group_scenes(slots, attrs, min_scene_s=1.0, max_scene_s=1000.0,
                          beats_spans=[], cities=[])
    assert len(scenes) == 1
    assert scenes[0].voice_share == pytest.approx(30.0 / 90.0)


def test_scene_ids_are_1_based_in_film_order():
    # Three different acts, so none of them are merge-eligible neighbours --
    # exactly three scenes come out, in the order the slots were given.
    slots, attrs_by_id = [], {}
    t = 0.0
    for i, act in enumerate((2, 3, 4)):
        slots.append(_slot(i, act=act, t_in=t, t_out=t + 60.0))
        attrs_by_id[f"s{i}"] = _attrs(act=act, speed_ms=0.5)
        t += 60.0
    attrs = lambda slot: attrs_by_id[slot["shot_id"]]
    scenes = group_scenes(slots, attrs, min_scene_s=1.0, max_scene_s=1000.0,
                          beats_spans=[], cities=[])
    assert [s.scene_id for s in scenes] == [1, 2, 3]
    assert [s.t_in for s in scenes] == [0.0, 60.0, 120.0]


# -- scene_target ------------------------------------------------------

def test_climb_energy_exceeds_village_energy():
    climb = _scene(act=3, activity="climbing", gain_m_per_h=300.0, speed_ms=1.0)
    village = _scene(act=2, activity="village", gain_m_per_h=0.0, speed_ms=0.5)
    climb_target = scene_target(climb, hr_rest=60.0, hr_max=170.0)
    village_target = scene_target(village, hr_rest=60.0, hr_max=170.0)
    assert climb_target["energy"] > village_target["energy"]


def test_heart_rate_drives_energy_when_present():
    hard = _scene(act=3, activity="climbing", hr_bpm=160.0)
    easy = _scene(act=3, activity="climbing", hr_bpm=70.0)
    assert (scene_target(hard, hr_rest=60.0, hr_max=170.0)["energy"] >
           scene_target(easy, hr_rest=60.0, hr_max=170.0)["energy"])


def test_summit_brightness_is_low():
    summit = _scene(act=4, activity="summit", alt_m=5100.0, hour=6.5)
    village_daytime = _scene(act=2, activity="village", alt_m=1400.0, hour=12.0)
    summit_target = scene_target(summit, hr_rest=60.0, hr_max=170.0)
    village_target = scene_target(village_daytime, hr_rest=60.0, hr_max=170.0)
    assert summit_target["brightness"] < village_target["brightness"]
    assert summit_target["brightness"] <= 0.3


def test_pre_dawn_scene_is_dark_regardless_of_altitude():
    predawn = _scene(act=1, activity="walking", alt_m=1400.0, hour=4.0)
    assert scene_target(predawn, hr_rest=60.0, hr_max=170.0)["brightness"] == pytest.approx(0.2)


def test_speech_heavy_scene_has_low_dynamics_unless_it_is_a_climb():
    talky = _scene(activity="walking", voice_share=0.8)
    assert scene_target(talky, hr_rest=60.0, hr_max=170.0)["dynamics"] == pytest.approx(0.3)
    climbing_and_talky = _scene(activity="climbing", voice_share=0.8)
    assert scene_target(climbing_and_talky, hr_rest=60.0, hr_max=170.0)["dynamics"] == pytest.approx(0.8)


def test_tempo_rises_with_speed_and_stays_bounded():
    still = _scene(speed_ms=0.0)
    fast = _scene(speed_ms=10.0)
    slow_target = scene_target(still, hr_rest=60.0, hr_max=170.0)
    fast_target = scene_target(fast, hr_rest=60.0, hr_max=170.0)
    assert slow_target["tempo_bpm"] == pytest.approx(60.0)
    assert fast_target["tempo_bpm"] == pytest.approx(150.0)   # 60 + 60*1.5 cap
