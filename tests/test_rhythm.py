"""Re-timing an act's slots against its own music (Film v2 step 4, task 8) --
pure, no database. See src/nepal/process/rhythm.py for what each rule means
and why the clamp is always the last word."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest

from nepal.process.assemble import SLOT_KEYS
from nepal.process.rhythm import energy_percentiles, retime, target_length

TABLE = {"low": [5.0, 8.0], "mid": [3.0, 5.0], "high": [1.5, 2.5]}


def _slot(shot_id, t_in, *, act=2, locked=0, **overrides):
    slot = {k: None for k in SLOT_KEYS if k != "slot_index"}
    slot.update(act=act, t_in=t_in, t_out=None, kind="video", shot_id=shot_id,
                src_in=0.0, src_out=0.0, speed=1.0, transition="cut", locked=locked)
    slot.update(overrides)
    return slot


def _shot(start_s=0.0, end_s=1000.0, media_kind="video"):
    return {"media_kind": media_kind, "start_s": start_s, "end_s": end_s}


# -- energy_percentiles: the mid-rank convention -----------------------

def test_percentile_ties_count_half():
    sections = [{"energy": 5.0}, {"energy": 5.0}, {"energy": 10.0}]
    pcts = energy_percentiles(sections)
    assert pcts[0] == pytest.approx(100 / 3)
    assert pcts[1] == pytest.approx(100 / 3)
    assert pcts[2] == pytest.approx(250 / 3)


def test_percentile_missing_energy_counts_as_zero():
    sections = [{"energy": None}, {"energy": 10.0}]
    pcts = energy_percentiles(sections)
    assert pcts[0] == pytest.approx(25.0)
    assert pcts[1] == pytest.approx(75.0)


# -- target_length: the brief's bands, boundaries inclusive to mid -----

def test_target_length_bands():
    assert target_length(10.0, TABLE) == (5.0, 8.0)
    assert target_length(33.0, TABLE) == (3.0, 5.0)      # 33 itself is mid, not low
    assert target_length(66.0, TABLE) == (3.0, 5.0)      # 66 itself is mid, not high
    assert target_length(66.1, TABLE) == (1.5, 2.5)
    assert target_length(90.0, TABLE) == (1.5, 2.5)


# -- retime: Step 1's three sections (low / mid / high) -----------------
# Five sections (energies 1, 2, 3, 8, 100 -> percentiles 10/30/50/70/90),
# so the section under test for "high" (70th percentile) is *not* the act's
# actual swell (the 100-energy section) -- otherwise testing "high" would
# accidentally also trigger the burst, which is a separate rule tested on
# its own below. A dense 0.5s beat grid is used for low/mid, and a
# deliberately sparser, off-grid downbeat list for high, so a beat landing
# that used `beats` instead of `downbeats` would produce a different,
# easily distinguished number.

SECTIONS = [
    {"t_in": 0.0, "t_out": 20.0, "energy": 1.0},     # low test section (pct 10)
    {"t_in": 20.0, "t_out": 40.0, "energy": 2.0},     # filler, between low and mid
    {"t_in": 40.0, "t_out": 60.0, "energy": 3.0},     # mid test section (pct 50)
    {"t_in": 60.0, "t_out": 80.0, "energy": 8.0},     # high test section (pct 70)
    {"t_in": 80.0, "t_out": 100.0, "energy": 100.0},  # the act's swell -- not queried directly here
]
BEATS = [round(i * 0.5, 1) for i in range(240)]           # 0, 0.5, 1.0 ... 119.5
DOWNBEATS = [60.0, 62.2, 64.4, 66.6]                       # off the 0.5 grid on purpose


def _retime_one(t_in, *, sections=SECTIONS, beats=BEATS, downbeats=DOWNBEATS,
                shots=None, **kw):
    slot = _slot("s1", t_in)
    shots = {"s1": _shot()} if shots is None else shots
    kw.setdefault("table", TABLE)
    kw.setdefault("burst_slots", [8, 12])
    kw.setdefault("is_act4", False)
    kw.setdefault("held_shot_s", [6.0, 10.0])
    kw.setdefault("silence_t", None)
    return retime([slot], sections=sections, beats=beats, downbeats=downbeats,
                 shots=shots, **kw)


def test_low_section_gets_a_length_in_the_low_band_ending_on_a_beat():
    out = _retime_one(0.0)
    length = out[0]["t_out"] - out[0]["t_in"]
    assert 5.0 <= length <= 8.0
    assert out[0]["t_out"] in BEATS


def test_mid_section_gets_a_length_in_the_mid_band_ending_on_a_beat():
    out = _retime_one(40.0)
    length = out[0]["t_out"] - out[0]["t_in"]
    assert 3.0 <= length <= 5.0
    assert out[0]["t_out"] in BEATS


def test_high_section_gets_a_length_in_the_high_band_ending_on_a_downbeat():
    out = _retime_one(60.0)
    length = out[0]["t_out"] - out[0]["t_in"]
    assert 1.5 <= length <= 2.5
    # 62.2 is the nearest *downbeat* to the unsnapped 62.0; the nearest plain
    # beat would have been 62.0 itself -- this is the number only the
    # downbeat grid produces, proving high energy chose that grid.
    assert out[0]["t_out"] == 62.2


def test_retimed_slot_keeps_every_key_it_arrived_with():
    out = _retime_one(0.0)
    assert set(out[0].keys()) == set(SLOT_KEYS) - {"slot_index"}


# -- retime: locked slots -------------------------------------------------

def test_locked_slot_keeps_its_own_bounds():
    locked = _slot("bridge", 5.0, t_out=95.0, locked=1)
    out = retime([locked], sections=SECTIONS, beats=BEATS, downbeats=DOWNBEATS,
                table=TABLE, burst_slots=[8, 12], is_act4=False,
                held_shot_s=[6.0, 10.0], silence_t=None, shots={})
    assert out[0]["t_in"] == 5.0
    assert out[0]["t_out"] == 95.0


def test_walk_resumes_from_a_locked_slots_t_out():
    locked = _slot("bridge", 0.0, t_out=20.0, locked=1)      # spans the whole low section
    unlocked = _slot("s1", 999.0)                            # t_in is irrelevant; the walk overrides it
    out = retime([locked, unlocked], sections=SECTIONS, beats=BEATS, downbeats=DOWNBEATS,
                table=TABLE, burst_slots=[8, 12], is_act4=False,
                held_shot_s=[6.0, 10.0], silence_t=None, shots={"s1": _shot()})
    assert out[1]["t_in"] == 20.0                            # continues from the locked slot's t_out


# -- retime: the burst on the act's highest swell ------------------------

def test_swell_section_yields_a_run_of_one_beat_slots():
    sections = [{"t_in": 0.0, "t_out": 10.0, "energy": 1.0},
               {"t_in": 10.0, "t_out": 1000.0, "energy": 100.0}]
    beats = [float(i) for i in range(0, 1000)]               # one beat per second
    slots = [_slot(f"s{i}", None) for i in range(4)]
    shots = {f"s{i}": _shot(end_s=10000.0) for i in range(4)}
    slots[0]["t_in"] = 10.0                                  # first slot starts inside the swell
    out = retime(slots, sections=sections, beats=beats, downbeats=[],
                table=TABLE, burst_slots=[8, 12], is_act4=False,
                held_shot_s=[6.0, 10.0], silence_t=None, shots=shots)
    assert len(out) == 4
    lengths = [round(s["t_out"] - s["t_in"], 3) for s in out]
    assert lengths == [1.0, 1.0, 1.0, 1.0]


def test_burst_capped_by_burst_slots_then_reverts_to_normal_banding():
    # burst_slots=[2,2] -> exactly 2 burst slots, even though 3 unlocked
    # slots fall inside the swell; the third must fall back to normal
    # section-banded timing, not a third one-beat cut.
    sections = [{"t_in": 0.0, "t_out": 1000.0, "energy": 1.0}]
    beats = [float(i) for i in range(0, 1000)]
    slots = [_slot(f"s{i}", None) for i in range(3)]
    slots[0]["t_in"] = 0.0
    shots = {f"s{i}": _shot(end_s=10000.0) for i in range(3)}
    out = retime(slots, sections=sections, beats=beats, downbeats=[],
                table=TABLE, burst_slots=[2, 2], is_act4=False,
                held_shot_s=[6.0, 10.0], silence_t=None, shots=shots)
    assert round(out[0]["t_out"] - out[0]["t_in"], 3) == 1.0
    assert round(out[1]["t_out"] - out[1]["t_in"], 3) == 1.0
    third_length = out[2]["t_out"] - out[2]["t_in"]
    assert 3.0 <= third_length <= 5.0                        # back to the mid band (pct 50 here)


def test_burst_with_no_beats_falls_back_to_normal_banding_without_crashing():
    sections = [{"t_in": 0.0, "t_out": 1000.0, "energy": 1.0}]
    slots = [_slot("s0", 0.0)]
    out = retime(slots, sections=sections, beats=[], downbeats=[],
                table=TABLE, burst_slots=[8, 12], is_act4=False,
                held_shot_s=[6.0, 10.0], silence_t=None, shots={"s0": _shot()})
    length = out[0]["t_out"] - out[0]["t_in"]
    assert 3.0 <= length <= 5.0


# -- retime: a slot must never claim more footage than its shot has -----

def test_a_shot_with_2s_of_material_never_gets_5():
    # mid band midpoint is 4.0s; with no beats to snap to, the clamp is the
    # only thing standing between the nominal length and the output.
    out = _retime_one(40.0, beats=[], downbeats=[], shots={"s1": _shot(end_s=2.0)})
    assert out[0]["t_out"] - out[0]["t_in"] == 2.0


def test_shot_missing_from_shots_keeps_its_nominal_length():
    out = _retime_one(40.0, beats=[], downbeats=[], shots={})
    assert out[0]["t_out"] - out[0]["t_in"] == 4.0           # mid band's midpoint, unclamped


# -- retime: dropping slots that fall off the end of the act -------------

def test_slot_at_or_past_act_end_is_dropped():
    out = _retime_one(100.0)                                 # sections[-1]["t_out"] == 100.0
    assert out == []


def test_every_slot_locked_passes_through_unchanged():
    a = _slot("a", 0.0, t_out=10.0, locked=1)
    b = _slot("b", 10.0, t_out=20.0, locked=1)
    out = retime([a, b], sections=SECTIONS, beats=BEATS, downbeats=DOWNBEATS,
                table=TABLE, burst_slots=[8, 12], is_act4=False,
                held_shot_s=[6.0, 10.0], silence_t=None, shots={})
    assert out == [a, b]


# -- retime: Act 4's held shot --------------------------------------------

def test_act4_held_shot_reaches_silence_t_when_the_shot_has_room():
    out = _retime_one(0.0, sections=[{"t_in": 0.0, "t_out": 100.0, "energy": 1.0}],
                      beats=[], downbeats=[], is_act4=True, silence_t=50.0,
                      shots={"s1": _shot(end_s=1000.0)})
    assert out[-1]["t_out"] == 50.0


def test_act4_held_shot_falls_short_when_the_shot_cannot_reach_silence_t():
    # Only 3s of footage left; held_shot_s's 8.0s midpoint and the 50.0s
    # reach to silence both exceed it, so the clamp -- not either of those
    # numbers -- decides the actual end.
    out = _retime_one(0.0, sections=[{"t_in": 0.0, "t_out": 100.0, "energy": 1.0}],
                      beats=[], downbeats=[], is_act4=True, silence_t=50.0,
                      shots={"s1": _shot(end_s=3.0)})
    assert out[-1]["t_out"] == 3.0


def test_act4_silence_before_every_slot_leaves_slots_untouched():
    out = _retime_one(20.0, sections=[{"t_in": 0.0, "t_out": 100.0, "energy": 1.0}],
                      beats=[], downbeats=[], is_act4=True, silence_t=5.0,
                      shots={"s1": _shot(end_s=1000.0)})
    assert out[0]["t_out"] == 24.0                           # mid band's midpoint, un-held
