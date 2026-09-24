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


def test_high_section_downbeat_past_avail_keeps_the_clamped_unsnapped_end():
    # Every downbeat below is well past t_in + avail (1.0s), so none of them
    # is a candidate at all -- this exercises the `b <= ceiling` filter
    # itself, which the other high-band test never binds (its avail is
    # effectively unlimited).
    out = _retime_one(60.0, downbeats=[65.0, 70.0], shots={"s1": _shot(end_s=1.0)})
    assert out[0]["t_out"] == 61.0                           # t_in(60) + the clamped 1.0s, unsnapped


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


def test_burst_with_no_beats_takes_the_high_band_floor_and_still_spends_budget():
    # burst_slots=[1,1] -> a budget of exactly 1. The grid is empty, so the
    # first slot can't find a beat at all; it still takes the high band's
    # floor (1.5s) and still spends the one burst slot the budget has --
    # the second slot must fall back to ordinary (mid) section banding,
    # not a second high-floor cut.
    sections = [{"t_in": 0.0, "t_out": 1000.0, "energy": 1.0}]
    slots = [_slot("s0", 0.0), _slot("s1", None)]
    shots = {"s0": _shot(end_s=1000.0), "s1": _shot(end_s=1000.0)}
    out = retime(slots, sections=sections, beats=[], downbeats=[],
                table=TABLE, burst_slots=[1, 1], is_act4=False,
                held_shot_s=[6.0, 10.0], silence_t=None, shots=shots)
    assert out[0]["t_out"] - out[0]["t_in"] == 1.5
    second_length = out[1]["t_out"] - out[1]["t_in"]
    assert 3.0 <= second_length <= 5.0


def test_burst_slot_whose_next_beat_would_overrun_the_footage_does_not_take_it():
    # The next beat after 0.0 is 1.0, but only 0.5s of footage is left --
    # _burst_end must refuse it rather than over-claim, falling back to the
    # high band's floor clamped by what's actually there.
    sections = [{"t_in": 0.0, "t_out": 1000.0, "energy": 1.0}]
    slots = [_slot("s0", 0.0)]
    out = retime(slots, sections=sections, beats=[0.0, 1.0, 100.0], downbeats=[],
                table=TABLE, burst_slots=[8, 12], is_act4=False,
                held_shot_s=[6.0, 10.0], silence_t=None, shots={"s0": _shot(end_s=0.5)})
    assert out[0]["t_out"] - out[0]["t_in"] == 0.5


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
    a_before, b_before = dict(a), dict(b)
    out = retime([a, b], sections=SECTIONS, beats=BEATS, downbeats=DOWNBEATS,
                table=TABLE, burst_slots=[8, 12], is_act4=False,
                held_shot_s=[6.0, 10.0], silence_t=None, shots={})
    assert out == [a, b]
    assert a == a_before and b == b_before        # retime must copy, never mutate its inputs


def test_first_unlocked_slot_with_no_t_in_starts_at_the_first_sections_t_in():
    # The sections are shifted off zero on purpose: an act after the first
    # does not begin at the film's zero, and against SECTIONS (which starts
    # at 0.0) this assertion would pass just as well for the last-resort
    # `t = 0.0`, proving nothing about reading the act's own first cue.
    sections = [dict(s, t_in=s["t_in"] + 30.0, t_out=s["t_out"] + 30.0) for s in SECTIONS]
    slot = _slot("s1", None)
    out = retime([slot], sections=sections, beats=[], downbeats=[],
                table=TABLE, burst_slots=[8, 12], is_act4=False,
                held_shot_s=[6.0, 10.0], silence_t=None, shots={"s1": _shot()})
    assert out[0]["t_in"] == 30.0 == sections[0]["t_in"]


def test_locked_slot_with_missing_bounds_raises_a_named_error():
    locked = _slot("bridge", None, t_out=None, locked=1)
    with pytest.raises(ValueError, match="bridge"):
        retime([locked], sections=SECTIONS, beats=[], downbeats=[],
              table=TABLE, burst_slots=[8, 12], is_act4=False,
              held_shot_s=[6.0, 10.0], silence_t=None, shots={})


def test_slot_with_zero_available_footage_is_dropped_not_stalled():
    out = _retime_one(0.0, beats=[], downbeats=[], shots={"s1": _shot(end_s=0.0)})
    assert out == []


def test_no_sections_at_all_does_not_crash_and_falls_back_to_the_low_band():
    # No section to read energy from -> percentile 0.0 -> the low band.
    out = _retime_one(0.0, sections=[], beats=[0.0, 1.0, 2.0, 3.0], downbeats=[])
    assert out[0]["t_out"] == 3.0        # low band's 6.5s midpoint, snapped to the nearest of a sparse grid


# -- retime: the clamp is tighter than shot_available_s when src_in offsets
# into the shot, or a split slot's own recorded overlap is tighter still --

def test_offset_src_in_tightens_the_clamp_below_shot_available_s():
    # shot_available_s(shot) is end_s - start_s = 10.0, but this slot's own
    # src_in already sits 8s into the shot -- pairs.py does exactly this for
    # a split aligned to the later of two starts -- so only 2s is actually
    # left; using the wider shot_available_s alone would over-claim the
    # mid band's 4.0s midpoint.
    slot = _slot("s1", 40.0, src_in=8.0)
    shot = _shot(start_s=0.0, end_s=10.0)
    out = retime([slot], sections=SECTIONS, beats=[], downbeats=[],
                table=TABLE, burst_slots=[8, 12], is_act4=False,
                held_shot_s=[6.0, 10.0], silence_t=None, shots={"s1": shot})
    assert out[0]["t_out"] - out[0]["t_in"] == 2.0
    assert out[0]["src_out"] == 10.0                          # src_in + the actual clamped length


def test_split_slot_never_outlasts_the_second_cameras_own_footage():
    # the primary runs on for far longer, but the other camera has only
    # 1.5s left from where the two were aligned, and a split may not use
    # more than either camera actually has at the aligned instant.
    slot = _slot("s1", 40.0, src_in=5.0, src_out=6.5, secondary_shot_id="s2",
                secondary_src_in=1.0)
    out = retime([slot], sections=SECTIONS, beats=[], downbeats=[],
                table=TABLE, burst_slots=[8, 12], is_act4=False,
                held_shot_s=[6.0, 10.0], silence_t=None,
                shots={"s1": _shot(start_s=0.0, end_s=1000.0),
                       "s2": _shot(start_s=0.0, end_s=2.5)})
    assert out[0]["t_out"] - out[0]["t_in"] == 1.5
    assert out[0]["src_out"] == 6.5


def test_a_split_does_not_shrink_when_the_act_is_re_timed_again():
    """A split stays unlocked, so every refill round re-times it. Reading
    its ceiling from the src_out the previous walk wrote made each round's
    result the next round's ceiling: a ratchet that could only ever
    shorten. The material is what bounds it, and the material does not
    change between rounds."""
    shots = {"s1": _shot(start_s=0.0, end_s=1000.0), "s2": _shot(start_s=0.0, end_s=1000.0)}
    kw = dict(sections=SECTIONS, beats=[], downbeats=[], table=TABLE, burst_slots=[8, 12],
              is_act4=False, held_shot_s=[6.0, 10.0], silence_t=None, shots=shots)
    # a pair whose overlap the previous round already cut down to 1.5s
    slot = _slot("s1", 40.0, src_in=5.0, src_out=6.5, secondary_shot_id="s2",
                secondary_src_in=1.0)
    first = retime([slot], **kw)
    second = retime(first, **kw)
    assert first[0]["t_out"] - first[0]["t_in"] == 4.0    # the mid band's midpoint, unclamped
    assert (second[0]["t_out"] - second[0]["t_in"]) == (first[0]["t_out"] - first[0]["t_in"])


# -- retime: Act 4's held shot --------------------------------------------

def test_act4_held_shot_holds_for_the_configured_midpoint_when_the_shot_has_room():
    # held_shot_s's own midpoint (8.0s) drives the hold, not the 50.0s
    # distance back to the silence -- otherwise held_shot_s would be dead
    # weight in every act whose last shot has room to spare.
    out = _retime_one(0.0, sections=[{"t_in": 0.0, "t_out": 100.0, "energy": 1.0}],
                      beats=[], downbeats=[], is_act4=True, silence_t=50.0,
                      shots={"s1": _shot(end_s=1000.0)})
    assert out[-1]["t_out"] == 50.0
    assert out[-1]["t_in"] == 42.0
    assert out[-1]["t_out"] - out[-1]["t_in"] == 8.0


def test_act4_held_shot_shrinks_but_still_ends_on_the_silence_when_the_shot_is_short():
    # Only 3s of footage left; the hold shrinks to 3s but t_out still lands
    # exactly on the silence -- t_in moves forward to meet it, rather than
    # t_out falling short of it.
    out = _retime_one(0.0, sections=[{"t_in": 0.0, "t_out": 100.0, "energy": 1.0}],
                      beats=[], downbeats=[], is_act4=True, silence_t=50.0,
                      shots={"s1": _shot(end_s=3.0)})
    assert out[-1]["t_out"] == 50.0
    assert out[-1]["t_in"] == 47.0
    assert out[-1]["t_out"] - out[-1]["t_in"] == 3.0


def test_act4_held_shot_shrinks_rather_than_overlapping_the_previous_slot():
    # The shot has room for the full 8s midpoint, which would start the hold
    # at 42.0 -- but the slot before it runs to 45.0. The hold gives up its
    # front instead: t_out still lands exactly on the silence, t_in meets the
    # previous slot's end. Without the shrink the two slots overlap by 3s,
    # and every act length measured off the timeline is that much wrong.
    # (A dummy high-energy section elsewhere, never queried, keeps the tested
    # section at percentile 50 rather than letting it become its own swell.)
    sections = [{"t_in": 0.0, "t_out": 100.0, "energy": 2.0},
               {"t_in": 100.0, "t_out": 150.0, "energy": 1.0},
               {"t_in": 150.0, "t_out": 200.0, "energy": 100.0}]
    slots = [_slot("s0", 41.0), _slot("s1", None)]
    shots = {"s0": _shot(end_s=1000.0), "s1": _shot(end_s=1000.0)}
    out = retime(slots, sections=sections, beats=[], downbeats=[],
                table=TABLE, burst_slots=[8, 12], is_act4=True,
                held_shot_s=[6.0, 10.0], silence_t=50.0, shots=shots)
    assert out[0]["t_out"] == 45.0                           # mid band's 4.0s from 41.0
    assert out[1]["t_in"] == 45.0                            # not the unshrunk 42.0
    assert out[1]["t_out"] == 50.0                           # still exactly on the silence
    assert out[1]["src_out"] == 5.0                          # src_in 0.0 + the hold it got


def test_act4_silence_before_every_slot_leaves_slots_untouched():
    # A single section would be its own swell (max of one is itself) and
    # trigger the burst instead of plain mid-band timing; a second, much
    # louder dummy section elsewhere keeps this section's percentile at 50
    # (mid) without ever being queried itself.
    sections = [{"t_in": 0.0, "t_out": 100.0, "energy": 2.0},
               {"t_in": 100.0, "t_out": 150.0, "energy": 1.0},
               {"t_in": 150.0, "t_out": 200.0, "energy": 100.0}]
    out = _retime_one(20.0, sections=sections,
                      beats=[], downbeats=[], is_act4=True, silence_t=5.0,
                      shots={"s1": _shot(end_s=1000.0)})
    assert out[0]["t_out"] == 24.0                           # mid band's midpoint, un-held


def test_act4_held_shot_never_taken_when_a_locked_slot_sits_before_the_silence():
    # The locked slot sits between the unlocked slot and the silence, so no
    # slot qualifies for the hold at all: the locked slot keeps its own
    # bounds, the unlocked slot keeps its ordinary (un-held) length, and
    # nothing stretches over the locked slot's span. A dummy high-energy
    # section elsewhere (never queried) keeps the tested section's own
    # percentile at 50 (mid) rather than letting it become its own swell.
    sections = [{"t_in": 0.0, "t_out": 50.0, "energy": 2.0},
               {"t_in": 50.0, "t_out": 60.0, "energy": 1.0},
               {"t_in": 60.0, "t_out": 100.0, "energy": 100.0}]
    unlocked = _slot("s1", 0.0)
    locked = _slot("bridge", 10.0, t_out=45.0, locked=1)
    out = retime([unlocked, locked], sections=sections, beats=[], downbeats=[],
                table=TABLE, burst_slots=[8, 12], is_act4=True,
                held_shot_s=[6.0, 10.0], silence_t=50.0,
                shots={"s1": _shot(end_s=1000.0)})
    assert out[1]["t_in"] == 10.0 and out[1]["t_out"] == 45.0
    assert out[0]["t_out"] == 4.0                            # mid band's own midpoint, un-held
    assert out[0]["t_out"] <= out[1]["t_in"]                 # no overlap


# -- sections with no music under them (Gate 3, 2026-09-24) ------------

def test_a_section_with_a_band_takes_it_instead_of_its_energy_rank():
    """Outside a music window there is no energy to rank, so the section
    names its band outright (music.placement.no_music_band). Its own
    percentile is whatever it is; the band is the answer."""
    assert target_length(99.0, TABLE) == (1.5, 2.5)
    assert target_length(99.0, TABLE, "low") == (5.0, 8.0)


def test_a_bandless_stretch_does_not_re_rank_the_music_around_it():
    """The ranking is relative: letting several no-music sections in at
    energy 0.0 would push every real cue up a band and shorten every cut
    under music that is actually playing."""
    music = [{"energy": 1.0}, {"energy": 2.0}, {"energy": 3.0}]
    assert energy_percentiles(music) == [pytest.approx(16.67, abs=0.01),
                                         pytest.approx(50.0), pytest.approx(83.33, abs=0.01)]
    with_silence = [{"energy": None, "band": "low"}] * 6 + music
    assert energy_percentiles(with_silence)[-3:] == energy_percentiles(music)


def test_slots_outside_a_window_are_cut_to_the_configured_band():
    """The film's design keeps running without music: the low band's long
    holds, not whatever the act's loudest cue would have asked for."""
    sections = [{"t_in": 0.0, "t_out": 20.0, "energy": 9.0},              # loud music
                {"t_in": 20.0, "t_out": 60.0, "energy": None, "band": "low"}]
    slots = [_slot(f"s{i}", None) for i in range(8)]
    slots[0]["t_in"] = 0.0
    out = retime(slots, sections=sections, beats=[], downbeats=[], table=TABLE,
                 burst_slots=(0, 0), is_act4=False, held_shot_s=[6.0, 10.0], silence_t=None,
                 shots={f"s{i}": _shot() for i in range(8)})
    lengths = [round(s["t_out"] - s["t_in"], 3) for s in out]
    under_music = [l for s, l in zip(out, lengths) if s["t_in"] < 20.0]
    under_none = [l for s, l in zip(out, lengths) if s["t_in"] >= 20.0]
    assert under_music and all(l == 2.0 for l in under_music), "the high band's midpoint"
    assert under_none and all(l == 6.5 for l in under_none), "the low band's midpoint"


def test_the_burst_never_fires_on_a_stretch_with_no_music():
    """The burst is the act's loudest swell; silence has no swell to be."""
    sections = [{"t_in": 0.0, "t_out": 60.0, "energy": None, "band": "low"}]
    slots = [_slot(f"s{i}", None) for i in range(5)]
    slots[0]["t_in"] = 0.0
    out = retime(slots, sections=sections, beats=[i * 0.5 for i in range(200)],
                 downbeats=[i * 2.0 for i in range(50)], table=TABLE,
                 burst_slots=(8, 12), is_act4=False, held_shot_s=[6.0, 10.0], silence_t=None,
                 shots={f"s{i}": _shot() for i in range(5)})
    assert all(round(s["t_out"] - s["t_in"], 3) >= 5.0 for s in out), \
        "a one-beat run here would be the fast cutting of music that is not playing"
