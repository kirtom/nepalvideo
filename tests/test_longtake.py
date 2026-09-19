"""The bridge crossing as one locked slot (Film v2 step 4, task 5) -- pure,
no database. See src/nepal/process/longtake.py for what is and isn't ranked
here (the DEM drop needs step 5/6's crossing point)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from nepal.process.assemble import SLOT_KEYS
from nepal.process.longtake import bridge_candidates, long_take_slot

KEYWORDS = ["мост", "мосту", "моста", "bridge", "suspension"]


def _shot(shot_id, recording_id, *, act=3, start_s=0.0, end_s=10.0,
         transcript=None, caption=None, has_face=0):
    return {"shot_id": shot_id, "recording_id": recording_id, "act": act,
            "start_s": start_s, "end_s": end_s, "transcript": transcript,
            "caption": caption, "alt_dem_m": None, "has_face": has_face}


def _recording(recording_id, duration_s, *, is_360=0):
    return {"recording_id": recording_id, "duration_s": duration_s, "is_360": is_360}


def test_keyword_recording_beats_longer_silent_one():
    shots = [
        _shot("s1", "rec_bridge", start_s=5.0, end_s=15.0, transcript="idem cherez bridge"),
        _shot("s2", "rec_silent", start_s=0.0, end_s=10.0),
    ]
    recordings = [
        _recording("rec_bridge", 70.0),
        _recording("rec_silent", 200.0),
    ]
    candidates = bridge_candidates(shots, recordings, keywords=KEYWORDS)
    assert len(candidates) == 1
    assert candidates[0]["recording_id"] == "rec_bridge"
    assert candidates[0]["first_shot_id"] == "s1"
    assert candidates[0]["act"] == 3


def test_cyrillic_keyword_matches_via_casefold():
    shots = [_shot("s1", "rec1", transcript="Мы стоим на МОСТУ над рекой")]
    recordings = [_recording("rec1", 100.0)]
    candidates = bridge_candidates(shots, recordings, keywords=KEYWORDS)
    assert len(candidates) == 1


def test_caption_also_matches_when_transcript_is_none():
    shots = [_shot("s1", "rec1", transcript=None, caption="Suspension bridge crossing")]
    recordings = [_recording("rec1", 100.0)]
    candidates = bridge_candidates(shots, recordings, keywords=KEYWORDS)
    assert len(candidates) == 1


def test_recording_with_no_keyword_shot_is_not_a_candidate():
    shots = [_shot("s1", "rec1", transcript="just walking")]
    recordings = [_recording("rec1", 100.0)]
    assert bridge_candidates(shots, recordings, keywords=KEYWORDS) == []


def test_first_shot_id_is_the_earliest_keyword_shot_by_start_s():
    shots = [
        _shot("late", "rec1", start_s=50.0, transcript="bridge"),
        _shot("early", "rec1", start_s=5.0, transcript="bridge"),
    ]
    recordings = [_recording("rec1", 100.0)]
    candidates = bridge_candidates(shots, recordings, keywords=KEYWORDS)
    assert candidates[0]["first_shot_id"] == "early"


def test_more_keyword_shots_breaks_a_duration_tie():
    # Both recordings cap at 120s (both are >= 120s long), so duration is
    # tied; the recording with two keyword shots outranks the one with one.
    shots = [
        _shot("a1", "recA", transcript="bridge"),
        _shot("b1", "recB", transcript="bridge"),
        _shot("b2", "recB", start_s=20.0, transcript="bridge"),
    ]
    recordings = [_recording("recA", 150.0), _recording("recB", 150.0)]
    candidates = bridge_candidates(shots, recordings, keywords=KEYWORDS)
    assert candidates[0]["recording_id"] == "recB"


def test_face_share_breaks_a_tie_the_lower_share_wins():
    # Same duration cap, same keyword-shot count; recA has no faces at all,
    # recB is all faces -- recA (lower face share) should rank first.
    shots = [
        _shot("a1", "recA", transcript="bridge", has_face=0),
        _shot("a2", "recA", start_s=20.0, has_face=0),
        _shot("b1", "recB", transcript="bridge", has_face=1),
        _shot("b2", "recB", start_s=20.0, has_face=1),
    ]
    recordings = [_recording("recA", 150.0), _recording("recB", 150.0)]
    candidates = bridge_candidates(shots, recordings, keywords=KEYWORDS)
    assert candidates[0]["recording_id"] == "recA"


def test_recording_with_no_duration_is_excluded_and_yields_no_slot():
    # A recording with duration_s: None can't be weighed against
    # length_range at all -- it must be dropped from candidates, and if it
    # somehow reached long_take_slot anyway, that must also return None
    # rather than crash on float(None).
    shots = [
        _shot("s1", "rec_no_duration", transcript="bridge"),
        _shot("s2", "rec_short", transcript="bridge"),
    ]
    recordings = [
        _recording("rec_no_duration", None),
        _recording("rec_short", 70.0),
    ]
    candidates = bridge_candidates(shots, recordings, keywords=KEYWORDS)
    assert [c["recording_id"] for c in candidates] == ["rec_short"]

    shots_by_recording = {"rec_no_duration": [shots[0]]}
    recordings_by_id = {"rec_no_duration": recordings[0]}
    candidate = {"recording_id": "rec_no_duration", "first_shot_id": "s1", "act": 3, "score": 0.0}
    assert long_take_slot(candidate, recordings_by_id, shots_by_recording,
                          length_range=[60, 90]) is None


def test_slot_shape_matches_slot_keys():
    shots_by_recording = {"rec1": [_shot("s1", "rec1", act=3, start_s=10.0, transcript="bridge")]}
    recordings_by_id = {"rec1": _recording("rec1", 300.0)}
    candidate = {"recording_id": "rec1", "first_shot_id": "s1", "act": 3, "score": (90.0, 1, 0.0)}
    slot = long_take_slot(candidate, recordings_by_id, shots_by_recording, length_range=[60, 90])
    assert slot["kind"] == "video"
    assert slot["shot_id"] == "s1"
    assert slot["src_in"] == 10.0
    assert slot["src_out"] == 100.0
    assert slot["act"] == 3
    assert slot["locked"] == 1
    assert slot["motion"] is None
    assert slot["speed"] == 1.0
    assert slot["transition"] == "cut"
    assert slot["t_in"] is None and slot["t_out"] is None
    assert set(SLOT_KEYS) - {"slot_index"} == set(slot.keys())


def test_40s_recording_yields_none_under_60_90():
    shots_by_recording = {"rec1": [_shot("s1", "rec1", start_s=0.0, transcript="bridge")]}
    recordings_by_id = {"rec1": _recording("rec1", 40.0)}
    candidate = {"recording_id": "rec1", "first_shot_id": "s1", "act": 3, "score": 0.0}
    assert long_take_slot(candidate, recordings_by_id, shots_by_recording,
                          length_range=[60, 90]) is None


def test_300s_recording_yields_90s_from_the_keyword_shot():
    shots_by_recording = {"rec1": [_shot("s1", "rec1", start_s=15.0, transcript="bridge")]}
    recordings_by_id = {"rec1": _recording("rec1", 300.0)}
    candidate = {"recording_id": "rec1", "first_shot_id": "s1", "act": 3, "score": 0.0}
    slot = long_take_slot(candidate, recordings_by_id, shots_by_recording, length_range=[60, 90])
    assert slot["src_in"] == 15.0
    assert slot["src_out"] == 105.0


def test_70s_remaining_under_60_90_gets_70():
    # Recording holds 70s after the keyword shot's start -- more than the 60s
    # floor but less than the 90s cap, so the slot takes what's actually there
    # rather than over-claiming footage the recording doesn't have.
    shots_by_recording = {"rec1": [_shot("s1", "rec1", start_s=30.0, transcript="bridge")]}
    recordings_by_id = {"rec1": _recording("rec1", 100.0)}
    candidate = {"recording_id": "rec1", "first_shot_id": "s1", "act": 3, "score": 0.0}
    slot = long_take_slot(candidate, recordings_by_id, shots_by_recording, length_range=[60, 90])
    assert slot["src_in"] == 30.0
    assert slot["src_out"] == 100.0
