"""Anchors from the beat sheet (Film v2 step 4, task 3) -- pure, no database."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from datetime import datetime

from nepal.process.assemble import SLOT_KEYS
from nepal.story.anchors import (Anchor, broll_candidates, cold_open_pick,
                                  place_anchors, quote_anchors, speech_anchors)


def _ts(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


def _shot(shot_id, recording_id, start_utc, *, start_s=0.0, end_s=60.0, act=3,
         face_score=0.5, media_kind="video", score_total=0.5):
    return {"shot_id": shot_id, "recording_id": recording_id, "media_kind": media_kind,
            "start_s": start_s, "end_s": end_s, "start_utc": start_utc, "act": act,
            "face_score": face_score, "score_total": score_total}


def _beat(beat_id, *, kind="speech", act=3, shot_id=None, msg_id=None, src_in=None,
         src_out=None, text="", levity=False, effect="none", rank=1, ts_utc=None):
    b = {"beat_id": beat_id, "kind": kind, "act": act, "shot_id": shot_id, "msg_id": msg_id,
         "src_in": src_in, "src_out": src_out, "text": text, "levity": levity,
         "effect": effect, "rank": rank, "rationale": ""}
    if ts_utc is not None:
        b["ts_utc"] = ts_utc                        # quote/closing: joined in by the caller
    return b


SHOTS = {
    "s1": _shot("s1", "r1", "2024-05-03T04:00:00+00:00", face_score=0.6),
    "s2": _shot("s2", "r1", "2024-05-03T04:10:00+00:00", face_score=0.2),   # own_picture
    "s3": _shot("s3", "r2", "2024-05-03T04:20:00+00:00", face_score=0.5),
}

BEATS = [
    _beat("b1", shot_id="s1", src_in=10.0, src_out=15.0, rank=2, text="one"),
    _beat("b2", shot_id="s2", src_in=5.0, src_out=8.0, rank=1, text="two"),
    _beat("b3", shot_id="s3", src_in=0.2, src_out=5.0, rank=3, text="three"),
]
QUOTE = _beat("q1", kind="quote", act=1, msg_id="m1", text="quote text", rank=4,
              ts_utc="2024-02-11T19:02:00+00:00")


# -- speech_anchors ------------------------------------------------------

def test_speech_anchors_preroll_utc_and_own_picture():
    anchors = speech_anchors(BEATS, SHOTS, face_hold_s=2.5, pre_roll_s=0.4,
                             own_picture_below=0.35)
    by_id = {a.beat_id: a for a in anchors}
    assert set(by_id) == {"b1", "b2", "b3"}

    a1 = by_id["b1"]                                # pre-rolled, plenty of room before it
    assert a1.src_in == 9.6 and a1.src_out == 15.0 and abs(a1.duration_s - 5.4) < 1e-9
    assert a1.utc == "2024-05-03T04:00:10+00:00"
    assert a1.own_picture is False and a1.face_hold_s == 2.5     # min(2.5, 5.4)
    assert a1.recording_id == "r1" and a1.t_in == -1.0

    a2 = by_id["b2"]                                # face_score 0.2 < 0.35 -> own_picture
    assert a2.own_picture is True and a2.face_hold_s == 0.0
    assert a2.src_in == 4.6

    a3 = by_id["b3"]                                # pre-roll clamped at shot.start_s
    assert a3.src_in == 0.0
    assert a3.utc == "2024-05-03T04:20:00.200000+00:00"


def test_speech_anchors_survive_a_malformed_or_missing_start_utc():
    shots = {
        "bad": _shot("bad", "r1", "not a date"),
        "missing": _shot("missing", "r1", None),
        "good": _shot("good", "r1", "2024-05-03T04:00:00+00:00"),
    }
    beats = [
        _beat("b_bad", shot_id="bad", src_in=1.0, src_out=2.0),
        _beat("b_missing", shot_id="missing", src_in=1.0, src_out=2.0),
        _beat("b_good", shot_id="good", src_in=1.0, src_out=2.0),
    ]
    anchors = speech_anchors(beats, shots, face_hold_s=2.5, pre_roll_s=0.4,
                             own_picture_below=0.35)
    by_id = {a.beat_id: a for a in anchors}
    assert set(by_id) == {"b_bad", "b_missing", "b_good"}    # the batch survived
    assert by_id["b_bad"].utc is None
    assert by_id["b_missing"].utc is None
    assert by_id["b_good"].utc == "2024-05-03T04:00:01+00:00"   # the rest is unaffected
    assert by_id["b_bad"].src_in == 0.6 and by_id["b_bad"].duration_s == 1.4


def test_quote_anchors_have_no_picture_and_zero_duration():
    anchors = quote_anchors([QUOTE])
    assert len(anchors) == 1
    a = anchors[0]
    assert a.kind == "quote" and a.recording_id is None and a.shot_id is None
    assert a.duration_s == 0.0 and a.src_in == 0.0 and a.src_out == 0.0
    assert a.utc == "2024-02-11T19:02:00+00:00"


def test_quote_anchors_ignore_other_kinds():
    speech_only = _beat("b1", shot_id="s1", src_in=0.0, src_out=1.0)
    assert quote_anchors([speech_only]) == []


# -- place_anchors ---------------------------------------------------------

def test_place_anchors_keeps_order_respects_the_gap_and_ends_in_the_act():
    anchors = speech_anchors(BEATS, SHOTS, face_hold_s=2.5, pre_roll_s=0.4,
                             own_picture_below=0.35)
    placed = place_anchors(anchors, act_t0=0.0, act_len_s=100.0,
                           act_utc0=_ts("2024-05-03T04:00:00+00:00"),
                           act_utc1=_ts("2024-05-03T04:30:00+00:00"))
    assert [a.beat_id for a in placed] == ["b1", "b2", "b3"]     # chronological order kept
    for prev, nxt in zip(placed, placed[1:]):
        assert nxt.t_in >= prev.t_in + prev.duration_s + 4.0 - 1e-9   # the gap held
    last = placed[-1]
    assert last.t_in + last.duration_s <= 100.0 + 1e-9           # ends inside the act
    assert all(a.t_in >= 0.0 for a in placed)


def test_place_anchors_crowds_against_act_t0_never_before_it():
    """Two anchors that cannot both fit with the gap: the act's *end* still
    holds (the next act counts from there); the crowd gives at the start."""
    anchors = [
        Anchor(beat_id="x1", act=3, kind="speech", utc="2024-05-03T04:00:01+00:00",
              recording_id="r1", shot_id="s1", src_in=0.0, src_out=6.0, duration_s=6.0,
              own_picture=False, face_hold_s=2.5, text="x1", effect="none"),
        Anchor(beat_id="x2", act=3, kind="speech", utc="2024-05-03T04:00:02+00:00",
              recording_id="r1", shot_id="s2", src_in=0.0, src_out=6.0, duration_s=6.0,
              own_picture=False, face_hold_s=2.5, text="x2", effect="none"),
    ]
    placed = place_anchors(anchors, act_t0=0.0, act_len_s=8.0,
                           act_utc0=_ts("2024-05-03T04:00:00+00:00"),
                           act_utc1=_ts("2024-05-03T04:00:10+00:00"))
    assert [a.beat_id for a in placed] == ["x1", "x2"]
    assert placed[0].t_in == 0.0                                 # never before act_t0
    assert placed[-1].t_in + placed[-1].duration_s <= 8.0 + 1e-9


def test_place_anchors_spreads_evenly_in_input_order_with_no_utc_at_all():
    anchors = [
        Anchor(beat_id=f"y{i}", act=3, kind="speech", utc=None, recording_id="r1",
              shot_id=f"s{i}", src_in=0.0, src_out=5.0, duration_s=5.0, own_picture=False,
              face_hold_s=2.5, text=f"y{i}", effect="none")
        for i in (1, 2, 3)
    ]
    placed = place_anchors(anchors, act_t0=0.0, act_len_s=90.0, act_utc0=None, act_utc1=None)
    assert [a.beat_id for a in placed] == ["y1", "y2", "y3"]     # input order kept
    assert [round(a.t_in, 3) for a in placed] == [15.0, 45.0, 75.0]   # spread evenly
    for prev, nxt in zip(placed, placed[1:]):
        assert nxt.t_in >= prev.t_in + prev.duration_s + 4.0 - 1e-9   # the gap held
    assert placed[-1].t_in + placed[-1].duration_s <= 90.0 + 1e-9


# -- broll_candidates --------------------------------------------------

def test_broll_candidates_excludes_own_recording_other_act_and_photos():
    anchor = speech_anchors(BEATS, SHOTS, face_hold_s=2.5, pre_roll_s=0.4,
                            own_picture_below=0.35)[0]        # b1, recording r1, act 3
    pool = [
        _shot("in_window_low", "r2", "2024-05-03T04:00:30+00:00", act=3, score_total=0.7),
        _shot("in_window_high", "r5", "2024-05-03T04:00:05+00:00", act=3, score_total=0.9),
        _shot("wrong_act", "r3", "2024-05-03T04:00:15+00:00", act=4, score_total=1.0),
        _shot("same_recording", "r1", "2024-05-03T04:00:12+00:00", act=3, score_total=1.0),
        _shot("a_photo", "r2", "2024-05-03T04:00:11+00:00", act=3, score_total=1.0,
             media_kind="photo"),
        _shot("too_far", "r4", "2024-05-03T05:10:00+00:00", act=3, score_total=1.0),
    ]
    got = broll_candidates(anchor, pool, window_s=60)
    assert [s["shot_id"] for s in got] == ["in_window_high", "in_window_low"]   # by score


# -- cold_open_pick ------------------------------------------------------

def test_cold_open_picks_the_lowest_rank_in_act_3_or_4_and_extends_it():
    slot = cold_open_pick(BEATS, SHOTS, length_range=[15.0, 25.0])
    assert slot["beat_id"] == "b2"                    # rank 1, act 3 -- the winner
    assert slot["kind"] == "video" and slot["locked"] == 1
    assert slot["transition"] == "dip_black" and slot["speed"] == 1.0
    assert slot["act"] == 3 and slot["shot_id"] == "s2"
    assert slot["t_in"] is None and slot["t_out"] is None
    length = slot["src_out"] - slot["src_in"]
    assert 15.0 <= length <= 25.0
    assert 0.0 <= slot["src_in"] and slot["src_out"] <= 60.0    # inside the shot's bounds
    assert set(slot) == set(SLOT_KEYS) - {"slot_index"}
    for k in ("secondary_shot_id", "secondary_src_in", "motion", "scene_id", "msg_id", "yaw"):
        assert slot[k] is None


def test_cold_open_gives_the_whole_shot_when_it_holds_less_than_the_minimum():
    short_shot = {"s_short": _shot("s_short", "r9", "2024-05-03T04:00:00+00:00",
                                    start_s=0.0, end_s=10.0)}
    beat = _beat("b_short", shot_id="s_short", src_in=4.0, src_out=6.0, rank=1)
    slot = cold_open_pick([beat], short_shot, length_range=[15.0, 25.0])
    assert slot["src_in"] == 0.0 and slot["src_out"] == 10.0


def test_cold_open_is_none_without_an_act_3_or_4_speech_beat():
    early = _beat("b_early", act=2, shot_id="s1", src_in=0.0, src_out=1.0, rank=1)
    assert cold_open_pick([early], SHOTS, length_range=[15.0, 25.0]) is None
