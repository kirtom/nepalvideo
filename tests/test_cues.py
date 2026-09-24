"""Audio cues and overlay rows (film v2 step 4, Part B), with no media and
no database: slot dicts in, rows out. Every level rule the mixer will trust
is pinned here, because the mix is where a wrong number becomes inaudible
dialogue rather than a failing assertion."""
import json
import math
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from nepal.process import cues
from nepal.story.anchors import Anchor

LEVELS = dict(lufs_under_music=-28.0, lufs_full=-18.0, lufs_under_speech=-24.0)
FADES = dict(fade_s=0.15, window_fade_s=1.0)


def _slot(i, act, t_in, t_out, *, kind="video", rec=None, src_in=0.0, beat=None, secondary=None):
    return {"slot_index": i, "act": act, "t_in": t_in, "t_out": t_out, "kind": kind,
            "shot_id": f"{rec}#{i}" if rec else None, "recording_id": rec,
            "src_in": src_in, "src_out": src_in + (t_out - t_in) if kind == "video" else None,
            "beat_id": beat, "secondary_shot_id": secondary}


def _anchor(beat, rec, src_in, src_out, *, t_in, own=True):
    return Anchor(beat_id=beat, act=3, kind="speech", utc=None, recording_id=rec,
                  shot_id=f"{rec}#0", src_in=src_in, src_out=src_out,
                  duration_s=src_out - src_in, own_picture=own, face_hold_s=0.0,
                  text="...", effect="none", t_in=t_in)


# -- speech ---------------------------------------------------------------

def test_a_walking_anchor_gives_one_speech_cue_at_its_own_source_span():
    a = _anchor("b_walk", "w3", 1.6, 12.0, t_in=200.0)
    rows = cues.speech_cues([a], lufs=-16.0, fade_s=0.15)
    assert len(rows) == 1
    c = rows[0]
    assert c["cue_id"] == "sp_b_walk" and c["track"] == "speech" and c["beat_id"] == "b_walk"
    assert c["source"] == "w3" and (c["src_in"], c["src_out"]) == (1.6, 12.0)
    assert c["t_in"] == 200.0 and math.isclose(c["t_out"], 210.4)
    assert c["gain_lufs"] == -16.0 and c["fade_in_s"] == c["fade_out_s"] == 0.15
    assert set(c) == set(cues.CUE_KEYS)


def test_only_speech_anchors_become_speech_cues():
    pair = Anchor(beat_id="pair:x", act=2, kind="pair", utc=None, recording_id="p", shot_id="p#0",
                  src_in=0.0, src_out=5.0, duration_s=5.0, own_picture=True, face_hold_s=0.0,
                  text="", effect="none", t_in=10.0)
    assert cues.speech_cues([pair], lufs=-16.0, fade_s=0.15) == []


def test_a_beat_on_two_runs_gets_a_speech_cue_per_run():
    # the cold open teases the pass line at t = 0, its slot extended 2 s
    # before the anchor's pre-roll, and act 4 delivers it: two cues, the
    # same utterance, each starting where its run's first slot reaches it
    a = _anchor("b_pass", "c4", 10.6, 18.0, t_in=-1.0)
    slots = [_slot(0, 0, 0.0, 15.0, rec="c4", src_in=8.6, beat="b_pass"),
             _slot(1, 0, 15.0, 18.0, kind="card"),
             _slot(2, 1, 18.0, 30.0, rec="a"),
             _slot(3, 4, 300.0, 304.0, rec="c4", src_in=10.6, beat="b_pass"),
             _slot(4, 4, 304.0, 307.0, rec="b", beat="b_pass")]
    rows = cues.speech_cues(cues.place_speech([a], slots), lufs=-16.0, fade_s=0.15)
    assert [(r["cue_id"], r["t_in"], r["t_out"]) for r in rows] == [
        ("sp_b_pass", 2.0, 9.4), ("sp_b_pass_2", 300.0, 307.4)]
    assert all((r["source"], r["src_in"], r["src_out"], r["beat_id"]) == ("c4", 10.6, 18.0, "b_pass")
               for r in rows)
    assert cues.place_speech([_anchor("b_none", "x", 0.0, 5.0, t_in=-1.0)], slots) == []
    # a cold open that opens 0.4 s after the pre-roll trims its own cue and
    # not act 4's: each run is placed from the anchor, never from the run before
    slots[0] = _slot(0, 0, 0.0, 15.0, rec="c4", src_in=11.0, beat="b_pass")
    rows = cues.speech_cues(cues.place_speech([a], slots), lufs=-16.0, fade_s=0.15)
    assert [(r["cue_id"], r["t_in"], round(r["src_in"], 6), r["t_out"]) for r in rows] == [
        ("sp_b_pass", 0.0, 11.0, 7.0), ("sp_b_pass_2", 300.0, 10.6, 307.4)]
    # a second run that begins while the first cue is still speaking gets no cue
    early = [_slot(0, 0, 0.0, 4.0, rec="c4", src_in=8.6, beat="b_pass"), _slot(1, 0, 4.0, 5.0, kind="card"),
             _slot(2, 4, 5.0, 9.0, rec="c4", src_in=10.6, beat="b_pass")]
    assert [r["cue_id"] for r in cues.speech_cues(cues.place_speech([a], early), lufs=-16.0, fade_s=0.15)] == ["sp_b_pass"]


def test_a_slot_that_opens_after_the_pre_roll_starts_the_voice_with_the_picture():
    a = _anchor("b_long", "c4", 0.6, 20.0, t_in=-1.0)      # pre-rolled 0.4 s before the words at 1.0
    slots = [_slot(0, 0, 0.0, 19.0, rec="c4", src_in=1.0, beat="b_long")]
    [c] = cues.speech_cues(cues.place_speech([a], slots), lufs=-16.0, fade_s=0.15)
    assert (c["t_in"], c["src_in"], c["src_out"], c["t_out"]) == (0.0, 1.0, 20.0, 19.0)


def test_speech_spans_are_contiguous_runs_not_one_union_per_beat():
    # the cold open plays the pass line at 0 and act 4 plays it again: two
    # spans, never one that covers the whole film in between
    slots = [_slot(0, 0, 0.0, 20.0, rec="c4", beat="b_pass"),
             _slot(1, 0, 20.0, 23.0, kind="card"),
             _slot(2, 1, 23.0, 30.0, rec="a"),
             _slot(3, 4, 30.0, 34.0, rec="c4", beat="b_pass"),
             _slot(4, 4, 34.0, 37.0, rec="b", beat="b_pass"),
             _slot(5, 4, 37.0, 40.0, rec="c")]
    assert cues.speech_spans(slots) == [(0.0, 20.0), (30.0, 37.0)]


# -- location -------------------------------------------------------------

def test_a_photo_holds_the_previous_video_ambience_from_where_it_stopped():
    slots = [_slot(0, 2, 0.0, 5.0, rec="r1", src_in=10.0),
             _slot(1, 2, 5.0, 8.0, kind="photo"),
             _slot(2, 2, 8.0, 11.0, kind="card"),
             _slot(3, 2, 11.0, 15.0, rec="r2", src_in=3.0)]
    rows = cues.location_cues(slots, speech_spans=[], windows=[], silence={}, **LEVELS, **FADES)
    by_id = {r["cue_id"]: r for r in rows}
    assert list(by_id) == ["lo_0", "lo_1", "lo_2", "lo_3"]
    assert all(r["track"] == "location" and r["beat_id"] is None for r in rows)
    assert by_id["lo_0"]["source"] == "r1" and (by_id["lo_0"]["src_in"], by_id["lo_0"]["src_out"]) == (10.0, 15.0)
    photo, card = by_id["lo_1"], by_id["lo_2"]
    assert photo["source"] == "r1" and (photo["src_in"], photo["src_out"]) == (15.0, 18.0)
    assert card["source"] == "r1" and (card["src_in"], card["src_out"]) == (18.0, 21.0)
    assert by_id["lo_3"]["source"] == "r2" and by_id["lo_3"]["src_in"] == 3.0
    assert all(r["gain_lufs"] == -28.0 and r["fade_in_s"] == r["fade_out_s"] == 0.15 for r in rows)


def test_a_photo_that_opens_an_act_has_nothing_to_hold_and_gets_no_cue():
    slots = [_slot(0, 1, 0.0, 5.0, rec="r1"),
             _slot(1, 2, 5.0, 8.0, kind="photo"),          # act 2 opens on a still
             _slot(2, 2, 8.0, 12.0, rec="r2")]
    rows = cues.location_cues(slots, speech_spans=[], windows=[], silence={}, **LEVELS, **FADES)
    assert [r["cue_id"] for r in rows] == ["lo_0", "lo_2"]


def test_a_split_slot_takes_its_primary_recording():
    slots = [_slot(0, 2, 0.0, 4.0, rec="keller", src_in=2.0, secondary="kulikov#9")]
    rows = cues.location_cues(slots, speech_spans=[], windows=[], silence={}, **LEVELS, **FADES)
    assert rows[0]["source"] == "keller" and rows[0]["src_in"] == 2.0


def test_levels_full_in_a_window_under_speech_in_a_span_else_under_music():
    slots = [_slot(i, 3, 5.0 * i, 5.0 * (i + 1), rec=f"r{i}") for i in range(6)]
    windows = [{"t_in": 9.0, "t_out": 23.0, "slot_index": 3}]     # slots 2, 3, 4 by centre
    rows = cues.location_cues(slots, speech_spans=[(0.0, 14.0)], windows=windows, silence={},
                              **LEVELS, **FADES)
    gains = [r["gain_lufs"] for r in rows]
    # slot 1 is under the voice; slot 2 is both under the voice and in the
    # window, and the window wins; slot 5 is under nothing but the music
    assert gains == [-24.0, -24.0, -18.0, -18.0, -18.0, -28.0]
    # an edge inside the window takes the window fade, the other edge the
    # cut fade: slot 1 ends in it and slot 4 starts in it, slots 2-3 lie in it
    assert [r["fade_in_s"] for r in rows] == [0.15, 0.15, 1.0, 1.0, 1.0, 0.15]
    assert [r["fade_out_s"] for r in rows] == [0.15, 1.0, 1.0, 1.0, 0.15, 0.15]


def test_the_silence_window_is_full_location_sound():
    slots = [_slot(i, 5, 100.0 + 5.0 * i, 105.0 + 5.0 * i, rec=f"r{i}") for i in range(3)]
    rows = cues.location_cues(slots, speech_spans=[], windows=[],
                              silence={"t_start": 100.0, "t_end": 110.0}, **LEVELS, **FADES)
    assert [r["gain_lufs"] for r in rows] == [-18.0, -18.0, -28.0]
    # slot 2 opens on the silence's last edge and ends outside it
    assert [r["fade_in_s"] for r in rows] == [1.0, 1.0, 1.0]
    assert [r["fade_out_s"] for r in rows] == [1.0, 1.0, 0.15]


# -- music ----------------------------------------------------------------

def _mmap():
    return {"acts": [
        {"act": 0, "t_start": 0.0, "t_end": 23.0,
         "segments": [{"track_id": "t2", "t_in": 0.0, "t_end": 23.0, "src_in": 90.0, "src_out": 113.0}]},
        {"act": 1, "t_start": 23.0, "t_end": 83.0,
         "segments": [{"track_id": "t1", "t_in": 0.0, "t_end": 40.0, "src_in": 0.0, "src_out": 40.0},
                      {"track_id": "t2", "t_in": 40.0, "t_end": 58.7, "src_in": 10.0, "src_out": 28.7}]},
        {"act": 2, "t_start": 83.0, "t_end": 90.0, "segments": []},
        {"act": 4, "t_start": 90.0, "t_end": 150.0,
         "segments": [{"track_id": "t1", "t_in": 0.0, "t_end": 60.0, "src_in": 40.0, "src_out": 100.0}]},
        {"act": 5, "t_start": 150.0, "t_end": 200.0,
         "segments": [{"track_id": "t2", "t_in": 0.0, "t_end": 50.0, "src_in": 0.0, "src_out": 50.0}]},
    ], "silence_window": {"t_start": 150.0, "t_end": 162.0}}


def test_music_cues_cover_every_act_end_to_end_on_film_time():
    rows = cues.music_cues(_mmap(), lufs=-14.0, xfade_s=2.0, window_fade_s=1.0)
    by_id = {r["cue_id"]: r for r in rows}
    assert list(by_id) == ["mu_0_0", "mu_1_0", "mu_1_1", "mu_4_0", "mu_5_0"]
    assert all(r["track"] == "music" and r["gain_lufs"] == -14.0 and r["beat_id"] is None for r in rows)
    assert (by_id["mu_0_0"]["t_in"], by_id["mu_0_0"]["t_out"]) == (0.0, 23.0)
    assert by_id["mu_0_0"]["source"] == "t2" and by_id["mu_0_0"]["src_in"] == 90.0
    assert (by_id["mu_1_0"]["t_in"], by_id["mu_1_0"]["t_out"]) == (23.0, 63.0)
    # the map's last segment stops at its last scene, 1.3 s short of the act:
    # the bed runs on to the act's end rather than dropping out before the cut
    assert (by_id["mu_1_1"]["t_in"], by_id["mu_1_1"]["t_out"]) == (63.0, 83.0)
    assert (by_id["mu_1_1"]["src_in"], by_id["mu_1_1"]["src_out"]) == (10.0, 30.0)
    assert by_id["mu_1_1"]["fade_in_s"] == 2.0
    for a in _mmap()["acts"]:
        if not a["segments"] or a["act"] == 5:
            continue
        mine = [r for r in rows if r["cue_id"].startswith(f"mu_{a['act']}_")]
        assert math.isclose(min(r["t_in"] for r in mine), a["t_start"])
        assert math.isclose(max(r["t_out"] for r in mine), a["t_end"])


def test_the_silence_gets_no_music_and_the_cue_before_it_fades_over_the_window():
    rows = cues.music_cues(_mmap(), lufs=-14.0, xfade_s=2.0, window_fade_s=1.0)
    by_id = {r["cue_id"]: r for r in rows}
    assert by_id["mu_4_0"]["t_out"] == 150.0 and by_id["mu_4_0"]["fade_out_s"] == 1.0
    assert by_id["mu_4_0"]["fade_in_s"] == 2.0
    # act 5 opens inside the silence: its music resumes where the silence ends,
    # the track picked up by the same amount so the map's grid still holds
    after = by_id["mu_5_0"]
    assert (after["t_in"], after["t_out"]) == (162.0, 200.0)
    assert (after["src_in"], after["src_out"]) == (12.0, 50.0)
    assert not any(r["t_in"] < 162.0 < r["t_out"] or 150.0 < r["t_in"] < 162.0 for r in rows)


def test_the_map_is_laid_on_the_acts_the_table_actually_has():
    # the map was built on the planned spans; on the table act 1 ran 2 s
    # long, so act 4 opens 2 s late and ends 5 s early, and act 5 starts there
    spans = {0: (0.0, 23.0), 1: (23.0, 85.0), 2: (85.0, 92.0), 4: (92.0, 147.0), 5: (147.0, 200.0)}
    rows = cues.music_cues(cues.map_on_film_time(_mmap(), spans), lufs=-14.0, xfade_s=2.0, window_fade_s=1.0)
    by_id = {r["cue_id"]: r for r in rows}
    assert (by_id["mu_1_1"]["t_in"], by_id["mu_1_1"]["t_out"]) == (63.0, 85.0)
    assert (by_id["mu_1_1"]["src_in"], by_id["mu_1_1"]["src_out"]) == (10.0, 32.0)
    # act 4's segment ran to the map's 150: cut at the act's real end, and
    # the silence moves with it, its length kept, so the fade before it holds
    assert (by_id["mu_4_0"]["t_in"], by_id["mu_4_0"]["t_out"]) == (92.0, 147.0)
    assert by_id["mu_4_0"]["src_out"] == 95.0 and by_id["mu_4_0"]["fade_out_s"] == 1.0
    assert (by_id["mu_5_0"]["t_in"], by_id["mu_5_0"]["t_out"]) == (159.0, 200.0)
    assert (by_id["mu_5_0"]["src_in"], by_id["mu_5_0"]["src_out"]) == (12.0, 53.0)


def test_a_segment_that_begins_past_the_acts_real_end_is_dropped():
    mmap = {"acts": [{"act": 3, "t_start": 100.0, "t_end": 150.0,
                      "segments": [{"track_id": "t1", "t_in": 0.0, "t_end": 30.0, "src_in": 0.0, "src_out": 30.0},
                                   {"track_id": "t2", "t_in": 30.0, "t_end": 50.0, "src_in": 0.0, "src_out": 20.0}]}],
            "silence_window": {}}
    rows = cues.music_cues(cues.map_on_film_time(mmap, {3: (100.0, 125.0)}), lufs=-14.0, xfade_s=2.0,
                           window_fade_s=1.0)
    assert [(r["cue_id"], r["t_in"], r["t_out"], r["src_out"]) for r in rows] == [("mu_3_0", 100.0, 125.0, 25.0)]


def test_the_last_cue_loops_rather_than_ask_for_a_track_that_has_run_out():
    """The stretch to the table's act end is unbounded -- up to
    music.min_scene_s (45 s) can land on one cue -- and ffmpeg delivers what
    the file holds and stops, so an unbounded stretch is dead bed under the
    act's closing shots. Here the table's act is 40 s longer than the map
    planned and the track has 10 s left past where the cue already plays."""
    mmap = {"acts": [{"act": 3, "t_start": 0.0, "t_end": 60.0,
                      "segments": [{"track_id": "t1", "t_in": 0.0, "t_end": 60.0,
                                    "src_in": 20.0, "src_out": 80.0}]}],
            "silence_window": {}}
    on_film = cues.map_on_film_time(mmap, {3: (0.0, 100.0)})

    rows = cues.music_cues(on_film, lufs=-14.0, xfade_s=2.0, window_fade_s=1.0,
                           track_s={"t1": 90.0})
    assert [(r["cue_id"], r["t_in"], r["t_out"], r["src_in"], r["src_out"]) for r in rows] == [
        ("mu_3_0", 0.0, 70.0, 20.0, 90.0),        # 60s of map + the 10s the track had left
        ("mu_3_0c1", 70.0, 100.0, 20.0, 50.0)]    # the rest, from the segment's own section cue
    assert all(r["source"] == "t1" for r in rows)

    # without the durations nothing knows where the file ends: today's
    # behaviour, one cue asking 30s past it
    plain = cues.music_cues(on_film, lufs=-14.0, xfade_s=2.0, window_fade_s=1.0)
    assert [(r["t_in"], r["t_out"], r["src_out"]) for r in plain] == [(0.0, 100.0, 120.0)]


def test_a_segment_wholly_inside_the_silence_is_dropped():
    mmap = {"acts": [{"act": 5, "t_start": 150.0, "t_end": 160.0,
                      "segments": [{"track_id": "t1", "t_in": 0.0, "t_end": 10.0, "src_in": 0.0, "src_out": 10.0}]}],
            "silence_window": {"t_start": 150.0, "t_end": 162.0}}
    assert cues.music_cues(mmap, lufs=-14.0, xfade_s=2.0, window_fade_s=1.0) == []


# -- overlays -------------------------------------------------------------

def _beat(beat_id, kind, *, rank=None, msg_id=None, author=None, text="20 км в день"):
    return {"beat_id": beat_id, "kind": kind, "act": 1 if kind == "quote" else 5, "rank": rank,
            "msg_id": msg_id, "author": author, "text": text}


def test_quote_cards_spread_over_act_one_alternating_sides_with_tags_not_names():
    slots = [_slot(0, 0, 0.0, 20.0, rec="c"), _slot(1, 0, 20.0, 23.0, kind="card")]
    slots += [_slot(2 + i, 1, 23.0 + 10.0 * i, 33.0 + 10.0 * i, rec=f"a{i}") for i in range(6)]   # act 1: 23..83
    slots += [_slot(8, 2, 83.0, 90.0, rec="z")]
    beats = [_beat("q_late", "quote", rank=5, msg_id="m2", author="Kulikov"),
             _beat("q_first", "quote", rank=1, msg_id="m1", author="Keller"),
             _beat("q_mid", "quote", rank=3, msg_id="m3", author="Kulikov"),
             _beat("b_speech", "speech", rank=2)]
    rows = cues.overlay_rows(beats, slots, chat_card_s=4.0, closing_card_s=6.0,
                             cast_tags={"Keller": "A", "Kulikov": "B"})
    assert [r["overlay_id"] for r in rows] == ["cc_q_first", "cc_q_mid", "cc_q_late"]
    assert all(r["kind"] == "chat_card" and r["asset_path"] is None for r in rows)
    assert set(rows[0]) == set(cues.OVERLAY_KEYS)
    payloads = [json.loads(r["payload"]) for r in rows]
    assert [p["side"] for p in payloads] == ["left", "right", "left"]
    assert [p["author_tag"] for p in payloads] == ["A", "B", "B"]
    assert [p["msg_id"] for p in payloads] == ["m1", "m3", "m2"]
    assert "Kulikov" not in json.dumps(rows) and "Keller" not in json.dumps(rows)
    # evenly over 23..83 in thirds, each card centred in its third and 4 s long
    assert [(r["t_in"], r["t_out"]) for r in rows] == [(31.0, 35.0), (51.0, 55.0), (71.0, 75.0)]


def test_the_closing_line_sits_over_the_last_seconds_and_no_quotes_is_fine():
    slots = [_slot(0, 4, 0.0, 30.0, rec="c"), _slot(1, 5, 30.0, 100.0, rec="d")]
    beats = [_beat("c_end", "closing", rank=1, msg_id="m9", author="Keller")]
    rows = cues.overlay_rows(beats, slots, chat_card_s=4.0, closing_card_s=6.0, cast_tags={"Keller": "A"})
    assert len(rows) == 1 and rows[0]["overlay_id"] == "cc_c_end"
    assert (rows[0]["t_in"], rows[0]["t_out"]) == (94.0, 100.0)
    assert json.loads(rows[0]["payload"]) == {"text": "20 км в день", "author_tag": "A",
                                              "side": "left", "msg_id": "m9"}


def test_a_beat_without_a_message_carries_no_author_tag():
    slots = [_slot(0, 1, 0.0, 10.0, rec="c")]
    rows = cues.overlay_rows([_beat("q_x", "quote", rank=1)], slots, chat_card_s=4.0,
                             closing_card_s=6.0, cast_tags={"Keller": "A"})
    assert json.loads(rows[0]["payload"])["author_tag"] is None
