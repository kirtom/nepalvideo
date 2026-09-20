"""The Gate 3 page: what the operator reads next to the draft (Film v2
step 4, task 14). Rendered from a handful of hand-built rows: the page is
pure, and its two promises -- every slot names the moment it shows on the
trek's clock, and nobody is named -- are checked on the string."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from nepal.story import gate3

CAST = {"Kulikov": "A", "Kirill Keller": "B"}
SLOTS = [
    {"slot_index": 0, "act": 0, "kind": "video", "shot_id": "c4#0000", "t_in": 0.0, "t_out": 18.0,
     "src_in": 1.0, "src_out": 19.0, "start_utc": "2024-05-04T01:00:00+00:00", "start_s": 0.0,
     "beat_id": "b_pass", "scene_id": None, "source_name": "camera", "secondary_shot_id": None},
    {"slot_index": 1, "act": 0, "kind": "card", "shot_id": None, "t_in": 18.0, "t_out": 21.0,
     "src_in": None, "src_out": None, "start_utc": None, "start_s": None,
     "beat_id": None, "scene_id": None, "source_name": None, "secondary_shot_id": None},
    {"slot_index": 2, "act": 1, "kind": "video", "shot_id": "q1#0000", "t_in": 21.0, "t_out": 27.5,
     "src_in": 4.5, "src_out": 11.0, "start_utc": "2024-03-12T09:00:00+00:00", "start_s": 0.0,
     "beat_id": None, "scene_id": 1, "source_name": "phone_kulikov", "secondary_shot_id": "k3#0000"},
    {"slot_index": 3, "act": 1, "kind": "photo", "shot_id": "photo_p1", "t_in": 27.5, "t_out": 31.5,
     "src_in": 0.0, "src_out": 4.0, "start_utc": "2024-03-15T10:30:00+00:00", "start_s": 20.0,
     "beat_id": None, "scene_id": 1, "source_name": "phone_keller", "secondary_shot_id": None},
]
CUES = [
    {"cue_id": "sp_b_pass", "track": "speech", "t_in": 0.0, "t_out": 8.0, "source": "c4", "src_in": 1.0,
     "src_out": 9.0, "gain_lufs": -16.0, "fade_in_s": 0.15, "fade_out_s": 0.15, "beat_id": "b_pass"},
    {"cue_id": "lo_0", "track": "location", "t_in": 0.0, "t_out": 18.0, "source": "c4", "src_in": 1.0,
     "src_out": 19.0, "gain_lufs": -24.0, "fade_in_s": 0.15, "fade_out_s": 0.15, "beat_id": None},
    {"cue_id": "lo_2", "track": "location", "t_in": 21.0, "t_out": 27.5, "source": "q1", "src_in": 4.5,
     "src_out": 11.0, "gain_lufs": -28.0, "fade_in_s": 0.15, "fade_out_s": 0.15, "beat_id": None},
    {"cue_id": "mu_1_0", "track": "music", "t_in": 21.0, "t_out": 31.5, "source": "t1", "src_in": 0.0,
     "src_out": 10.5, "gain_lufs": -14.0, "fade_in_s": 2.0, "fade_out_s": 2.0, "beat_id": None},
]
OVERLAYS = [{"overlay_id": "cc_q_plan", "kind": "chat_card", "t_in": 22.0, "t_out": 26.0,
             "payload": '{"text": "20 км в день не проблема", "author_tag": "A", "side": "left", "msg_id": "m1"}',
             "asset_path": None}]
PER_ACT = {"1": {"phone_kulikov": {"slots": 3, "available": 12, "videos": 10, "video_slots": 3},
                 "camera": {"slots": 9, "available": 30, "videos": 30, "video_slots": 9}}}
REPORT = {"timeline": {"n_slots": 4, "duration_s": 31.5, "n_anchors": 3, "per_act": {"1": 3},
                       "long_take": "rb"},
          "cues": {"n_cues": {"speech": 1, "location": 2, "music": 1}, "n_overlays": 1},
          "draft": {"draft_s": 31.47, "audio_tracks": 3, "n_dropped_cues": 0}}


def _page():
    return gate3.render_gate3(SLOTS, CUES, OVERLAYS, per_act_sources=PER_ACT, report=REPORT, cast_tags=CAST)


def test_the_wall_clock_is_the_shots_stamp_moved_into_the_slot_on_nepals_clock():
    # 01:00:00 UTC, one second into the recording, at UTC+05:45
    assert gate3.wall_clock(SLOTS[0]) == "2024-05-04 06:45:01"
    # 4.5 s into a recording that started at 09:00:00 UTC
    assert gate3.wall_clock(SLOTS[2]) == "2024-03-12 14:45:04"
    # a still's src_in counts from its start_s, which is not zero here
    assert gate3.wall_clock(SLOTS[3]) == "2024-03-15 16:14:40"
    assert gate3.wall_clock(SLOTS[1]) == "-"


def test_the_page_lists_every_slot_by_wall_clock_and_index():
    page = _page()
    for s in SLOTS:
        assert gate3.wall_clock(s) in page
    assert "c4#0000" in page and "b_pass" in page and "photo_p1" in page
    assert "split" in page, "a split slot says so"


def test_the_page_carries_the_source_ratio_the_cue_counts_and_the_cards():
    page = _page()
    assert "25%" in page and "75%" in page, "3 and 9 of 12 slots in act 1"
    assert "speech 1, location 2, music 1" in page
    assert "sp_b_pass" in page and "mu_1_0" in page
    assert "20 км в день не проблема" in page
    assert "31.47" in page and "long_take" in page, "the run report's numbers"


def test_nobody_is_named_anywhere_on_the_page():
    page = _page()
    for name in ("Kulikov", "kulikov", "Keller", "keller", "Kirill"):
        assert name not in page, name
    assert "phone of A" in page and "phone of B" in page, "phones are named by their owner's tag"
    assert ">A<" in page, "the card carries the author's tag"
