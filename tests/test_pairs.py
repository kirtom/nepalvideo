"""Two phones, one moment (Film v2 step 4, task 4) -- pure, no database."""
import json
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from nepal.process.assemble import SLOT_KEYS
from nepal.process.pairs import display_dimensions, find_pairs, is_portrait


def _shot(shot_id, source, start_utc, *, start_s=0.0, end_s=60.0, act=3,
         score_total=0.5, face_cluster=None, width=1080, height=1920,
         recording_id=None):
    return {"shot_id": shot_id, "recording_id": recording_id or shot_id, "source": source,
            "start_utc": start_utc, "start_s": start_s, "end_s": end_s,
            "score_total": score_total, "face_cluster": face_cluster,
            "width": width, "height": height, "act": act}


def test_two_phones_20s_apart_pairs_aligned_to_later_start():
    a = _shot("a", "phone_keller", "2024-05-03T04:00:00+00:00", score_total=0.4)
    b = _shot("b", "phone_kulikov", "2024-05-03T04:00:20+00:00", score_total=0.7)
    pairs = find_pairs([a, b], window_s=60, per_act=2)
    assert len(pairs) == 1
    p = pairs[0]
    # b starts later and scores higher -> primary
    assert p["shot_id"] == "b"
    assert p["secondary_shot_id"] == "a"
    assert p["src_in"] == b["start_s"]
    assert p["secondary_src_in"] == a["start_s"] + 20.0
    assert p["kind"] == "video"
    assert p["motion"] == '{"type":"split"}'
    assert p["speed"] == 1.0
    assert p["transition"] == "cut"
    assert p["locked"] == 0
    assert p["act"] == 3
    assert p["t_in"] is None and p["t_out"] is None
    assert set(SLOT_KEYS) - {"slot_index"} == set(p.keys())


def test_same_source_never_pairs():
    a = _shot("a", "phone_keller", "2024-05-03T04:00:00+00:00")
    b = _shot("b", "phone_keller", "2024-05-03T04:00:20+00:00")
    assert find_pairs([a, b], window_s=60, per_act=2) == []


def test_per_act_cap():
    # Three disjoint candidate pairs in the same act -> only the top 2 by rank survive.
    shots = [
        _shot("a1", "phone_keller", "2024-05-03T04:00:00+00:00", score_total=0.9),
        _shot("a2", "phone_kulikov", "2024-05-03T04:00:10+00:00", score_total=0.9),
        _shot("b1", "phone_keller", "2024-05-03T05:00:00+00:00", score_total=0.5),
        _shot("b2", "phone_kulikov", "2024-05-03T05:00:10+00:00", score_total=0.5),
        _shot("c1", "phone_keller", "2024-05-03T06:00:00+00:00", score_total=0.1),
        _shot("c2", "phone_kulikov", "2024-05-03T06:00:10+00:00", score_total=0.1),
    ]
    pairs = find_pairs(shots, window_s=60, per_act=2)
    assert len(pairs) == 2
    ranked_ids = {frozenset((p["shot_id"], p["secondary_shot_id"])) for p in pairs}
    assert frozenset(("c1", "c2")) not in ranked_ids


def test_shot_cannot_appear_in_two_pairs():
    # a overlaps both b and c; only the higher-ranked pair should be kept, and
    # the loser's shot must not resurface paired with the third shot.
    a = _shot("a", "camera", "2024-05-03T04:00:00+00:00", score_total=0.5)
    b = _shot("b", "phone_keller", "2024-05-03T04:00:05+00:00", score_total=0.9)
    c = _shot("c", "phone_kulikov", "2024-05-03T04:00:10+00:00", score_total=0.1)
    pairs = find_pairs([a, b, c], window_s=60, per_act=2)
    used = [sid for p in pairs for sid in (p["shot_id"], p["secondary_shot_id"])]
    assert len(used) == len(set(used))
    # a-b (0.5+0.9=1.4) outranks a-c (0.5+0.1=0.6) and b-c is same source-family
    # incompatible only if same source; b and c are different sources so b-c also
    # qualifies (0.9+0.1=1.0) but a is already claimed by the top pair, so with
    # per_act=2 both non-overlapping-in-shots pairs cannot both include a.
    assert len(pairs) == 1
    assert {pairs[0]["shot_id"], pairs[0]["secondary_shot_id"]} == {"a", "b"}


def test_landscape_landscape_never_pairs():
    a = _shot("a", "phone_keller", "2024-05-03T04:00:00+00:00", width=1920, height=1080)
    b = _shot("b", "phone_kulikov", "2024-05-03T04:00:20+00:00", width=1920, height=1080)
    assert find_pairs([a, b], window_s=60, per_act=2) == []


def test_gap_within_window_but_no_overlap_never_pairs():
    # A ends before B starts; the starts are 5s apart (inside window_s=10)
    # but the spans share no footage, so this must not become a pair.
    a = _shot("a", "phone_keller", "2024-05-03T04:00:00+00:00", start_s=0.0, end_s=4.0)
    b = _shot("b", "phone_kulikov", "2024-05-03T04:00:05+00:00", start_s=0.0, end_s=60.0)
    assert find_pairs([a, b], window_s=10, per_act=2) == []


def test_is_portrait():
    assert is_portrait({"width": 1080, "height": 1920}) is True
    assert is_portrait({"width": 1920, "height": 1080}) is False
    assert is_portrait({"width": None, "height": None}) is False
    assert is_portrait({}) is False


def _probe(video_stream):
    return json.dumps({"streams": [video_stream]})


def test_display_dimensions_tag_rotate_90_swaps():
    probe = _probe({"codec_type": "video", "tags": {"rotate": "90"}})
    assert display_dimensions(1920, 1080, probe) == (1080, 1920)


def test_display_dimensions_tag_rotate_270_swaps():
    probe = _probe({"codec_type": "video", "tags": {"rotate": "270"}})
    assert display_dimensions(1920, 1080, probe) == (1080, 1920)


def test_display_dimensions_tag_rotate_0_unchanged():
    probe = _probe({"codec_type": "video", "tags": {"rotate": "0"}})
    assert display_dimensions(1920, 1080, probe) == (1920, 1080)


def test_display_dimensions_tag_rotate_180_unchanged():
    probe = _probe({"codec_type": "video", "tags": {"rotate": "180"}})
    assert display_dimensions(1920, 1080, probe) == (1920, 1080)


def test_display_dimensions_no_rotate_tag_unchanged():
    probe = _probe({"codec_type": "video"})
    assert display_dimensions(1920, 1080, probe) == (1920, 1080)


def test_display_dimensions_side_data_rotation_minus_90_swaps():
    # newer ffprobe reports rotation via side_data_list rather than tags,
    # and the sign is the direction of rotation, not whether it is a quarter turn
    probe = _probe({"codec_type": "video",
                    "side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90.0}]})
    assert display_dimensions(1920, 1080, probe) == (1080, 1920)


def test_display_dimensions_malformed_probe_json_unchanged():
    assert display_dimensions(1920, 1080, "{not json") == (1920, 1080)


def test_display_dimensions_none_probe_json_unchanged():
    assert display_dimensions(1920, 1080, None) == (1920, 1080)


def test_display_dimensions_missing_dimensions_unchanged():
    probe = _probe({"codec_type": "video", "tags": {"rotate": "90"}})
    assert display_dimensions(None, None, probe) == (None, None)
