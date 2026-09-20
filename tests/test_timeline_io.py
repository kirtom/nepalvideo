"""S06 exports -- the files the operator opens in an editor."""
import json
import sys, pathlib
import xml.etree.ElementTree as ET
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest

from nepal.process import timeline_io as tio

ROWS = [
    {"shot_id": "a", "act": 1, "t_in": 0.0,  "t_out": 4.0,  "src_in": 10.0, "src_out": 14.0},
    {"shot_id": "b", "act": 1, "t_in": 4.0,  "t_out": 9.0,  "src_in": 0.0,  "src_out": 5.0},
    {"shot_id": "c", "act": 2, "t_in": 9.0,  "t_out": 15.0, "src_in": 2.0,  "src_out": 8.0},
]


# -- OTIO --------------------------------------------------------------

def test_every_shot_becomes_a_clip_in_order():
    t = tio.to_otio(ROWS)
    clips = t["tracks"]["children"][0]["children"]
    assert [c["name"] for c in clips] == ["a", "b", "c"]
    assert all(c["OTIO_SCHEMA"] == "Clip.1" for c in clips)


def test_clip_duration_is_the_timeline_duration_in_frames():
    clips = tio.to_otio(ROWS, fps=25)["tracks"]["children"][0]["children"]
    assert clips[0]["source_range"]["duration"]["value"] == 100     # 4 s at 25 fps
    assert clips[2]["source_range"]["duration"]["value"] == 150     # 6 s


def test_the_clip_starts_where_the_shot_starts_in_its_source():
    clips = tio.to_otio(ROWS, fps=25)["tracks"]["children"][0]["children"]
    assert clips[0]["source_range"]["start_time"]["value"] == 250   # 10 s in


def test_a_hole_in_the_timeline_becomes_a_gap_not_a_silent_overlap():
    rows = [dict(ROWS[0]), {"shot_id": "z", "t_in": 6.0, "t_out": 8.0, "src_in": 0.0}]
    kinds = [c["OTIO_SCHEMA"] for c in
             tio.to_otio(rows)["tracks"]["children"][0]["children"]]
    assert kinds == ["Clip.1", "Gap.1", "Clip.1"]


def test_the_otio_is_json_serialisable():
    json.dumps(tio.to_otio(ROWS))          # raises if a numpy type leaked in


# -- FCPXML ------------------------------------------------------------

def test_fcpxml_parses_and_has_one_clip_per_shot():
    root = ET.fromstring(tio.to_fcpxml(ROWS, media_dir="/m"))
    clips = root.findall(".//asset-clip")
    assert [c.get("name") for c in clips] == ["a", "b", "c"]


def test_times_land_on_frame_boundaries():
    """A time off the grid is silently rounded by the importer, which is how a
    cut drifts from what the pipeline computed."""
    assert tio.fcp_time(4.0, 25) == "4000/1000s"
    assert tio.fcp_time(0.04, 25) == "40/1000s"
    root = ET.fromstring(tio.to_fcpxml(ROWS, media_dir="/m"))
    for clip in root.findall(".//asset-clip"):
        for attr in ("offset", "start", "duration"):
            num, den = clip.get(attr).rstrip("s").split("/")
            assert int(num) % (int(den) // 25) == 0, (attr, clip.get(attr))


def test_one_asset_per_distinct_source_not_per_clip():
    """Two shots from one recording must not import as two media files."""
    rows = [{"shot_id": "a", "source": "rec1.mp4", "t_in": 0.0, "t_out": 2.0, "src_in": 0.0},
            {"shot_id": "b", "source": "rec1.mp4", "t_in": 2.0, "t_out": 4.0, "src_in": 9.0}]
    root = ET.fromstring(tio.to_fcpxml(rows, media_dir="/m"))
    assert len(root.findall(".//asset")) == 1
    assert len(root.findall(".//asset-clip")) == 2


def test_the_sequence_is_as_long_as_the_cut():
    root = ET.fromstring(tio.to_fcpxml(ROWS, media_dir="/m"))
    assert root.find(".//sequence").get("duration") == tio.fcp_time(15.0)


def test_an_empty_timeline_still_produces_a_valid_file():
    root = ET.fromstring(tio.to_fcpxml([], media_dir="/m"))
    assert root.find(".//spine") is not None
    assert root.find(".//sequence").get("duration") == "0/1000s"


def test_write_puts_both_files_where_they_are_expected(tmp_path):
    paths = tio.write(ROWS, tmp_path, media_dir="/m")
    assert paths["otio"].exists() and paths["fcpxml"].exists()
    ET.fromstring(paths["fcpxml"].read_text())
    json.loads(paths["otio"].read_text())


# -- the audio tracks and the overlays (Film v2 step 4, task 14) -------

def _cue(**f):
    return {"cue_id": "x", "track": "speech", "t_in": 0.0, "t_out": 1.0, "source": "r", "src_in": 0.0,
            "src_out": 1.0, "gain_lufs": -16.0, "fade_in_s": 0.15, "fade_out_s": 0.15, "beat_id": None, **f}


CUES = [_cue(cue_id="sp_b", track="speech", t_in=4.0, t_out=7.0, source="rb", src_in=2.0, src_out=5.0, beat_id="b"),
        _cue(cue_id="lo_0", track="location", t_in=0.0, t_out=4.0, source="ra", src_in=10.0, src_out=14.0),
        _cue(cue_id="lo_1", track="location", t_in=4.0, t_out=9.0, source="rb", src_in=0.0, src_out=5.0),
        _cue(cue_id="mu_1_0", track="music", t_in=0.0, t_out=15.0, source="t1", src_in=30.0, src_out=45.0)]
OVERLAYS = [{"overlay_id": "cc_q", "kind": "chat_card", "t_in": 2.0, "t_out": 6.0,
             "payload": '{"text": "20 km a day", "author_tag": "A", "side": "left"}', "asset_path": None}]


def test_without_cues_the_otio_is_the_one_video_track_it_always_was():
    tracks = tio.to_otio(ROWS)["tracks"]["children"]
    assert [t["name"] for t in tracks] == ["V1"]


def test_cues_become_three_audio_tracks_and_the_overlays_a_fourth():
    tracks = tio.to_otio(ROWS, fps=30, cues=CUES, overlays=OVERLAYS)["tracks"]["children"]
    assert [t["name"] for t in tracks] == ["V1", "speech", "location", "music", "overlays"]
    assert [t["kind"] for t in tracks[1:4]] == ["Audio"] * 3
    by_name = {t["name"]: t["children"] for t in tracks}
    # a cue that starts late sits after a gap, on film time, at its own source range
    gap, sp = by_name["speech"]
    assert gap["OTIO_SCHEMA"] == "Gap.1" and gap["source_range"]["duration"]["value"] == 120   # 4 s at 30 fps
    assert sp["OTIO_SCHEMA"] == "Clip.1" and sp["name"] == "sp_b"
    assert sp["source_range"]["start_time"]["value"] == 60 and sp["source_range"]["duration"]["value"] == 90
    assert sp["metadata"]["nepal"]["source"] == "rb" and sp["metadata"]["nepal"]["beat_id"] == "b"
    # abutting cues need no gap between them
    assert [c["OTIO_SCHEMA"] for c in by_name["location"]] == ["Clip.1", "Clip.1"]
    assert by_name["music"][0]["source_range"]["start_time"]["value"] == 900
    # an overlay is a placed gap carrying its payload, not a clip of any media
    spacer, card = by_name["overlays"]
    assert card["OTIO_SCHEMA"] == "Gap.1" and card["name"] == "cc_q"
    assert spacer["source_range"]["duration"]["value"] == 60 and card["source_range"]["duration"]["value"] == 120
    assert card["metadata"]["nepal"]["kind"] == "chat_card" and "20 km" in card["metadata"]["nepal"]["payload"]
    json.dumps(tracks)


def test_write_carries_the_cues_into_the_otio_and_the_fps_into_the_fcpxml(tmp_path):
    paths = tio.write(ROWS, tmp_path, media_dir="/m", fps=30, cues=CUES, overlays=OVERLAYS)
    otio = json.loads(paths["otio"].read_text())
    assert [t["name"] for t in otio["tracks"]["children"]] == ["V1", "speech", "location", "music", "overlays"]
    root = ET.fromstring(paths["fcpxml"].read_text())
    assert root.find(".//format").get("frameDuration") == tio.fcp_time(1.0 / 30, 30)
    assert [c.get("name") for c in root.findall(".//asset-clip")] == ["a", "b", "c"], "the FCPXML is unchanged"
