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
