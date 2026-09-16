"""S07 -- the draft render command, and one real render at the boundary."""
import shutil
import subprocess
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest

from nepal.process import render

ROWS = [
    {"shot_id": "a#0001", "t_in": 0.0, "t_out": 3.0, "src_in": 5.0},
    {"shot_id": "b#0002", "t_in": 3.0, "t_out": 7.0, "src_in": 0.0},
]
SRC = {"a#0001": pathlib.Path("/m/a.mp4"), "b#0002": pathlib.Path("/m/b.mp4")}


# -- timecode and escaping ---------------------------------------------

@pytest.mark.parametrize("s,want", [(0, "0:00"), (9.4, "0:09"), (61, "1:01"), (252, "4:12")])
def test_timecode_reads_the_way_a_person_says_it(s, want):
    assert render.timecode(s) == want


def test_drawtext_escapes_what_ffmpeg_would_eat():
    """A shot_id contains '#', a place name can contain an apostrophe, and a
    colon ends a drawtext option -- all of these arrive from the database."""
    got = render.escape_drawtext("camera_1#0002 4:12 O'Hara 50%")
    assert r"\:" in got and r"\'" in got and r"\%" in got
    assert "#" in got, "a hash is safe and must not be mangled"


# -- the filter chain ---------------------------------------------------

def test_each_shot_is_trimmed_at_the_input_not_after_the_decoder():
    """A trim filter runs after decoding, so a shot twenty minutes into a
    recording costs twenty minutes of decode. Measured: no output frames at
    all in three minutes over this timeline. Seek at the input instead."""
    assert "trim=" not in render.segment_filters(ROWS[0], 0)
    cmd = render.build_command(ROWS, sources=SRC, out_path=pathlib.Path("/o.mp4"))
    i = cmd.index("-i")
    assert cmd[i - 4] == "-ss" and cmd[i - 3] == "5.000"
    assert cmd[i - 2] == "-t" and cmd[i - 1] == "3.000"


def test_timestamps_are_reset_after_every_trim():
    """Without this the concat lands at the source's time, not at zero --
    a whole cut that starts minutes in and plays black."""
    assert "setpts=PTS-STARTPTS" in render.segment_filters(ROWS[0], 0)


def test_the_frame_is_fitted_not_stretched():
    """The same lesson as the S03.1 proxies: this corpus is not all 16:9."""
    f = render.segment_filters(ROWS[0], 0, width=960, height=540)
    assert "force_original_aspect_ratio=decrease" in f
    assert "pad=960:540" in f, "letterbox rather than distort"


@pytest.mark.skipif(not render.has_drawtext(), reason="ffmpeg has no drawtext")
def test_the_overlay_names_a_row_someone_can_find():
    f = render.segment_filters(ROWS[1], 1)
    # the timecode's colon is escaped for drawtext, so look for that form
    assert "b#0002" in f and r"0\:03" in f


def test_an_ffmpeg_without_drawtext_still_renders(monkeypatch):
    """Many distribution builds omit libfreetype. The overlay is lost, which
    the stage says out loud -- but the draft still gets made."""
    monkeypatch.setattr(render, "has_drawtext", lambda **kw: False)
    f = render.segment_filters(ROWS[0], 0, overlay=True)
    assert "drawtext" not in f
    cmd = render.build_command(ROWS, sources=SRC, out_path=pathlib.Path("/o.mp4"))
    assert "drawtext" not in " ".join(cmd)


def test_the_overlay_can_be_turned_off():
    assert "drawtext" not in render.segment_filters(ROWS[0], 0, overlay=False)


# -- the command --------------------------------------------------------

def test_every_shot_becomes_an_input_and_a_concat_leg():
    cmd = render.build_command(ROWS, sources=SRC, out_path=pathlib.Path("/o.mp4"))
    assert cmd.count("-i") == 2
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "concat=n=2:v=1:a=0[vout]" in fc


def test_music_is_normalised_to_the_spec_target_not_assumed():
    cmd = render.build_command(ROWS, sources=SRC, out_path=pathlib.Path("/o.mp4"),
                               music_path=pathlib.Path("/m/bed.mp3"), music_lufs=-14.0)
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "loudnorm=I=-14" in fc
    assert "-shortest" in cmd, "the bed must not outlast the picture"


def test_without_music_there_is_no_audio_stream_to_map():
    cmd = render.build_command(ROWS, sources=SRC, out_path=pathlib.Path("/o.mp4"))
    assert "[aout]" not in " ".join(cmd)
    assert "-c:a" not in cmd


def test_a_missing_source_is_an_error_not_a_silent_gap():
    with pytest.raises(KeyError):
        render.build_command(ROWS, sources={"a#0001": pathlib.Path("/m/a.mp4")},
                             out_path=pathlib.Path("/o.mp4"))


def test_an_empty_timeline_refuses_rather_than_rendering_nothing():
    with pytest.raises(ValueError):
        render.build_command([], sources=SRC, out_path=pathlib.Path("/o.mp4"))


# -- photographs --------------------------------------------------------

def test_a_photograph_is_held_for_its_slot_not_shown_for_one_frame():
    """94 of the first draft's 220 slots were stills, and without -loop each
    contributed a single frame: the cut lost half its running time."""
    rows = [{"shot_id": "p1", "media_kind": "photo", "t_in": 0.0, "t_out": 3.0}]
    cmd = render.build_command(rows, sources={"p1": pathlib.Path("/m/a.jpg")},
                               out_path=pathlib.Path("/o.mp4"))
    # the hold is in the filter graph, not a -loop input option: that option
    # is private to image2 and absent from the demuxer that reads HEIC
    assert "-loop" not in cmd
    assert "-ss" not in cmd, "a still has nowhere to seek to"
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "loop=loop=-1:size=1" in fc and "trim=duration=3.000" in fc


def test_video_and_stills_can_share_one_timeline():
    rows = [{"shot_id": "v1", "media_kind": "video", "t_in": 0.0, "t_out": 2.0, "src_in": 7.0},
            {"shot_id": "p1", "media_kind": "photo", "t_in": 2.0, "t_out": 5.0}]
    cmd = render.build_command(rows, sources={"v1": pathlib.Path("/m/v.mp4"),
                                              "p1": pathlib.Path("/m/a.jpg")},
                               out_path=pathlib.Path("/o.mp4"))
    assert "-ss" in cmd
    fc0 = cmd[cmd.index("-filter_complex") + 1]
    assert "loop=loop=-1" in fc0, "the still is held"
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "concat=n=2" in fc


@pytest.mark.slow
def test_ffmpeg_holds_a_real_photograph_for_its_full_slot(tmp_path):
    """At the boundary: a JPEG must become three seconds of video, not one
    frame. Checked with ffprobe rather than by reading the command back."""
    if not shutil.which("ffmpeg"):
        pytest.skip("needs ffmpeg")
    img = tmp_path / "still.jpg"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc=size=1200x900:rate=1:duration=1",
                    "-frames:v", "1", str(img)], check=True)
    out = tmp_path / "held.mp4"
    rows = [{"shot_id": "p1", "media_kind": "photo", "t_in": 0.0, "t_out": 3.0}]
    subprocess.run(render.build_command(rows, sources={"p1": img}, out_path=out),
                   check=True)
    dur = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(out)], capture_output=True, text=True,
        check=True).stdout.strip())
    assert dur == pytest.approx(3.0, abs=0.3), f"a held still should be 3 s, got {dur}"


# -- the real boundary --------------------------------------------------

@pytest.mark.slow
def test_ffmpeg_renders_a_cut_of_the_expected_length(tmp_path):
    """Two generated clips, cut to 3 s and 4 s. The result must be 7 s of
    960x540 -- the arithmetic the whole stage rests on, checked by ffprobe
    rather than by reading the command back."""
    if not shutil.which("ffmpeg"):
        pytest.skip("needs ffmpeg")
    srcs = {}
    for name, size in (("a#0001", "640x480"), ("b#0002", "1080x1920")):
        p = tmp_path / f"{name.replace('#', '_')}.mp4"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-f", "lavfi", "-i", f"testsrc=size={size}:rate=25:duration=10",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)], check=True)
        srcs[name] = p
    out = tmp_path / "draft.mp4"
    cmd = render.build_command(ROWS, sources=srcs, out_path=out)
    subprocess.run(cmd, check=True)
    assert out.exists()
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height:format=duration", "-of", "default=nw=1", str(out)],
        capture_output=True, text=True, check=True).stdout
    assert "width=960" in probe and "height=540" in probe
    dur = float([l for l in probe.splitlines() if l.startswith("duration=")][0].split("=")[1])
    assert dur == pytest.approx(7.0, abs=0.3), f"expected 7 s of cut, got {dur}"


# -- one frame rate (Film v2 step 1) ------------------------------------

def test_every_segment_is_resampled_and_the_output_rate_is_fixed():
    """15 fps proxies, 25 fps stills and 60 fps phone clips concatenated
    without a rate came out as a 120 fps file."""
    rows = [{"shot_id": "v1", "media_kind": "video", "t_in": 0.0, "t_out": 2.0,
             "src_in": 5.0, "src_out": 7.0},
            {"shot_id": "p1", "media_kind": "photo", "t_in": 2.0, "t_out": 5.0}]
    cmd = render.build_command(rows, sources={"v1": pathlib.Path("/m/v.mp4"),
                                              "p1": pathlib.Path("/m/a.jpg")},
                               out_path=pathlib.Path("/o.mp4"), fps=30)
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert fc.count("fps=30") == 2
    assert cmd[cmd.index("-r") + 1] == "30"
    assert "fps=25" not in fc
