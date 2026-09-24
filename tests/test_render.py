"""S07 -- the draft render command, and one real render at the boundary."""
import json
import logging
import re
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


def _graph(cmd):
    """The filter graph now lives in a script file next to out_path, not in
    the command itself -- read it back the way ffmpeg would."""
    return pathlib.Path(cmd[cmd.index("-filter_complex_script") + 1]).read_text(encoding="utf-8")


def _apads(cmd):
    """Every audio input pad the graph reads, in order."""
    return sorted(int(k) for k in re.findall(r"\[(\d+):a\]", _graph(cmd)))


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

def test_each_shot_is_trimmed_at_the_input_not_after_the_decoder(tmp_path):
    """A trim filter runs after decoding, so a shot twenty minutes into a
    recording costs twenty minutes of decode. Measured: no output frames at
    all in three minutes over this timeline. Seek at the input instead."""
    assert "trim=" not in render.segment_filters(ROWS[0], 0)
    cmd = render.build_command(ROWS, sources=SRC, out_path=tmp_path / "o.mp4")
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


def test_an_ffmpeg_without_drawtext_still_renders(monkeypatch, tmp_path):
    """Many distribution builds omit libfreetype. The overlay is lost, which
    the stage says out loud -- but the draft still gets made."""
    monkeypatch.setattr(render, "has_drawtext", lambda **kw: False)
    f = render.segment_filters(ROWS[0], 0, overlay=True)
    assert "drawtext" not in f
    cmd = render.build_command(ROWS, sources=SRC, out_path=tmp_path / "o.mp4")
    assert "drawtext" not in _graph(cmd)


def test_the_overlay_can_be_turned_off():
    assert "drawtext" not in render.segment_filters(ROWS[0], 0, overlay=False)


# -- the command --------------------------------------------------------

def test_every_shot_becomes_an_input_and_a_concat_leg(tmp_path):
    cmd = render.build_command(ROWS, sources=SRC, out_path=tmp_path / "o.mp4")
    assert cmd.count("-i") == 2
    fc = _graph(cmd)
    assert "concat=n=2:v=1:a=0[vout]" in fc


def test_without_music_there_is_no_audio_stream_to_map(tmp_path):
    cmd = render.build_command(ROWS, sources=SRC, out_path=tmp_path / "o.mp4")
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

def test_a_photograph_is_held_for_its_slot_not_shown_for_one_frame(tmp_path):
    """94 of the first draft's 220 slots were stills, and without -loop each
    contributed a single frame: the cut lost half its running time."""
    rows = [{"shot_id": "p1", "media_kind": "photo", "t_in": 0.0, "t_out": 3.0}]
    cmd = render.build_command(rows, sources={"p1": pathlib.Path("/m/a.jpg")},
                               out_path=tmp_path / "o.mp4")
    # the hold is in the filter graph, not a -loop input option: that option
    # is private to image2 and absent from the demuxer that reads HEIC
    assert "-loop" not in cmd
    assert "-ss" not in cmd, "a still has nowhere to seek to"
    fc = _graph(cmd)
    assert "loop=loop=-1:size=1" in fc and "trim=duration=3.000" in fc


def test_video_and_stills_can_share_one_timeline(tmp_path):
    rows = [{"shot_id": "v1", "media_kind": "video", "t_in": 0.0, "t_out": 2.0, "src_in": 7.0},
            {"shot_id": "p1", "media_kind": "photo", "t_in": 2.0, "t_out": 5.0}]
    cmd = render.build_command(rows, sources={"v1": pathlib.Path("/m/v.mp4"),
                                              "p1": pathlib.Path("/m/a.jpg")},
                               out_path=tmp_path / "o.mp4")
    assert "-ss" in cmd
    fc = _graph(cmd)
    assert "loop=loop=-1" in fc, "the still is held"
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
    assert render.run_render(render.build_command(rows, sources={"p1": img}, out_path=out),
                             out).returncode == 0
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
    # The graph goes through a script file, not the command line -- confirm
    # ffmpeg actually reads and renders it from there, not just that the
    # command shape looks right.
    assert "-filter_complex_script" in cmd
    filters_path = out.with_suffix(".filters")
    assert filters_path.read_text(), "the script file must hold the real graph"
    assert render.run_render(cmd, out).returncode == 0
    assert out.exists()
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height:format=duration", "-of", "default=nw=1", str(out)],
        capture_output=True, text=True, check=True).stdout
    assert "width=960" in probe and "height=540" in probe
    dur = float([l for l in probe.splitlines() if l.startswith("duration=")][0].split("=")[1])
    assert dur == pytest.approx(7.0, abs=0.3), f"expected 7 s of cut, got {dur}"


@pytest.mark.slow
def test_ffmpeg_renders_a_card_between_real_footage_and_a_still(tmp_path):
    """A card (lavfi color + drawtext, no file) sitting between a decoded
    video and a decoded JPEG -- a real photograph decodes to yuvj444p or
    yuvj420p (full-range), not the yuv420p the encoder is asked for -- must
    still concat cleanly to one yuv420p file of the sum of the three
    segment lengths. Checked with ffprobe, not by reading the command back."""
    if not shutil.which("ffmpeg"):
        pytest.skip("needs ffmpeg")
    vid = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc=size=640x480:rate=25:duration=10",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(vid)], check=True)
    img = tmp_path / "still.jpg"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc=size=1200x900:rate=1:duration=1",
                    "-pix_fmt", "yuvj444p", "-frames:v", "1", str(img)], check=True)
    rows = [
        {"shot_id": "v1", "media_kind": "video", "t_in": 0.0, "t_out": 3.0, "src_in": 0.0},
        {"kind": "card", "t_in": 3.0, "t_out": 5.0,
         "motion": json.dumps({"type": "card", "text": "Nepal"})},
        {"shot_id": "p1", "media_kind": "photo", "t_in": 5.0, "t_out": 9.0},
    ]
    out = tmp_path / "draft.mp4"
    cmd = render.build_command(rows, sources={"v1": vid, "p1": img}, out_path=out)
    assert render.run_render(cmd, out).returncode == 0
    assert out.exists()
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=pix_fmt:format=duration", "-of", "default=nw=1", str(out)],
        capture_output=True, text=True, check=True).stdout
    assert "pix_fmt=yuv420p" in probe
    dur = float([l for l in probe.splitlines() if l.startswith("duration=")][0].split("=")[1])
    expected = sum(r["t_out"] - r["t_in"] for r in rows)
    assert dur == pytest.approx(expected, abs=0.3), f"expected {expected} s, got {dur}"


# -- the card and split slots (Film v2 step 4) --------------------------

@pytest.mark.skipif(not render.has_drawtext(), reason="ffmpeg has no drawtext")
def test_a_card_row_becomes_a_black_frame_with_its_caption(tmp_path):
    """A card slot (the cold-open title) has no shot behind it -- the inner
    join in render_draft used to drop it, shortening the draft by exactly
    its length. It must still take a slot in the concat, at its own length,
    with its caption escaped like any other drawtext."""
    row = {"kind": "card", "t_in": 0.0, "t_out": 2.5,
           "motion": json.dumps({"type": "card", "text": "Nepal: the trek"})}
    cmd = render.build_command([row], sources={}, out_path=tmp_path / "o.mp4")
    assert cmd.count("-i") == 1
    i = cmd.index("-i")
    assert cmd[i - 2:i] == ["-f", "lavfi"], "no file backs a card -- lavfi synthesises it"
    assert "color=" in cmd[i + 1] and "d=2.500" in cmd[i + 1]
    fc = _graph(cmd)
    assert r"drawtext=text='Nepal\: the trek'" in fc, "the caption is escaped, not passed through raw"


def test_a_card_without_drawtext_is_plain_black(monkeypatch, tmp_path):
    """Same tradeoff as the shot overlay: no libfreetype means no caption,
    not a failed render -- the card still holds its length as plain black."""
    monkeypatch.setattr(render, "has_drawtext", lambda **kw: False)
    row = {"kind": "card", "t_in": 0.0, "t_out": 1.0,
           "motion": json.dumps({"type": "card", "text": "Nepal"})}
    f = render.segment_filters(row, 0)
    assert "drawtext" not in f
    cmd = render.build_command([row], sources={}, out_path=tmp_path / "o.mp4")
    assert "drawtext" not in _graph(cmd)


def test_a_split_whose_secondary_has_no_source_renders_its_primary_alone(tmp_path):
    """render_draft did not know the other phone's clip before step 4 wired
    it, and a draft that dies on one slot is worse than one whose split
    shows a single phone -- the same policy as a slot with no media at all:
    degrade loudly, never drop the slot or read the secondary as a source."""
    row = {"shot_id": "a#0001", "t_in": 0.0, "t_out": 3.0, "src_in": 5.0,
           "secondary_shot_id": "z#9999", "secondary_src_in": 1.0,
           "motion": json.dumps({"type": "split"})}
    cmd = render.build_command([row], sources={"a#0001": pathlib.Path("/m/a.mp4")},
                               out_path=tmp_path / "o.mp4")
    assert cmd.count("-i") == 1
    assert "z#9999" not in " ".join(cmd)
    assert "z#9999" not in _graph(cmd)


def test_a_mixed_timeline_keeps_concat_order_and_one_input_per_row(tmp_path):
    rows = [
        {"kind": "card", "t_in": 0.0, "t_out": 1.0,
         "motion": json.dumps({"type": "card", "text": "x"})},
        {"shot_id": "a#0001", "t_in": 1.0, "t_out": 3.0, "src_in": 5.0},
        {"shot_id": "b#0002", "t_in": 3.0, "t_out": 4.0, "src_in": 0.0,
         "secondary_shot_id": "z#9999", "secondary_src_in": 1.0,
         "motion": json.dumps({"type": "split"})},
    ]
    cmd = render.build_command(rows, sources={"a#0001": pathlib.Path("/m/a.mp4"),
                                              "b#0002": pathlib.Path("/m/b.mp4")},
                               out_path=tmp_path / "o.mp4")
    assert cmd.count("-i") == len(rows)
    fc = _graph(cmd)
    assert "[v0][v1][v2]concat=n=3:v=1:a=0[vout]" in fc, "concat must list the legs in row order"


# -- one frame rate (Film v2 step 1) ------------------------------------

def test_every_segment_is_resampled_and_the_output_rate_is_fixed(tmp_path):
    """15 fps proxies, 25 fps stills and 60 fps phone clips concatenated
    without a rate came out as a 120 fps file."""
    rows = [{"shot_id": "v1", "media_kind": "video", "t_in": 0.0, "t_out": 2.0,
             "src_in": 5.0, "src_out": 7.0},
            {"shot_id": "p1", "media_kind": "photo", "t_in": 2.0, "t_out": 5.0}]
    cmd = render.build_command(rows, sources={"v1": pathlib.Path("/m/v.mp4"),
                                              "p1": pathlib.Path("/m/a.jpg")},
                               out_path=tmp_path / "o.mp4", fps=30)
    fc = _graph(cmd)
    assert fc.count("fps=30") == 2
    assert cmd[cmd.index("-r") + 1] == "30"
    assert "fps=25" not in fc


# -- the argument-length ceiling (564 slots is real, 700 is headroom) --

def test_a_700_row_timeline_keeps_every_argument_short(tmp_path):
    """564 slots put the whole filter graph past Linux's MAX_ARG_STRLEN
    (128 KiB) as a single -filter_complex argument, and subprocess.run
    raised OSError: Argument list too long. The graph now goes to a file
    beside out_path, so no argument should even approach the limit."""
    rows = [{"shot_id": f"s#{i:04d}", "t_in": float(i), "t_out": float(i + 1),
             "src_in": 0.0} for i in range(700)]
    sources = {r["shot_id"]: pathlib.Path(f"/m/{r['shot_id']}.mp4") for r in rows}
    cmd = render.build_command(rows, sources=sources, out_path=tmp_path / "o.mp4")
    assert max(len(c) for c in cmd) < 128 * 1024
    assert "-filter_complex_script" in cmd
    assert "-filter_complex" not in cmd


# -- the split slot with both phones, and the caller's caption ----------

SPLIT = {"shot_id": "a#0001", "t_in": 0.0, "t_out": 3.0, "src_in": 5.0,
         "secondary_shot_id": "z#9999", "secondary_src_in": 1.0,
         "motion": json.dumps({"type": "split"})}
BOTH = {"a#0001": pathlib.Path("/m/a.mp4"), "z#9999": pathlib.Path("/m/z.mp4")}


def _inputs(cmd):
    return [cmd[k + 1] for k, c in enumerate(cmd) if c == "-i"]


def test_a_split_slot_puts_both_phones_side_by_side(tmp_path):
    """Both clips are inputs -- the secondary right after its primary, seeked
    to its own src_in for the slot's length -- each fitted to half the frame
    at the one rate and stacked into one concat leg."""
    cmd = render.build_command([SPLIT], sources=BOTH, out_path=tmp_path / "o.mp4",
                               width=960, height=540)
    assert _inputs(cmd) == ["/m/a.mp4", "/m/z.mp4"]
    j = [k for k, c in enumerate(cmd) if c == "-i"][1]
    assert cmd[j - 4:j] == ["-ss", "1.000", "-t", "3.000"]
    fc = _graph(cmd)
    assert "[0:v]" in fc and "[1:v]" in fc and "hstack=inputs=2" in fc
    assert fc.count("scale=480:540") == 2 and fc.count("crop=480:540") == 2
    assert fc.count("setsar=1") == 2 and fc.count("fps=30") == 2
    assert "concat=n=1:v=1:a=0[vout]" in fc


def test_a_split_in_the_middle_keeps_the_legs_in_row_order(tmp_path):
    rows = [ROWS[0], SPLIT | {"t_in": 3.0, "t_out": 6.0}, ROWS[1] | {"t_in": 6.0, "t_out": 10.0}]
    cmd = render.build_command(rows, sources=BOTH | SRC, out_path=tmp_path / "o.mp4")
    assert _inputs(cmd) == ["/m/a.mp4", "/m/a.mp4", "/m/z.mp4", "/m/b.mp4"]
    fc = _graph(cmd)
    assert "[3:v]" in fc, "the row after the split reads the input after the secondary"
    assert "concat=n=3:v=1:a=0[vout]" in fc


@pytest.mark.slow
def test_ffmpeg_renders_a_split_between_two_shots_at_the_frame_size(tmp_path):
    """At the boundary: a landscape clip and a portrait one stacked into one
    960x540 leg between two plain shots. hstack refuses halves that differ
    in height or format, and concat refuses a leg that differs from the
    others -- both only visible when ffmpeg runs the graph."""
    if not shutil.which("ffmpeg"):
        pytest.skip("needs ffmpeg")
    srcs = {}
    for name, size in (("wide", "640x480"), ("tall", "1080x1920")):
        p = tmp_path / f"{name}.mp4"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-f", "lavfi", "-i", f"testsrc=size={size}:rate=25:duration=6",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)], check=True)
        srcs[name] = p
    rows = [{"shot_id": "wide", "t_in": 0.0, "t_out": 2.0, "src_in": 0.0},
            {"shot_id": "wide", "t_in": 2.0, "t_out": 5.0, "src_in": 2.0,
             "secondary_shot_id": "tall", "secondary_src_in": 1.0,
             "motion": json.dumps({"type": "split"})},
            {"shot_id": "tall", "t_in": 5.0, "t_out": 7.0, "src_in": 0.0}]
    out = tmp_path / "draft.mp4"
    assert render.run_render(render.build_command(rows, sources=srcs, out_path=out),
                             out).returncode == 0
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height:format=duration", "-of", "default=nw=1", str(out)],
        capture_output=True, text=True, check=True).stdout
    assert "width=960" in probe and "height=540" in probe
    dur = float([l for l in probe.splitlines() if l.startswith("duration=")][0].split("=")[1])
    assert dur == pytest.approx(7.0, abs=0.3), f"expected 7 s, got {dur}"


@pytest.mark.skipif(not render.has_drawtext(), reason="ffmpeg has no drawtext")
def test_a_card_takes_its_caption_from_the_caller_over_the_row():
    row = {"kind": "card", "t_in": 0.0, "t_out": 2.0,
           "motion": json.dumps({"type": "card", "text": "from the row"})}
    assert "from the row" in render.segment_filters(row, 0)
    f = render.segment_filters(row, 0, card_text="from the caller")
    assert "from the caller" in f and "from the row" not in f


# -- the audio graph (Film v2 step 4, part B) ---------------------------

LEVELS = {"speech_lufs": -16.0, "location_full_lufs": -18.0, "final_lufs": -14.0,
          "true_peak_db": -1.5, "music_xfade_s": 2.0, "cue_fade_s": 0.15,
          "audio_bitrate_k": 192}
# What mix.volume_expr produces: a flat sum of between(t,..)*gain terms,
# opaque to the renderer -- written by hand here so this file never
# imports mix.py.
ENVELOPES = {"location": "between(t,0,3)*1+between(t,3,7)*0.25",
             "music": "between(t,0,7)*0.5"}


def _cue(**f):
    return {"cue_id": None, "track": None, "t_in": None, "t_out": None, "source": None,
            "src_in": None, "src_out": None, "gain_lufs": None, "fade_in_s": 0.15,
            "fade_out_s": 0.15, "beat_id": None} | f


# deliberately out of time order, and interleaved across tracks
CUES = [
    _cue(cue_id="mu_1", track="music", t_in=4.0, t_out=7.0, source="trk_b",
         src_in=10.0, src_out=13.0, gain_lufs=-14.0, fade_in_s=2.0, fade_out_s=2.0),
    _cue(cue_id="lo_1", track="location", t_in=3.0, t_out=7.0, source="rec_b",
         src_in=0.0, src_out=4.0, gain_lufs=-24.0, fade_in_s=0.5, fade_out_s=1.0),
    _cue(cue_id="sp_0", track="speech", t_in=0.5, t_out=2.5, source="rec_a",
         src_in=5.0, src_out=7.0, gain_lufs=-16.0, beat_id="b1",
         fade_in_s=None, fade_out_s=None),
    _cue(cue_id="lo_0", track="location", t_in=0.0, t_out=3.0, source="rec_a",
         src_in=5.0, src_out=8.0, gain_lufs=-18.0),
    _cue(cue_id="mu_0", track="music", t_in=0.0, t_out=4.0, source="trk_a",
         src_in=0.0, src_out=4.0, gain_lufs=-14.0, fade_in_s=0.0, fade_out_s=2.0),
]
AUDIO = {"rec_a": pathlib.Path("/w/rec_a.wav"), "rec_b": pathlib.Path("/w/rec_b.wav")}
MUSIC = {"trk_a": pathlib.Path("/mu/a.mp3"), "trk_b": pathlib.Path("/mu/b.mp3")}


def _mixed(tmp_path, cues=CUES, **kw):
    kw = {"audio_sources": AUDIO, "music_sources": MUSIC,
          "envelopes": ENVELOPES, "levels": LEVELS} | kw
    return render.build_command(ROWS, sources=SRC, out_path=tmp_path / "o.mp4", cues=cues, **kw)


def test_cue_inputs_follow_the_video_inputs_by_track_in_time_order(tmp_path):
    """Every index in the graph is computed from this order, so it is
    pinned: the video rows (each secondary right after its primary), then
    the speech, location and music cues, each track by t_in whatever order
    the cues arrived in."""
    cmd = _mixed(tmp_path)
    assert _inputs(cmd) == ["/m/a.mp4", "/m/b.mp4", "/w/rec_a.wav",
                            "/w/rec_a.wav", "/w/rec_b.wav", "/mu/a.mp3", "/mu/b.mp3"]
    i = [k for k, c in enumerate(cmd) if c == "-i"]
    assert cmd[i[2] - 4:i[2]] == ["-ss", "5.000", "-t", "2.000"], "a cue is seeked at the input like a shot"
    assert cmd[i[6] - 2:i[6]] == ["-t", "13.000"], "a music cue is read from its start to xfade_s past its slot"


def test_a_music_cue_is_decoded_from_the_start_and_cut_in_the_graph(tmp_path):
    """The library is VBR MP3, and an input-side seek on MP3 without a
    table of contents goes by a bitrate estimate that can land seconds
    off -- which the overlap arithmetic cannot survive. (An -ss after the
    -i is not the answer either: between two -i it belongs to the next
    input.) So no -ss before a music input: -t bounds the read, atrim
    cuts to the sample, and the timestamps are reset as a seek would."""
    cmd = _mixed(tmp_path)
    i = [k for k, c in enumerate(cmd) if c == "-i"]
    assert cmd[i[5] - 2:i[5] + 2] == ["-t", "6.000", "-i", "/mu/a.mp3"] and cmd[i[5] - 3] != "-ss"
    assert cmd[i[6] - 2:i[6] + 2] == ["-t", "13.000", "-i", "/mu/b.mp3"] and cmd[i[6] - 3] != "-ss"
    fc = _graph(cmd)
    assert "[5:a]atrim=start=0.000:end=6.000,asetpts=PTS-STARTPTS," in fc
    assert "[6:a]atrim=start=10.000:end=13.000,asetpts=PTS-STARTPTS," in fc


def test_speech_cues_are_normalised_faded_placed_and_summed(tmp_path):
    """sp_0 carries no fades of its own, so it gets the config's cut fade."""
    fc = _graph(_mixed(tmp_path))
    assert ("[2:a]loudnorm=I=-16:TP=-1.5:LRA=11,afade=t=in:d=0.15,"
            "afade=t=out:st=1.850:d=0.15,adelay=500|500[sp0]") in fc
    assert "[sp0]amix=inputs=1:normalize=0[speech]" in fc


def _loc(fc):
    """The location track's own parts of the graph."""
    return [p for p in fc.split(";") if re.search(r"\[log?\d+\]$|\[loc\]$", p)]


def test_location_cues_sit_against_full_level_fade_at_their_edges_and_ride_the_envelope(tmp_path):
    """Every change of recording is a butt-splice of two unrelated
    waveforms unless each cue fades at its edges -- hundreds of clicks at
    picture cuts. The row's own fades where it has them, the config's
    cut fade where it does not. The gains, the fades and the envelope are
    what they always were; only the placement changed."""
    fc = _graph(_mixed(tmp_path))
    fmt = render.CONCAT_AFORMAT
    assert ("[3:a]volume=0dB,afade=t=in:d=0.15,afade=t=out:st=2.850:d=0.15,"
            f"atrim=duration=3.000,apad=whole_dur=3.000,asetpts=N/SR/TB,{fmt}[lo0]") in fc
    assert ("[4:a]volume=-6dB,afade=t=in:d=0.5,afade=t=out:st=3.000:d=1,"
            f"atrim=duration=4.000,apad=whole_dur=4.000,asetpts=N/SR/TB,{fmt}[lo1]") in fc
    assert ("[lo0][lo1]concat=n=2:v=0:a=1,"
            "volume='between(t,0,3)*1+between(t,3,7)*0.25':eval=frame[loc]") in fc


def test_the_location_track_is_laid_end_to_end_rather_than_delayed_and_summed(tmp_path):
    """735 location cues adelayed and amixed made the audio-only
    measurement pass 19.5 minutes at 100 % of one core -- amix summing 735
    streams whose lengths average half the film, ~3e10 samples, for a
    track that is one film laid end to end. The cues tile the timeline, so
    the track is a concat and each piece carries only its own span."""
    fc = _graph(_mixed(tmp_path))
    assert not any("adelay" in p for p in _loc(fc)), "no cue carries the film ahead of it"
    assert "concat=n=2:v=0:a=1" in fc


def test_every_location_piece_times_itself_off_its_own_samples(tmp_path):
    """apad gives a piece its length; it cannot give those samples a time.
    An input that decodes nothing -- a cue seeked at or past the end of its
    recording -- leaves apad's padding with no pts at all, and concat adds
    each piece's end time to the delta it shifts every later piece by. One
    such cue therefore puts the whole rest of the track at a nonsense time.
    Every piece, not only the empty ones, so nothing depends on which
    inputs happened to deliver."""
    pieces = [p for p in _loc(_graph(_mixed(tmp_path))) if "apad=whole_dur" in p]
    assert len(pieces) == 2, pieces
    for piece in pieces:
        assert "asetpts=N/SR/TB" in piece, piece
        assert piece.index("apad=whole_dur") < piece.index("asetpts=N/SR/TB"), \
            "the piece is timed after it is padded, not before"


def test_a_stretch_with_no_location_cue_becomes_silence_of_its_own_length(tmp_path):
    """A card that opens an act has no location cue: intended dead air.
    Summed, that is simply nothing; laid end to end it has to be said, or
    every cue after it lands early."""
    shifted = {"lo_0": 0.5, "lo_1": 4.5}
    moved = [c | {"t_in": shifted[c["cue_id"]]} if c["cue_id"] in shifted else c for c in CUES]
    fc = _graph(_mixed(tmp_path, cues=moved))
    fmt = render.CONCAT_AFORMAT
    assert f"anullsrc=r=48000:cl=stereo,atrim=duration=0.500,{fmt}[log0]" in fc, "the run before the first cue"
    assert f"anullsrc=r=48000:cl=stereo,atrim=duration=1.500,{fmt}[log1]" in fc, "3.0 to 4.5"
    assert "[log0][lo0][log1][lo1]concat=n=4:v=0:a=1" in fc


def test_overlapping_location_cues_go_back_to_being_summed(tmp_path, caplog):
    """Two cues that sound at once cannot be laid end to end. Correctness
    over speed: the track falls back to the placement that can hold them,
    and says which pair cost it."""
    clash = [c if c["cue_id"] != "lo_1" else c | {"t_in": 2.0} for c in CUES]
    with caplog.at_level(logging.WARNING, logger="nepal.process.render"):
        fc = _graph(_mixed(tmp_path, cues=clash))
    assert "concat=n=2:v=0:a=1" not in fc
    assert "adelay=0|0[lo0]" in fc and "adelay=2000|2000[lo1]" in fc
    assert ("[lo0][lo1]amix=inputs=2:normalize=0,"
            "volume='between(t,0,3)*1+between(t,3,7)*0.25':eval=frame[loc]") in fc
    assert "lo_0" in caplog.text and "lo_1" in caplog.text and "overlap" in caplog.text


def test_music_cues_land_at_their_own_t_in_and_crossfade_by_overlap(tmp_path):
    """Chaining cues with acrossfade collapsed every gap and shortened the
    run by one crossfade per join, so every cue after the first hole landed
    early under a picture cut. Each cue is placed by adelay like the others;
    the crossfade is the outgoing cue read its own fade-out longer -- here
    music_xfade_s, as cues.py writes it where a cue hands over -- and
    fading out under the incoming one, which fades in as its row says. The extension
    is clamped to the film's end (mu_1: 3 s of a possible 5 for a film of
    7), the fade-out then sitting inside the cue, so the mix never
    outlasts the picture."""
    fc = _graph(_mixed(tmp_path))
    assert "acrossfade" not in fc
    assert ("[5:a]atrim=start=0.000:end=6.000,asetpts=PTS-STARTPTS,"
            "afade=t=out:st=4.000:d=2,adelay=0|0[mu0]") in fc, "a row with fade_in_s 0 opens clean"
    assert ("[6:a]atrim=start=10.000:end=13.000,asetpts=PTS-STARTPTS,"
            "afade=t=in:d=2,afade=t=out:st=1.000:d=2,adelay=4000|4000[mu1]") in fc
    assert "[mu0][mu1]amix=inputs=2:normalize=0,volume='between(t,0,7)*0.5':eval=frame[mus]" in fc


def test_a_music_cue_fades_out_as_its_row_says_and_reads_only_that_much(tmp_path):
    """The fade-out is the row's, like the other two tracks. cues.py gives
    the cue that hands over to the next one ``music_xfade_s`` and the cue
    that runs into the silence window the shorter window fade; a renderer
    that overrode both with the flat crossfade played a second of music
    into the window and threw the distinction away. The extension read
    past the slot is that same length, so the fade still begins at the
    cue's own end."""
    into_window = [c if c["cue_id"] != "mu_0" else c | {"fade_out_s": 1.0} for c in CUES]
    cmd = _mixed(tmp_path, cues=into_window)
    i = [k for k, c in enumerate(cmd) if c == "-i"]
    assert cmd[i[5] - 2:i[5] + 2] == ["-t", "5.000", "-i", "/mu/a.mp3"], "one second past the slot, not two"
    assert ("[5:a]atrim=start=0.000:end=5.000,asetpts=PTS-STARTPTS,"
            "afade=t=out:st=4.000:d=1,adelay=0|0[mu0]") in _graph(cmd)


def test_a_music_cue_fades_in_as_its_row_says_not_as_the_gap_suggests(tmp_path):
    """cues.py decides: the cue after the silence window carries
    music_xfade_s, the first of a run carries 0; a gap in the timeline is
    no longer read as a reason to skip the fade."""
    later = [c if c["cue_id"] != "mu_1" else c | {"t_in": 6.0, "t_out": 7.0} for c in CUES]
    fc = _graph(_mixed(tmp_path, cues=later))
    assert ("[6:a]atrim=start=10.000:end=11.000,asetpts=PTS-STARTPTS,"
            "afade=t=in:d=2,afade=t=out:st=0.000:d=2,adelay=6000|6000[mu1]") in fc
    clean = [c if c["cue_id"] != "mu_1" else c | {"fade_in_s": 0.0} for c in CUES]
    fc = _graph(_mixed(tmp_path, cues=clean))
    assert ("[6:a]atrim=start=10.000:end=13.000,asetpts=PTS-STARTPTS,"
            "afade=t=out:st=1.000:d=2,adelay=4000|4000[mu1]") in fc, "a zero fade is no afade at all"


def test_the_three_tracks_meet_in_one_mix_and_one_final_normalisation(tmp_path):
    cmd = _mixed(tmp_path)
    fc = _graph(cmd)
    assert "-filter_complex_script" in cmd
    assert "[speech][loc][mus]amix=inputs=3:normalize=0,loudnorm=I=-14:TP=-1.5:LRA=11[aout]" in fc
    assert fc.count("adelay=") == sum(1 for c in CUES if c["track"] != "location"), \
        "speech and music are placed on film time; location is laid end to end"
    assert cmd[cmd.index("-map") + 1] == "[vout]" and "[aout]" in cmd
    assert cmd[cmd.index("-c:a") + 1] == "aac" and cmd[cmd.index("-b:a") + 1] == "192k"
    assert cmd[cmd.index("-ar") + 1] == "48000", "loudnorm hands the encoder 192 kHz; the film is 48"
    assert "-shortest" not in cmd, "a short audio track must never cut the picture"


def test_a_track_without_cues_is_silence_for_the_length_of_the_film(tmp_path):
    fc = _graph(_mixed(tmp_path, cues=[c for c in CUES if c["track"] == "music"]))
    assert "anullsrc=r=48000:cl=stereo,atrim=duration=7.000[speech]" in fc
    assert "anullsrc=r=48000:cl=stereo,atrim=duration=7.000[loc]" in fc
    assert "amix=inputs=3:normalize=0" in fc


def test_without_cues_the_command_is_the_silent_draft_byte_for_byte(monkeypatch, tmp_path):
    monkeypatch.setattr(render, "has_drawtext", lambda **kw: False)
    out = tmp_path / "o.mp4"
    cmd = render.build_command(ROWS, sources=SRC, out_path=out, cues=(), levels=LEVELS)
    assert cmd == ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                   "-ss", "5.000", "-t", "3.000", "-i", "/m/a.mp4",
                   "-ss", "0.000", "-t", "4.000", "-i", "/m/b.mp4",
                   "-filter_complex_script", str(tmp_path / "o.filters"), "-map", "[vout]",
                   "-r", "30", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                   # the render publishes itself by renaming this on exit 0
                   "-pix_fmt", "yuv420p", str(render.part_path(out))]
    leg = ("fps=30,setpts=PTS-STARTPTS,scale=960:540:force_original_aspect_ratio=decrease"
           ":force_divisible_by=2,pad=960:540:(ow-iw)/2:(oh-ih)/2:black,setsar=1")
    assert _graph(cmd) == f"[0:v]{leg}[v0];[1:v]{leg}[v1];[v0][v1]concat=n=2:v=1:a=0[vout]"


def test_the_graph_can_be_written_where_the_caller_says(tmp_path):
    script = tmp_path / "elsewhere" / "graph.txt"
    script.parent.mkdir()
    cmd = render.build_command(ROWS, sources=SRC, out_path=tmp_path / "o.mp4", script_path=script)
    assert cmd[cmd.index("-filter_complex_script") + 1] == str(script)
    assert "concat=n=2" in script.read_text()
    assert "concat=n=2" in render.describe(cmd), "describe still inlines the graph from wherever it went"


def test_a_cue_on_an_unknown_track_or_without_a_source_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        _mixed(tmp_path, cues=[CUES[0] | {"track": "ambience"}])
    with pytest.raises(KeyError):
        _mixed(tmp_path, audio_sources={})


def test_the_measuring_pass_is_audio_only_and_prints_its_numbers(tmp_path):
    """The final normalisation is two-pass. The first pass hears the mix
    and nothing else: no video inputs, no picture chain, no encode, the
    cue inputs indexed from zero, loudnorm printing its measurement --
    which it does at INFO, so this pass cannot run at -loglevel error."""
    cmd = _mixed(tmp_path, measure_only=True)
    assert _inputs(cmd) == ["/w/rec_a.wav", "/w/rec_a.wav", "/w/rec_b.wav", "/mu/a.mp3", "/mu/b.mp3"]
    assert cmd[cmd.index("-loglevel") + 1] == "info"
    assert cmd[-3:] == ["-f", "null", "-"] and "-vn" in cmd
    assert cmd.count("-map") == 1 and cmd[cmd.index("-map") + 1] == "[aout]"
    assert "-c:v" not in cmd and "-c:a" not in cmd
    fc = _graph(cmd)
    assert "[vout]" not in fc and ":v=1:a=0" not in fc, \
        "no picture -- the only concat left is the location track's, which is audio"
    assert "[0:a]loudnorm=I=-16" in fc, "the first cue reads the first input"
    assert fc.endswith("loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json[aout]")


def test_input_indices_agree_between_the_two_passes(tmp_path):
    """Both passes read the same cue files in the same order; only the
    offset differs, by exactly the number of video inputs."""
    render_cmd, measure_cmd = _mixed(tmp_path), _mixed(tmp_path, measure_only=True)
    assert _inputs(render_cmd)[2:] == _inputs(measure_cmd)
    assert _apads(measure_cmd) == list(range(5))
    assert _apads(render_cmd) == list(range(2, 7))


def test_cue_pads_start_after_a_split_secondary(tmp_path):
    """A split adds an input the rows do not count; the cue pads follow it."""
    rows = [ROWS[0], SPLIT | {"t_in": 3.0, "t_out": 6.0}, ROWS[1] | {"t_in": 6.0, "t_out": 10.0}]
    cmd = render.build_command(rows, sources=BOTH | SRC, out_path=tmp_path / "o.mp4", cues=CUES,
                               audio_sources=AUDIO, music_sources=MUSIC,
                               envelopes=ENVELOPES, levels=LEVELS)
    assert _inputs(cmd)[:4] == ["/m/a.mp4", "/m/a.mp4", "/m/z.mp4", "/m/b.mp4"]
    assert _apads(cmd) == list(range(4, 9))
    assert "[4:a]loudnorm=I=-16" in _graph(cmd)


def test_a_measured_render_applies_one_static_gain(tmp_path):
    """With the numbers from the measuring pass the final stage is what
    loudnorm's linear mode is made of -- one gain to the target, then a
    true-peak limiter -- and nothing else, so the envelope's dynamics
    survive. loudnorm's own linear=true carries a precondition the film
    cannot meet (measured_LRA <= LRA, an option that stops at 20 LU) and
    reverts to the dynamic re-levelling this pass exists to avoid."""
    measured = {"input_i": -10.63, "input_lra": 1.1, "input_tp": -3.68, "input_thresh": -20.68}
    cmd = _mixed(tmp_path, loudnorm_measured=measured)
    fc = _graph(cmd)
    assert fc.endswith("amix=inputs=3:normalize=0,"
                       "volume=-3.37dB,alimiter=limit=0.841395:level=false[aout]")
    assert "loudnorm" not in fc.split("amix=inputs=3")[-1], "no loudnorm left in the final stage"
    assert "[vout]" in fc and "-c:v" in cmd and cmd[-1].endswith("o.part.mp4"), "a full render otherwise"
    # No range precondition left: a mix far wider than the LRA option could
    # ever ask for is normalised in exactly the same way.
    assert _graph(_mixed(tmp_path, loudnorm_measured=measured | {"input_lra": 24.3})) == fc
    assert _graph(_mixed(tmp_path)).endswith("LRA=11[aout]"), "without numbers: the single pass, unchanged"


def test_measuring_without_cues_is_an_error():
    with pytest.raises(ValueError):
        render.build_command(ROWS, sources=SRC, out_path=pathlib.Path("/o.mp4"), measure_only=True)


LOUDNORM_STDERR = """Input #0, wav, from '/tmp/speech.wav':
  Duration: 00:00:05.00, bitrate: 705 kb/s
[Parsed_loudnorm_9 @ 0x55d0c0a3b2c0]
{
\t"input_i" : "-10.63",
\t"input_tp" : "-3.68",
\t"input_lra" : "1.10",
\t"input_thresh" : "-20.68",
\t"output_i" : "-14.05",
\t"output_tp" : "-7.10",
\t"output_lra" : "1.10",
\t"output_thresh" : "-24.10",
\t"normalization_type" : "dynamic",
\t"target_offset" : "0.05"
}
"""


def test_parse_loudnorm_json_reads_the_block_ffmpeg_prints():
    got = render.parse_loudnorm_json(LOUDNORM_STDERR)
    assert got == {"input_i": -10.63, "input_lra": 1.1, "input_tp": -3.68, "input_thresh": -20.68}
    assert all(isinstance(v, float) for v in got.values())


def test_parse_loudnorm_json_names_what_is_missing():
    with pytest.raises(ValueError, match="no loudnorm"):
        render.parse_loudnorm_json("Input #0, wav\nsize=N/A time=00:00:12.00\n")
    with pytest.raises(ValueError, match="does not parse"):
        render.parse_loudnorm_json("[Parsed_loudnorm_9 @ 0x1]\n{not json}\n")
    with pytest.raises(ValueError, match="input_tp"):
        render.parse_loudnorm_json(LOUDNORM_STDERR.replace('"input_tp" : "-3.68",\n', ""))
    with pytest.raises(ValueError, match="input_i"):
        # a silent mix measures -inf, which loudnorm would refuse at graph init
        render.parse_loudnorm_json(LOUDNORM_STDERR.replace('"input_i" : "-10.63"', '"input_i" : "-inf"'))


def _integrated(path):
    """The rendered file's own integrated loudness, heard the way the
    measuring pass hears the mix -- loudnorm printing what it measures."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-nostats", "-i", str(path), "-vn",
         "-af", "loudnorm=print_format=json", "-f", "null", "-"],
        capture_output=True, text=True, check=True)
    return render.parse_loudnorm_json(proc.stderr)["input_i"]


@pytest.mark.slow
def test_ffmpeg_mixes_speech_over_music_and_keeps_the_silence_window(tmp_path):
    """At the boundary: a 14 s cut; a 440 Hz "speech" cue at 2-5 s over a
    bed in three cues -- 220 Hz at 0-5 s and 330 Hz at 5-7 s abutting,
    220 Hz again at 10-14 s after a silence window, long enough that the
    end-of-film fade the clamp pulls inside it leaves a flat stretch --
    with the envelope flat at 1 and muting 7-9 s. Both passes run. The
    levels are read off
    the two-pass render (one static gain, so they are the graph's numbers,
    not loudnorm's dynamics): the speech at least 3 dB over the bed alone,
    the join at 5-7 s within 3 dB of it (a crossfade is not a dip, and the
    two cues differ in pitch so nothing sums by coincidence), the cue
    after the window within 1.5 dB of it past its own half-second fade-in
    (a cue after a gap lands where the table says), the muted stretch
    silent although the outgoing cue's tail runs on under it, and 9-10 s
    silent because nothing is scheduled there. The envelopes are the flat
    between(t,..)*g sums mix.volume_expr writes, quoted inside the script
    file -- the proof that ffmpeg parses them that way."""
    if not shutil.which("ffmpeg"):
        pytest.skip("needs ffmpeg")

    def lavfi(src, path, *extra):
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-f", "lavfi", "-i", src, *extra, str(path)], check=True)
    vid, speech = tmp_path / "v.mp4", tmp_path / "speech.wav"
    low, mid = tmp_path / "low.wav", tmp_path / "mid.wav"
    lavfi("testsrc=size=320x240:rate=25:duration=15", vid, "-c:v", "libx264", "-pix_fmt", "yuv420p")
    lavfi("sine=frequency=440:duration=5", speech)
    lavfi("sine=frequency=220:duration=14", low, "-ac", "2")
    lavfi("sine=frequency=330:duration=14", mid, "-ac", "2")
    rows = [{"shot_id": "v1", "t_in": 0.0, "t_out": 7.0, "src_in": 0.0},
            {"shot_id": "v2", "t_in": 7.0, "t_out": 14.0, "src_in": 7.0}]
    cues = [_cue(cue_id="sp", track="speech", t_in=2.0, t_out=5.0, source="r",
                 src_in=0.0, src_out=3.0, gain_lufs=-16.0)]
    for k, (t0, t1, src, fade_in) in enumerate(((0.0, 5.0, "low", 0.0), (5.0, 7.0, "mid", 2.0),
                                                (10.0, 14.0, "low", 0.5))):
        cues.append(_cue(cue_id=f"mu_{k}", track="music", t_in=t0, t_out=t1, source=src,
                         src_in=t0, src_out=t1, gain_lufs=-14.0, fade_in_s=fade_in, fade_out_s=2.0))
    envelopes = {"music": "between(t,0,7)*1+between(t,9,14)*1",
                 "location": "between(t,0,14)*1"}
    kw = dict(sources={"v1": vid, "v2": vid}, cues=cues, audio_sources={"r": speech},
              music_sources={"low": low, "mid": mid}, envelopes=envelopes, levels=LEVELS)
    out = tmp_path / "draft.mp4"
    assert render.run_render(render.build_command(rows, out_path=out, **kw), out).returncode == 0
    dur = float(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=duration", "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, check=True).stdout.strip())
    assert 13.5 <= dur <= 14.1, f"the mix runs the film's length and not past it, got {dur}"

    # Two-pass: hear the mix audio-only, then render with its numbers.
    measure = subprocess.run(render.build_command(rows, out_path=out, measure_only=True, **kw),
                             capture_output=True, text=True, check=True)
    measured = render.parse_loudnorm_json(measure.stderr)
    assert measured["input_i"] < 0 and measured["input_tp"] < 0
    out2 = tmp_path / "draft2.mp4"
    assert render.run_render(
        render.build_command(rows, out_path=out2, loudnorm_measured=measured, **kw), out2).returncode == 0

    def probe(a, b):
        return render.probe_loudness(out2, t_in=a, t_out=b)
    alone = probe(0.0, 2.0)
    assert probe(2.0, 5.0) - alone >= 3.0, f"the speech must be heard over the bed: {probe(2.0, 5.0)} dB against {alone} dB"
    join = probe(5.0, 7.0)
    assert abs(join - alone) < 3.0, f"across the join reads {join} dB against {alone} dB for the bed alone"
    # The envelope is evaluated once per audio frame, so its edges land
    # within a frame of 7 s and 9 s; the probes stay a quarter second inside.
    assert probe(7.25, 8.75) < -50.0, "the outgoing tail is muted"
    assert probe(9.1, 9.9) < -50.0, "nothing plays before the late cue's t_in"
    late = probe(10.5, 12.0)
    assert abs(late - alone) < 1.5, f"the cue after the window reads {late} dB against {alone} dB"

    # Where the mix ends up: one gain from what the first pass measured to
    # the target, so the file measures the target back.
    got = _integrated(out2)
    assert abs(got - LEVELS["final_lufs"]) <= 1.0, \
        f"the draft measures {got} LUFS, asked for {LEVELS['final_lufs']}"

    # And the film's own range, which no fixture this size reaches: the
    # first end-to-end mix measured 20.1 LU, past what loudnorm's LRA option
    # can even be given, so linear mode reverted to dynamic and re-levelled
    # the envelope. A gain and a limiter have no range to refuse: the same
    # mix declared far wider must render AND hold the level the envelope
    # designed. Asked at the real boundary, because that is where it was
    # wrong.
    out3 = tmp_path / "draft3.mp4"
    assert render.run_render(
        render.build_command(rows, out_path=out3,
                             loudnorm_measured=measured | {"input_lra": 24.3}, **kw),
        out3).returncode == 0
    wide_alone = render.probe_loudness(out3, t_in=0.0, t_out=2.0)
    wide_late = render.probe_loudness(out3, t_in=10.5, t_out=12.0)
    assert abs(wide_late - wide_alone) < 1.5, (
        f"a mix wider than the LRA option keeps its envelope: the cue after the window reads "
        f"{wide_late} dB against {wide_alone} dB for the bed alone")


@pytest.mark.slow
def test_the_concatenated_location_track_sounds_like_the_summed_one(tmp_path, monkeypatch):
    """The equivalence, at the boundary. Three location cues -- two
    abutting at two different levels, then two seconds of dead air, then a
    third -- rendered once laid end to end and once summed, and heard
    window by window.

    Only the tiling tolerance is monkeypatched to force the fallback, so
    both renders are given the same cues, the same gains, the same fades
    and the same placement; anything concat did differently to any of the
    three would move a window or change its level. The final stage is the
    static gain of a measured render rather than loudnorm's dynamics,
    which would react to each mix separately and hide exactly that.
    """
    if not shutil.which("ffmpeg"):
        pytest.skip("needs ffmpeg")

    def lavfi(src, path, *extra):
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-f", "lavfi", "-i", src, *extra, str(path)], check=True)
    vid, low, mid = tmp_path / "v.mp4", tmp_path / "low.wav", tmp_path / "mid.wav"
    lavfi("testsrc=size=320x240:rate=25:duration=14", vid, "-c:v", "libx264", "-pix_fmt", "yuv420p")
    # Two pitches, so a cue landing in the wrong window cannot pass for the
    # one that belongs there by coincidence.
    lavfi("sine=frequency=220:duration=14", low, "-ac", "2")
    lavfi("sine=frequency=330:duration=14", mid, "-ac", "2")
    rows = [{"shot_id": "v1", "t_in": 0.0, "t_out": 7.0, "src_in": 0.0},
            {"shot_id": "v2", "t_in": 7.0, "t_out": 13.0, "src_in": 7.0}]
    cues = [_cue(cue_id="lo_0", track="location", t_in=0.0, t_out=4.0, source="low",
                 src_in=0.0, src_out=4.0, gain_lufs=-18.0),
            _cue(cue_id="lo_1", track="location", t_in=4.0, t_out=7.0, source="mid",
                 src_in=4.0, src_out=7.0, gain_lufs=-24.0),
            _cue(cue_id="lo_2", track="location", t_in=9.0, t_out=13.0, source="low",
                 src_in=9.0, src_out=13.0, gain_lufs=-18.0)]
    # A measurement this mix could plausibly have produced: one gain of
    # -11 dB, far enough under the limiter that nothing is compressed and
    # the two renders differ in nothing but how the track was assembled.
    measured = {"input_i": -3.0, "input_lra": 2.0, "input_tp": -0.5, "input_thresh": -13.0}
    kw = dict(sources={"v1": vid, "v2": vid}, cues=cues, audio_sources={"low": low, "mid": mid},
              music_sources={}, levels=LEVELS, loudnorm_measured=measured,
              envelopes={"location": "between(t,0,13)*1", "music": "between(t,0,13)*1"})
    joined, summed = tmp_path / "joined.mp4", tmp_path / "summed.mp4"
    cmd = render.build_command(rows, out_path=joined, **kw)
    assert "concat=n=4:v=0:a=1" in _graph(cmd), "two cues, the dead air between them, and the third"
    assert render.run_render(cmd, joined).returncode == 0

    # A tolerance no abutting pair can satisfy: every join now reads as an
    # overlap, so the same cues take the amix path untouched.
    monkeypatch.setattr(render, "EDGE_TOL_S", -1.0)
    cmd = render.build_command(rows, out_path=summed, **kw)
    assert "[lo0][lo1][lo2]amix=inputs=3:normalize=0" in _graph(cmd)
    assert render.run_render(cmd, summed).returncode == 0

    windows = {"the first cue": (0.5, 3.5), "the second": (4.5, 6.5),
               "the dead air": (7.3, 8.7), "the cue after it": (9.5, 12.5)}
    heard = {}
    for what, (a, b) in windows.items():
        heard[what] = x = render.probe_loudness(joined, t_in=a, t_out=b)
        y = render.probe_loudness(summed, t_in=a, t_out=b)
        assert abs(x - y) <= 0.5, f"{what} reads {x} dB laid end to end and {y} dB summed"

    # And that the windows are worth comparing: the second cue's row puts
    # it 6 dB under the other two, and the gap is the silence the timeline
    # asked for rather than the previous cue running on into it.
    assert abs((heard["the first cue"] - heard["the second"]) - 6.0) <= 1.0, heard
    assert abs(heard["the cue after it"] - heard["the first cue"]) <= 1.0, heard
    assert heard["the dead air"] < -50.0, heard


# -- the draft is published only once it is whole -----------------------

def test_the_render_writes_a_part_file_and_renames_it_on_success(tmp_path):
    """The rule this pipeline already has for anything measured in hours,
    now for the one output that is: the draft appears at its name only
    after ffmpeg has exited 0."""
    out = tmp_path / "draft.mp4"
    cmd = render.build_command(ROWS, sources=SRC, out_path=out)
    # ffmpeg picks its muxer from the extension, so the temporary keeps one
    assert cmd[-1] == str(tmp_path / "draft.part.mp4")
    assert render.part_path(out).suffix == ".mp4"

    render.part_path(out).write_bytes(b"rendered")
    proc = render.run_render(["true"], out)
    assert proc.returncode == 0
    assert out.read_bytes() == b"rendered" and not render.part_path(out).exists()


def test_a_failed_render_leaves_the_previous_draft_alone(tmp_path):
    out = tmp_path / "draft.mp4"
    out.write_bytes(b"last good draft")
    render.part_path(out).write_bytes(b"half a render")
    assert render.run_render(["false"], out).returncode != 0
    assert out.read_bytes() == b"last good draft"


@pytest.mark.slow
def test_a_location_cue_with_nothing_left_to_play_still_renders_the_whole_film(tmp_path):
    """The real corpus, 2026-09-25: the ambience held on under a still ran
    past the end of the recording it was held from, so that cue's input was
    seeked to the wav's own length and decoded no samples at all. apad still
    padded the piece to its span, but with frames carrying no pts, and
    concat folded that into the delta it shifts every later piece by: the
    graph died at "Invalid data found when processing input" and ffmpeg
    wrote a valid trailer and exited 0 on 138.7 s of a 2145.9 s draft.

    Run against real ffmpeg because that is where it went wrong: a 1 s cue,
    a 6.5 s cue whose source has nothing left, a 1 s cue after it, and the
    film has to come out 8.5 s long on both streams. (ffmpeg prints
    "Invalid value NaN for volume" once per eval=frame volume filter
    whatever the inputs do -- af_volume seeds every variable NaN and
    evaluates once at init, before any frame has a t. It is not a symptom
    of this and it does not go away.)"""
    if not shutil.which("ffmpeg"):
        pytest.skip("needs ffmpeg")

    def lavfi(src, path, *extra):
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-f", "lavfi", "-i", src, *extra, str(path)], check=True)
    vid, amb = tmp_path / "v.mp4", tmp_path / "amb.wav"
    lavfi("testsrc=size=320x240:rate=25:duration=10", vid, "-c:v", "libx264", "-pix_fmt", "yuv420p")
    lavfi("sine=frequency=220:duration=2", amb, "-ac", "2")
    rows = [{"shot_id": "v1", "t_in": 0.0, "t_out": 8.5, "src_in": 0.0}]
    cues = [_cue(cue_id="lo_0", track="location", t_in=0.0, t_out=1.0, source="amb",
                 src_in=0.0, src_out=1.0, gain_lufs=-18.0),
            # Seeked to exactly the wav's length: nothing decodes at all.
            _cue(cue_id="lo_1", track="location", t_in=1.0, t_out=7.5, source="amb",
                 src_in=2.0, src_out=8.5, gain_lufs=-18.0),
            _cue(cue_id="lo_2", track="location", t_in=7.5, t_out=8.5, source="amb",
                 src_in=0.5, src_out=1.5, gain_lufs=-18.0)]
    out = tmp_path / "draft.mp4"
    cmd = render.build_command(rows, sources={"v1": vid}, out_path=out, cues=cues,
                               audio_sources={"amb": amb}, levels=LEVELS,
                               envelopes={"location": "between(t,0,8.5)*1", "music": "between(t,0,8.5)*1"})
    proc = render.run_render(cmd, out, expect_s=8.5, tol_s=0.3)
    assert proc.returncode == 0, proc.stderr[-800:]
    assert out.exists()

    def stream_s(kind):
        return float(subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", kind, "-show_entries",
             "stream=duration", "-of", "csv=p=0", str(out)],
            capture_output=True, text=True, check=True).stdout.strip())
    assert stream_s("v:0") == pytest.approx(8.5, abs=0.2), "the picture stopped short"
    assert stream_s("a:0") == pytest.approx(8.5, abs=0.2), "the mix stopped short"


@pytest.mark.slow
def test_a_render_that_stopped_part_way_is_never_published(tmp_path):
    """ffmpeg's exit code cannot tell a finished film from one that stopped:
    it breaks out of its transcode loop on a mid-graph error, flushes, writes
    a valid trailer and returns 0. The length is the verdict it will not
    give, so a file short of the timeline stays at its temporary name and
    comes back as the failure it is -- checked against real ffmpeg output
    rather than a hand-written stub, because the number being read is
    ffprobe's."""
    if not shutil.which("ffmpeg"):
        pytest.skip("needs ffmpeg")
    src = tmp_path / "s.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", "testsrc=size=320x240:rate=25:duration=4", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(src)], check=True)
    rows = [{"shot_id": "v1", "t_in": 0.0, "t_out": 2.0, "src_in": 0.0}]
    out = tmp_path / "draft.mp4"
    out.write_bytes(b"last good draft")
    cmd = render.build_command(rows, sources={"v1": src}, out_path=out)

    proc = render.run_render(cmd, out, expect_s=20.0, tol_s=0.5)
    assert proc.returncode != 0, "a two-second file is not a twenty-second film"
    assert "2.000 s" in proc.stderr and "20.000 s" in proc.stderr, proc.stderr[-400:]
    assert out.read_bytes() == b"last good draft", "the last good draft must survive"
    assert render.part_path(out).exists(), "the short file is kept, to look at where it stopped"

    # The same render, asked for the length it actually is, publishes.
    assert render.run_render(cmd, out, expect_s=2.0, tol_s=0.5).returncode == 0
    assert out.read_bytes() != b"last good draft" and not render.part_path(out).exists()

    # And a draft LONGER than its timeline publishes too: every leg rounds up
    # to a whole frame at the draft's rate, so a finished film always runs a
    # shade over -- 2151.433 s against a 2145.944 s timeline on the real cut.
    # Refusing that would fail every render there is.
    assert render.run_render(cmd, out, expect_s=1.0, tol_s=0.5).returncode == 0
    assert out.exists() and not render.part_path(out).exists()


def test_the_edge_tolerance_is_the_one_cues_uses(tmp_path):
    """Two modules answering "do these times coincide?" differently would
    put a gap in the mix that the picture does not have. The value is
    copied rather than imported across the module boundary, so the only
    thing stopping it drifting is this assertion."""
    from nepal.process import cues
    assert render.EDGE_TOL_S == cues._EDGE_TOL_S
