import sys, pathlib, shutil
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest

from nepal.process.reproject import (normalise_yaw, build_360_graph, build_flat_graph,
                                     plan, build_command, pick_source,
                                     detect_hwaccel, detect_encoder, reset_detection_cache)


# -- yaw normalisation -------------------------------------------------

@pytest.mark.parametrize("given,expected", [
    (0, 0.0), (90, 90.0), (180, 180.0), (270, -90.0), (360, 0.0),
    (-90, -90.0), (450, 90.0), (181, -179.0),
])
def test_normalise_yaw(given, expected):
    assert normalise_yaw(given) == expected


def test_every_normalised_yaw_is_in_ffmpeg_range():
    """ffmpeg's v360 rejects yaw outside [-180, 180] outright. The spec's command
    uses yaw=270, which fails rather than degrading."""
    for deg in range(-720, 721, 7):
        assert -180.0 <= normalise_yaw(deg) <= 180.0


# -- filter graph ------------------------------------------------------

def test_split_count_matches_consumers():
    """The spec's split=6 leaves an [aud] output unconnected, and ffmpeg refuses
    a filter with an unconnected output. It must equal 1 + len(yaws)."""
    fc, labels = build_360_graph(196, yaws=(0, 90, 180, 270))
    assert fc.startswith("[0:v]split=5[eq][v0][v1][v2][v3];")
    assert len(labels) == 5


@pytest.mark.parametrize("yaws", [(0,), (0, 180), (0, 90, 180, 270), (0, 60, 120, 180, 240, 300)])
def test_split_count_tracks_yaw_count(yaws):
    fc, labels = build_360_graph(193, yaws=yaws)
    assert f"split={1 + len(yaws)}" in fc
    assert len(labels) == 1 + len(yaws)
    # every declared split label is consumed exactly once
    for i in range(len(yaws)):
        assert fc.count(f"[v{i}]") == 2      # declared, then used


def test_graph_uses_the_signed_yaw_but_labels_by_compass():
    fc, labels = build_360_graph(196, yaws=(0, 90, 180, 270))
    assert "yaw=-90" in fc, "270 must be emitted as -90"
    assert "yaw=270" not in fc
    assert "y270" in labels, "the label stays in compass terms for the data model"


def test_graph_chains_from_the_split_with_no_intermediate_file():
    """Going via an intermediate equirect file would double the decode."""
    fc, _ = build_360_graph(196)
    for branch in fc.split(";")[1:]:
        assert branch.lstrip().startswith("["), branch
        assert "v360=input=dfisheye" in branch


def test_fov_appears_on_every_branch():
    fc, _ = build_360_graph(201.5)
    assert fc.count("ih_fov=201.5") == 5
    assert fc.count("iv_fov=201.5") == 5


def test_sizes_are_honoured():
    fc, _ = build_360_graph(196, proxy_size=(2048, 1024), view_size=(1280, 720))
    assert "scale=2048:1024" in fc
    assert fc.count("scale=1280:720") == 4


def test_flat_graph_skips_v360_entirely():
    fc, labels = build_flat_graph(proxy_size=(960, 540))
    assert "v360" not in fc
    assert fc == "[0:v]scale=960:540[eqout]"
    assert labels == ["eqout"]


# -- plan and command --------------------------------------------------

def test_plan_defaults_to_phase_a_proxy_only(tmp_path):
    """The default is the two-phase design: no yaw videos, since nothing
    consumes them as video until S07 conforms the selected shots."""
    p = plan(tmp_path / "src.lrv", "rec_1", tmp_path / "work", is_360=True, fov_deg=196)
    assert [path.name for _, path, _ in p.outputs] == ["rec_1_eq.mp4"]
    assert p.yaws == ()
    assert p.audio_path.name == "rec_1.wav"
    assert "split" not in p.filter_complex, "a split with unused outputs fails the graph"


def test_plan_names_outputs_by_recording_and_yaw(tmp_path):
    """The spec's single-pass design, still reachable with yaw_videos=True."""
    p = plan(tmp_path / "src.lrv", "rec_1", tmp_path / "work", is_360=True,
             fov_deg=196, yaw_videos=True)
    names = [path.name for _, path, _ in p.outputs]
    assert names == ["rec_1_eq.mp4", "rec_1_y0.mp4", "rec_1_y90.mp4",
                     "rec_1_y180.mp4", "rec_1_y270.mp4"]
    assert p.audio_path.name == "rec_1.wav"


def test_plan_for_flat_source_has_one_output(tmp_path):
    p = plan(tmp_path / "src.mp4", "rec_2", tmp_path / "work", is_360=False)
    assert len(p.outputs) == 1
    assert p.yaws == ()


def test_command_maps_every_output(tmp_path):
    p = plan(tmp_path / "s.lrv", "r", tmp_path / "w", is_360=True, fov_deg=196,
             yaw_videos=True)
    cmd = build_command(p, encoder="libx264")
    for label, path, _ in p.outputs:
        assert f"[{label}]" in cmd
        assert str(path) in cmd
    assert str(p.audio_path) in cmd
    assert "pcm_s16le" in cmd and "16000" in cmd


def test_command_omits_audio_when_the_source_is_silent(tmp_path):
    p = plan(tmp_path / "s.lrv", "r", tmp_path / "w", is_360=True, yaw_videos=True)
    cmd = build_command(p, encoder="libx264", has_audio=False)
    assert "0:a:0" not in cmd
    assert str(p.audio_path) not in cmd


def test_command_omits_hwaccel_when_none_available(tmp_path):
    p = plan(tmp_path / "s.lrv", "r", tmp_path / "w", is_360=True, yaw_videos=True)
    assert "-hwaccel" not in build_command(p, hwaccel=None)
    assert "-hwaccel" in build_command(p, hwaccel="cuda")


# -- source selection --------------------------------------------------

def _asset(container, name, kind="video360", chapter=1):
    return {"s3_key": f"raw/media_from_camera/{name}", "container": container,
            "kind": kind, "chapter_index": chapter}


def test_prefers_the_lrv_proxy_over_the_insv_original(tmp_path):
    d = tmp_path / "media_from_camera"
    d.mkdir(parents=True)
    (d / "a.insv").write_bytes(b"x")
    (d / "a.lrv").write_bytes(b"x")
    got = pick_source([_asset("insv", "a.insv"), _asset("lrv", "a.lrv")], tmp_path)
    assert got[0].name == "a.lrv", "decoding 5.7K H.265 for 540p output is wasted work"
    assert got[1] is True


def test_falls_back_to_the_original_when_no_proxy_exists(tmp_path):
    d = tmp_path / "media_from_camera"
    d.mkdir(parents=True)
    (d / "a.insv").write_bytes(b"x")
    got = pick_source([_asset("insv", "a.insv")], tmp_path)
    assert got[0].name == "a.insv"


def test_skips_assets_whose_files_are_absent(tmp_path):
    d = tmp_path / "media_from_camera"
    d.mkdir(parents=True)
    (d / "b.insv").write_bytes(b"x")
    got = pick_source([_asset("lrv", "gone.lrv"), _asset("insv", "b.insv")], tmp_path)
    assert got[0].name == "b.insv"


def test_returns_none_when_nothing_is_readable(tmp_path):
    assert pick_source([_asset("lrv", "gone.lrv")], tmp_path) is None


def test_picks_the_lowest_chapter_first(tmp_path):
    d = tmp_path / "media_from_camera"
    d.mkdir(parents=True)
    for n in ("c_002.lrv", "c_001.lrv"):
        (d / n).write_bytes(b"x")
    got = pick_source([_asset("lrv", "c_002.lrv", chapter=2),
                       _asset("lrv", "c_001.lrv", chapter=1)], tmp_path)
    assert got[0].name == "c_001.lrv"


def test_flat_source_reports_not_360(tmp_path):
    d = tmp_path / "media_from_camera"
    d.mkdir(parents=True)
    (d / "f.mp4").write_bytes(b"x")
    got = pick_source([_asset("mp4", "f.mp4", kind="video_flat")], tmp_path)
    assert got[1] is False


# -- capability detection ----------------------------------------------

@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_detection_probes_rather_than_trusting_the_list():
    """ffmpeg advertises every accelerator compiled into the binary whether or
    not the hardware exists, so a CPU-only machine claims cuda and h264_nvenc
    and then fails at runtime. Detection must be functional."""
    reset_detection_cache()
    enc = detect_encoder()
    assert enc in ("h264_nvenc", "h264_videotoolbox", "h264_qsv", "libx264")
    # whatever was chosen must actually encode
    from nepal.util import proc
    r = proc.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                  "-f", "lavfi", "-i", "testsrc2=size=128x128:rate=1:duration=1",
                  "-c:v", enc, "-frames:v", "1", "-f", "null", "-"], check=False)
    assert r.returncode == 0, f"{enc} was chosen but does not work"


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_hwaccel_detection_is_cached():
    reset_detection_cache()
    first = detect_hwaccel()
    assert detect_hwaccel() == first


# -- phase B stills ----------------------------------------------------

from nepal.process.reproject import (build_proxy_only_graph, yaw_still_command,
                                     flat_still_command)


def test_proxy_only_graph_has_no_split():
    fc, labels = build_proxy_only_graph(196)
    assert "split" not in fc
    assert labels == ["eqout"]
    assert "v360=input=dfisheye:output=e" in fc


def test_yaw_still_uses_the_signed_yaw(tmp_path):
    cmd = yaw_still_command(tmp_path / "s.lrv", 4.0, 270, tmp_path / "f.jpg", fov_deg=196)
    joined = " ".join(cmd)
    assert "yaw=-90" in joined and "yaw=270" not in joined


def test_yaw_still_reads_from_the_source_not_a_proxy(tmp_path):
    """Sampling the original avoids a generation of H.264 loss before face
    embedding and CLIP."""
    src = tmp_path / "original.insv"
    cmd = yaw_still_command(src, 4.0, 0, tmp_path / "f.jpg", fov_deg=196)
    assert str(src) in cmd
    assert "-ss" in cmd and "4.000" in cmd
    assert cmd[cmd.index("-frames:v") + 1] == "1"


def test_flat_still_skips_reprojection(tmp_path):
    cmd = flat_still_command(tmp_path / "s.mp4", 2.0, tmp_path / "f.jpg")
    assert "v360" not in " ".join(cmd)
    assert "scale=960:540" in " ".join(cmd)


# -- every chapter, not just the first ----------------------------------

def test_all_chapters_are_returned_in_order(tmp_path):
    """82 of this corpus's 112 recordings are chapter-split. Proxying only
    chapter one would silently drop the rest of the take."""
    from nepal.process.reproject import pick_sources
    assets = []
    for i in (3, 1, 2):
        rel = f"media_from_camera/VID_00_{i:03d}.insv"
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"x")
        assets.append({"s3_key": f"raw/{rel}", "container": "insv",
                       "kind": "video360", "chapter_index": i})
    paths, is_360 = pick_sources(assets, tmp_path)
    assert [p.name for p in paths] == ["VID_00_001.insv", "VID_00_002.insv",
                                       "VID_00_003.insv"]
    assert is_360


def test_a_proxy_container_still_wins_over_the_original(tmp_path):
    from nepal.process.reproject import pick_sources
    assets = []
    for rel, container in (("a.insv", "insv"), ("a.lrv", "lrv")):
        (tmp_path / rel).write_bytes(b"x")
        assets.append({"s3_key": f"raw/{rel}", "container": container,
                       "kind": "video360", "chapter_index": 1})
    paths, _ = pick_sources(assets, tmp_path)
    assert [p.name for p in paths] == ["a.lrv"]


def test_missing_files_are_skipped_not_returned(tmp_path):
    from nepal.process.reproject import pick_sources
    (tmp_path / "there.mp4").write_bytes(b"x")
    assets = [{"s3_key": "raw/there.mp4", "container": "mp4",
               "kind": "video_flat", "chapter_index": 1},
              {"s3_key": "raw/gone.mp4", "container": "mp4",
               "kind": "video_flat", "chapter_index": 2}]
    paths, _ = pick_sources(assets, tmp_path)
    assert [p.name for p in paths] == ["there.mp4"]


def test_nothing_readable_returns_none(tmp_path):
    from nepal.process.reproject import pick_sources
    assert pick_sources([{"s3_key": "raw/x.mp4", "container": "mp4",
                          "kind": "video_flat"}], tmp_path) is None


def test_the_concat_list_is_an_ffconcat_file(tmp_path):
    from nepal.process.reproject import concat_list
    paths = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    for p in paths:
        p.write_bytes(b"x")
    text = concat_list(paths, tmp_path / "l.ffconcat").read_text()
    assert text.splitlines()[0] == "ffconcat version 1.0"
    assert text.count("file '") == 2
    assert "a.mp4" in text and "b.mp4" in text


def test_an_apostrophe_in_a_path_is_escaped(tmp_path):
    """Unescaped, it would close the quote and truncate the list."""
    from nepal.process.reproject import concat_list
    d = tmp_path / "kirill's photos"
    d.mkdir()
    p = d / "a.mp4"
    p.write_bytes(b"x")
    line = [ln for ln in concat_list([p], tmp_path / "l.ffconcat").read_text().splitlines()
            if ln.startswith("file ")][0]
    assert line.endswith("'")
    assert "'\\''" in line


def test_the_concat_input_flags_go_before_the_input():
    """-f concat after -i is ignored, and ffmpeg would read the list as video."""
    from nepal.process.reproject import plan, build_command
    p = plan(pathlib.Path("list.ffconcat"), "r1", pathlib.Path("/tmp/w"),
             is_360=False)
    cmd = build_command(p, extra_input=["-f", "concat", "-safe", "0"])
    assert cmd.index("-f") < cmd.index("-i")
    assert cmd[cmd.index("-i") + 1] == "list.ffconcat"
