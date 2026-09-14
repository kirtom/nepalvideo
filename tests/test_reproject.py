import sys, pathlib, shutil
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest

from nepal.process.reproject import (normalise_yaw, build_360_graph, build_flat_graph,
                                     plan, build_command,
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


# -- three formats, three treatments ------------------------------------

def _shaped(name, container, w, h, shape, chapter=1):
    return {"s3_key": f"raw/c/{name}", "container": container, "width": w,
            "height": h, "frame_shape": shape, "chapter_index": chapter,
            "kind": "video_flat" if shape == "flat" else "video360"}


def _on_disk(tmp_path, assets):
    for a in assets:
        f = tmp_path / a["s3_key"][4:]
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x")
    return assets


def test_a_dual_fisheye_source_is_reprojected_as_is(tmp_path):
    from nepal.process.reproject import pick_sources
    a = _on_disk(tmp_path, [_shaped("v.insv", "insv", 3840, 1920, "dual_fisheye")])
    paths, mode = pick_sources(a, tmp_path)
    assert mode == "dual_fisheye" and len(paths) == 1


def test_a_lens_pair_is_recognised_as_two_files(tmp_path):
    """_00_ and _10_ of the same moment, one circle each."""
    from nepal.process.reproject import pick_sources
    a = _on_disk(tmp_path, [
        _shaped("VID_034346_10_039.insv", "insv", 2880, 2880, "single_fisheye", 39),
        _shaped("VID_034346_00_039.insv", "insv", 2880, 2880, "single_fisheye", 39)])
    paths, mode = pick_sources(a, tmp_path)
    assert mode == "lens_pair"
    assert [p.name for p in paths] == ["VID_034346_00_039.insv",
                                       "VID_034346_10_039.insv"]


def test_flat_footage_in_a_360_container_skips_reprojection(tmp_path):
    from nepal.process.reproject import pick_sources
    a = _on_disk(tmp_path, [_shaped("LRV_01_001.lrv", "lrv", 640, 360, "flat")])
    assert pick_sources(a, tmp_path)[1] == "flat"


def test_the_dual_fisheye_proxy_beats_the_original(tmp_path):
    """Both hold the same layout and the output is 540p either way."""
    from nepal.process.reproject import pick_sources
    a = _on_disk(tmp_path, [_shaped("v.insv", "insv", 3840, 1920, "dual_fisheye"),
                            _shaped("v.lrv", "lrv", 1024, 512, "dual_fisheye")])
    paths, mode = pick_sources(a, tmp_path)
    assert [p.name for p in paths] == ["v.lrv"] and mode == "dual_fisheye"


def test_real_360_is_preferred_over_a_flat_sibling(tmp_path):
    from nepal.process.reproject import pick_sources
    a = _on_disk(tmp_path, [_shaped("flat.lrv", "lrv", 640, 360, "flat"),
                            _shaped("dual.lrv", "lrv", 1024, 512, "dual_fisheye")])
    assert pick_sources(a, tmp_path)[1] == "dual_fisheye"


def test_an_odd_number_of_lenses_is_refused_not_guessed(tmp_path):
    """Stacking the wrong two circles is worse than skipping the recording."""
    from nepal.process.reproject import pick_sources
    a = _on_disk(tmp_path, [
        _shaped(f"VID_0{i}_039.insv", "insv", 2880, 2880, "single_fisheye", 39)
        for i in range(3)])
    assert pick_sources(a, tmp_path) is None


def test_the_shape_falls_back_to_the_dimensions_when_unrecorded(tmp_path):
    """A database written before frame_shape existed still works."""
    from nepal.process.reproject import pick_sources
    a = _on_disk(tmp_path, [_shaped("v.insv", "insv", 3840, 1920, None)])
    assert pick_sources(a, tmp_path)[1] == "dual_fisheye"


def test_the_lens_pair_graph_stacks_before_reprojecting(tmp_path):
    """v360 wants both circles in one frame; one alone is half a world."""
    from nepal.process.reproject import build_lens_pair_graph
    fc, _ = build_lens_pair_graph(193.0)
    assert fc.index("hstack") < fc.index("v360")
    assert "hstack=inputs=2" in fc
    assert "[0:v]" in fc and "[1:v]" in fc


def test_reprojection_happens_after_the_downscale_not_before():
    """v360 costs per pixel and is the whole cost of this pass. Reprojecting a
    5760x2880 stack to make a 1024x512 proxy is 4x the necessary work --
    measured at 12.0s against 3.0s for the same five seconds of footage."""
    from nepal.process.reproject import build_lens_pair_graph, build_proxy_only_graph
    for fc, _ in (build_lens_pair_graph(193.0), build_proxy_only_graph(193.0)):
        assert fc.index("scale") < fc.index("v360"), fc


def test_the_working_size_is_larger_than_the_output():
    """Downscaling below the output would throw away detail the proxy keeps."""
    from nepal.process.reproject import build_proxy_only_graph
    fc, _ = build_proxy_only_graph(193.0, proxy_size=(1024, 512), work_scale=2.0)
    assert "min(iw,2048)" in fc and "min(ih,1024)" in fc
    assert fc.rstrip().endswith("scale=1024:512[eqout]")


def test_the_downscale_never_becomes_an_upscale():
    """Half this corpus's 360 material is already a 1024x512 .lrv. Enlarging it
    to reproject it and shrinking it back was 4.3s against 1.6s for the same ten
    seconds -- a downscale that upscales is worse than none at all."""
    from nepal.process.reproject import build_proxy_only_graph, build_lens_pair_graph
    for fc, _ in (build_proxy_only_graph(193.0), build_lens_pair_graph(193.0)):
        pre = fc[:fc.index("v360")]
        assert "min(iw," in pre and "min(ih," in pre, pre


def test_the_flat_plan_never_mentions_v360(tmp_path):
    from nepal.process.reproject import plan_for_mode
    p = plan_for_mode([pathlib.Path("a.mp4")], "r", tmp_path, mode="flat")
    assert "v360" not in p.filter_complex
    assert not p.is_360


def test_a_lens_pair_gets_two_inputs_in_order(tmp_path):
    from nepal.process.reproject import plan_for_mode, build_command
    srcs = [pathlib.Path("a.insv"), pathlib.Path("b.insv")]
    p = plan_for_mode(srcs, "r", tmp_path, mode="lens_pair")
    cmd = build_command(p, inputs=srcs)
    assert cmd.count("-i") == 2
    assert cmd[cmd.index("-i") + 1] == "a.insv"


def test_the_proxy_frame_rate_can_be_capped(tmp_path):
    """v360 is the whole cost, so halving the frames halves the pass."""
    from nepal.process.reproject import plan_for_mode, build_command
    p = plan_for_mode([pathlib.Path("a.mp4")], "r", tmp_path, mode="flat")
    assert "-r" in build_command(p, fps=15)
    assert "-r" not in build_command(p, fps=None)


def test_a_square_telegram_video_is_not_mistaken_for_a_lens(tmp_path):
    """Round video messages are square by definition. Guessing shape from
    dimensions demanded a partner lens for each of them, then skipped the
    recording for not having one."""
    from nepal.process.reproject import pick_sources
    a = _on_disk(tmp_path, [{"s3_key": "raw/c/round.mp4", "container": "mp4",
                             "width": 384, "height": 384, "frame_shape": None,
                             "kind": "video_flat", "chapter_index": 1}])
    assert pick_sources(a, tmp_path)[1] == "flat"


def test_a_square_phone_clip_is_flat_too(tmp_path):
    from nepal.process.reproject import pick_sources
    a = _on_disk(tmp_path, [{"s3_key": "raw/c/IMG_1.MOV", "container": "mov",
                             "width": 1080, "height": 1080, "frame_shape": None,
                             "kind": "video_flat", "chapter_index": 1}])
    assert pick_sources(a, tmp_path)[1] == "flat"


def test_a_square_360_container_is_still_a_lens(tmp_path):
    """The narrowing must not lose the real case it exists for."""
    from nepal.process.reproject import pick_sources
    a = _on_disk(tmp_path, [
        {"s3_key": f"raw/c/VID_{i}0_039.insv", "container": "insv", "width": 2880,
         "height": 2880, "frame_shape": None, "kind": "video360",
         "chapter_index": 39} for i in (0, 1)])
    assert pick_sources(a, tmp_path)[1] == "lens_pair"


# -- not re-encoding a proxy that is already a proxy --------------------

def test_a_source_no_larger_than_the_proxy_is_remuxed(tmp_path):
    """An .lrv is already 640x360. Enlarging it to 960x540 and re-encoding
    costs a full pass to produce something worse: 4.09s against 0.08s."""
    from nepal.process.reproject import plan_for_mode, build_command
    p = plan_for_mode([pathlib.Path("a.lrv")], "r", tmp_path, mode="flat",
                      passthrough=True)
    assert p.filter_complex == ""
    cmd = build_command(p, inputs=[pathlib.Path("a.lrv")], fps=15, preset="veryfast")
    assert "-c:v" in cmd and cmd[cmd.index("-c:v") + 1] == "copy"
    assert "-filter_complex" not in cmd
    assert "libx264" not in cmd


def test_a_remux_still_extracts_audio(tmp_path):
    """S03.4 needs the 16 kHz track whatever happened to the video."""
    from nepal.process.reproject import plan_for_mode, build_command
    p = plan_for_mode([pathlib.Path("a.lrv")], "r", tmp_path, mode="flat",
                      passthrough=True)
    cmd = build_command(p, inputs=[pathlib.Path("a.lrv")], has_audio=True)
    assert "pcm_s16le" in cmd and str(p.audio_path) in cmd


def test_flat_video_larger_than_the_proxy_is_scaled_down_only(tmp_path):
    from nepal.process.reproject import plan_for_mode
    p = plan_for_mode([pathlib.Path("a.mp4")], "r", tmp_path, mode="flat",
                      view_size=(960, 540))
    assert "min(iw,960)" in p.filter_complex
    assert "v360" not in p.filter_complex


def test_the_preset_is_applied_only_where_something_is_encoded(tmp_path):
    from nepal.process.reproject import plan_for_mode, build_command
    enc = plan_for_mode([pathlib.Path("a.mp4")], "r", tmp_path, mode="flat")
    assert "-preset" in build_command(enc, preset="veryfast")
    remux = plan_for_mode([pathlib.Path("a.lrv")], "r", tmp_path, mode="flat",
                          passthrough=True)
    assert "-preset" not in build_command(remux, preset="veryfast")
