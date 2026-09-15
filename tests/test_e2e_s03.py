"""End-to-end S03: a real clip through reprojection and shot detection.

Slow: builds media with ffmpeg. Run with:  pytest -m slow
"""
import json
import shutil
import sys
import pathlib
from datetime import datetime, timedelta, timezone

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

pytestmark = pytest.mark.slow

needs_tools = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
UTC = timezone.utc
START = datetime(2024, 5, 6, 8, 12, tzinfo=UTC)


def ffmpeg(*args):
    from nepal.util import proc
    proc.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """One flat recording of two visibly different halves, so exactly one cut."""
    from nepal import db
    from nepal.config import Config

    root = tmp_path_factory.mktemp("s03")
    cam = root / "data" / "media_from_camera"
    cam.mkdir(parents=True)
    a, b = root / "a.mp4", root / "b.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc=size=640x320:rate=10:duration=3",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", str(a))
    ffmpeg("-f", "lavfi", "-i", "smptebars=size=640x320:rate=10:duration=3",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", str(b))
    clip = cam / "VID_20240506_081200_00_001.mp4"
    ffmpeg("-i", str(a), "-i", str(b), "-filter_complex",
           "[0:v][1:v]concat=n=2:v=1[v]", "-map", "[v]",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip))

    cfg_path = root / "pipeline.yaml"
    cfg_path.write_text(
        f"project: {{name: t, data_root: {root}/data, work_root: {root}/work,\n"
        f"  db_path: {root}/work/db/n.sqlite}}\n"
        f"probe: {{fov: {{fallback_deg: 193}}}}\n"
        f"spine: {{max_interp_gap_s: 14400}}\n"
        f"process: {{scene_threshold: 27.0, min_shot_s: 1.5, yaw_videos: false,\n"
        f"  photo_analysis_px: 512, photo_slot_s: 3.0, photo_workers: 0}}\n"
        f"quality_curves: {{camera: {{min_sharpness: 4.0, max_exposure_pen: 0.15}},\n"
        f"  phone: {{min_sharpness: 3.5, max_exposure_pen: 0.20}},\n"
        f"  telegram: {{min_sharpness: 2.0, max_exposure_pen: 0.35}}}}\n")
    cfg = Config.load(cfg_path)
    conn = db.init(cfg.db_path)
    conn.execute("INSERT INTO recordings(recording_id, source, is_360, start_utc, "
                 "duration_s, asset_count) VALUES "
                 "('camera_20240506_081200','camera',0,?,6.0,1)", (START.isoformat(),))
    conn.execute("INSERT INTO assets(asset_id, s3_key, source, kind, container, "
                 "chapter_index, recording_id, created_at_utc) VALUES ('a1',"
                 "'raw/media_from_camera/VID_20240506_081200_00_001.mp4','camera',"
                 "'video_flat','mp4',1,'camera_20240506_081200',?)", (START.isoformat(),))
    db.set_decision(conn, "act_boundaries", json.dumps([
        {"act": 4, "start_utc": (START - timedelta(hours=4)).isoformat(),
         "end_utc": (START + timedelta(hours=4)).isoformat(), "method": "alt"}]))
    for i in range(3):
        conn.execute("INSERT INTO gps_points(ts_utc, lat, lon, alt_dem_m, source) "
                     "VALUES (?,?,?,?,'t')",
                     ((START + timedelta(minutes=i)).isoformat(),
                      28.5 + i * 0.01, 84.6, 5100.0))
    conn.commit()
    yield cfg, conn
    conn.close()


@needs_tools
def test_s03_1_writes_a_proxy_and_audio_track(project):
    from nepal.stages.s03_process import build_proxies
    cfg, conn = project
    rep = build_proxies(cfg, conn)
    assert rep["n_built"] == 1 and rep["n_failed"] == 0
    assert (cfg.work_root / "proxies" / "camera_20240506_081200_eq.mp4").exists()


@needs_tools
def test_s03_1_is_resumable_and_does_not_redo_work(project):
    from nepal.stages.s03_process import build_proxies
    cfg, conn = project
    rep = build_proxies(cfg, conn)
    assert rep["n_skipped"] == 1, "a completed recording must not be re-encoded"
    assert rep["n_built"] == 0


@needs_tools
def test_s03_2_finds_the_cut_where_it_actually_is(project):
    """Two visibly different halves joined at 3 s: one cut, two shots."""
    pytest.importorskip("scenedetect")
    from nepal.stages.s03_process import detect_shots
    cfg, conn = project
    rep = detect_shots(cfg, conn)
    assert rep["n_shots"] == 2, "one hard cut should give exactly two shots"
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM shots WHERE media_kind='video' ORDER BY start_s")]
    assert rows[0]["end_s"] == pytest.approx(3.0, abs=0.3)
    assert rows[1]["start_s"] == pytest.approx(3.0, abs=0.3)


@needs_tools
def test_video_shots_carry_their_moment_act_and_position(project):
    pytest.importorskip("scenedetect")
    from nepal.stages.s03_process import detect_shots
    cfg, conn = project
    detect_shots(cfg, conn)
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM shots WHERE media_kind='video' ORDER BY start_s")]
    assert all(r["act"] == 4 for r in rows)
    assert rows[0]["start_utc"].startswith("2024-05-06T08:12:00")
    # a shot is a span and the walker moved, so the two differ
    assert rows[0]["lat"] is not None and rows[1]["lat"] is not None
    assert rows[0]["lat"] != rows[1]["lat"]


@needs_tools
def test_a_video_shot_has_a_recording_and_no_asset(project):
    pytest.importorskip("scenedetect")
    from nepal.stages.s03_process import detect_shots
    cfg, conn = project
    detect_shots(cfg, conn)
    row = conn.execute("SELECT recording_id, asset_id FROM shots "
                       "WHERE media_kind='video' LIMIT 1").fetchone()
    assert row["recording_id"] == "camera_20240506_081200"
    assert row["asset_id"] is None


@needs_tools
def test_rerunning_detection_does_not_duplicate_shots(project):
    pytest.importorskip("scenedetect")
    from nepal.stages.s03_process import detect_shots
    cfg, conn = project
    detect_shots(cfg, conn)
    n1 = conn.execute("SELECT COUNT(*) FROM shots").fetchone()[0]
    detect_shots(cfg, conn)
    assert conn.execute("SELECT COUNT(*) FROM shots").fetchone()[0] == n1


# -- S03.3 and S03.7, against a real proxy ---------------------------
@needs_tools
def test_s03_3_measures_every_shot_from_the_proxy(project):
    """Measured through OpenCV against a file ffmpeg wrote, not against a
    hand-built array. Every metric this pipeline has got wrong so far was wrong
    at exactly this boundary."""
    from nepal.stages.s03_process import measure_shots
    cfg, conn = project
    rep = measure_shots(cfg, conn)
    assert rep["n_measured"] == rep["n_shots"] > 0
    assert not rep["failed"]
    rows = conn.execute("SELECT sharpness, exposure_pen, motion_mag, stability "
                        "FROM shots WHERE media_kind='video'").fetchall()
    assert rows and all(all(v is not None for v in r) for r in rows)
    assert all(0.0 <= r["stability"] <= 1.0 for r in rows)


@needs_tools
def test_s03_3_does_not_remeasure_what_it_already_measured(project):
    from nepal.stages.s03_process import measure_shots
    cfg, conn = project
    assert measure_shots(cfg, conn)["n_shots"] == 0


@needs_tools
def test_a_static_shot_scores_as_steadier_than_a_moving_one(project):
    """The first half is a scrolling test pattern, the second is still bars.
    If the motion metrics cannot tell those apart they cannot tell a walking
    shot from a locked-off one either."""
    cfg, conn = project
    rows = conn.execute("SELECT motion_mag, stability FROM shots "
                        "WHERE media_kind='video' ORDER BY start_s").fetchall()
    assert len(rows) == 2
    moving, static = rows[0], rows[1]
    assert static["motion_mag"] < moving["motion_mag"]
    assert static["stability"] >= moving["stability"]


@needs_tools
def test_s03_4_measures_loudness_wind_and_speech_per_shot(project):
    """The fixture clip is silent, so the track is planted: a 60 Hz rumble at
    -20 dBFS under the first shot, digital silence under the second. Loudness
    must be read per window, the rumble must count as wind, and neither half
    is speech. The flags are then derived at the gate from what was stored."""
    import numpy as np
    import soundfile as sf
    from nepal.stages.s03_process import apply_gate, measure_audio, run
    cfg, conn = project
    if not conn.execute("SELECT 1 FROM shots LIMIT 1").fetchone():
        run(cfg)                                   # stands alone under -k
    sr = 16000
    t = np.arange(sr * 3) / sr
    rumble = (0.1 * np.sin(2 * np.pi * 60 * t)).astype(np.float32)
    wav = cfg.work_root / "audio" / "camera_20240506_081200.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(wav, np.concatenate([rumble, np.zeros(sr * 3, np.float32)]), sr, subtype="PCM_16")

    rep = measure_audio(cfg, conn)
    assert rep["n_measured"] == 2 and not rep["failed"]
    first, second = conn.execute(
        "SELECT audio_lufs, wind_lf_share, speech_s FROM shots "
        "WHERE media_kind='video' ORDER BY start_s").fetchall()
    assert first["audio_lufs"] is not None and first["audio_lufs"] > -40
    assert second["audio_lufs"] is None            # digital silence measures nothing
    assert first["wind_lf_share"] > 0.9 and second["wind_lf_share"] == 0.0
    assert first["speech_s"] == 0.0 and second["speech_s"] == 0.0

    assert measure_audio(cfg, conn)["n_shots"] == 0    # resumable through the data

    apply_gate(cfg, conn)
    first, second = conn.execute(
        "SELECT has_speech, wind FROM shots WHERE media_kind='video' ORDER BY start_s").fetchall()
    assert (first["has_speech"], first["wind"]) == (0, 1)
    assert (second["has_speech"], second["wind"]) == (0, 0)


@needs_tools
def test_s03_7_gives_every_shot_a_status(project):
    from nepal.stages.s03_process import apply_gate
    cfg, conn = project
    rep = apply_gate(cfg, conn)
    assert rep["n_shots"] > 0 and rep["n_unmeasured"] == 0
    assert not conn.execute(
        "SELECT 1 FROM shots WHERE status NOT IN ('candidate','rejected')").fetchall()


@needs_tools
def test_moving_jerk_ref_changes_stability_at_the_gate_without_a_remeasure(project):
    """The reference is tuned against the corpus. Re-deriving stability from
    the stored jerk makes that a gate re-run, not the better part of an hour."""
    from nepal.stages.s03_process import apply_gate
    cfg, conn = project
    conn.execute("UPDATE shots SET jerk_px = 1.0 WHERE media_kind='video'")
    conn.commit()
    cfg._data.setdefault("process", {})["metric_jerk_ref_px"] = 1.0
    apply_gate(cfg, conn)
    assert {r[0] for r in conn.execute(
        "SELECT stability FROM shots WHERE media_kind='video'")} == {0.5}
    cfg._data["process"]["metric_jerk_ref_px"] = 3.0
    apply_gate(cfg, conn)
    assert {r[0] for r in conn.execute(
        "SELECT stability FROM shots WHERE media_kind='video'")} == {0.75}


@needs_tools
def test_the_gate_judges_a_shot_on_its_own_sources_curve(project):
    """The camera curve would reject this material; the telegram curve is the
    one that keeps Act 1 alive, and it must be the asset's curve that decides."""
    from nepal.stages.s03_process import apply_gate
    cfg, conn = project
    conn.execute("UPDATE shots SET sharpness = 2.5")
    conn.commit()
    conn.execute("UPDATE assets SET quality_curve='camera'")
    assert apply_gate(cfg, conn)["n_kept"] == 0
    conn.execute("UPDATE assets SET quality_curve='telegram'")
    assert apply_gate(cfg, conn)["n_kept"] > 0


@needs_tools
def test_re_detection_replaces_a_recordings_shots_rather_than_adding_to_them(project):
    """Boundaries move whenever the threshold or the maximum length moves. An
    upsert alone leaves the old numbering behind as shots that describe nothing."""
    from nepal.stages.s03_process import detect_shots
    cfg, conn = project
    before = conn.execute("SELECT COUNT(*) FROM shots WHERE media_kind='video'").fetchone()[0]
    conn.execute("INSERT INTO shots(shot_id, recording_id, media_kind, start_s, end_s) "
                 "VALUES ('camera_20240506_081200#0099','camera_20240506_081200',"
                 "'video',0.0,1.0)")
    conn.commit()
    detect_shots(cfg, conn)
    assert conn.execute("SELECT COUNT(*) FROM shots WHERE media_kind='video'"
                        ).fetchone()[0] == before
    assert not conn.execute("SELECT 1 FROM shots WHERE shot_id LIKE '%#0099'").fetchall()
