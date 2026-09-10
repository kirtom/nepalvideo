"""End-to-end S01 against fixtures with planted ground truth.

Slow (builds real media with ffmpeg, ~2 min). Run with:  pytest -m slow
"""
import json
import shutil
import sys
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

pytestmark = pytest.mark.slow

needs_tools = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("exiftool")),
    reason="needs ffmpeg and exiftool",
)


@pytest.fixture(scope="module")
def probed(tmp_path_factory):
    from tools.make_fixtures import build
    from nepal.config import Config
    from nepal.stages import s01_probe
    import yaml

    base = tmp_path_factory.mktemp("e2e")
    data = base / "nepal_data"
    truth = build(data, quick=True)

    cfg_data = yaml.safe_load((ROOT / "config" / "pipeline.yaml").read_text())
    cfg_data["project"]["data_root"] = str(data)
    cfg_data["project"]["work_root"] = str(base / "work")
    cfg_data["project"]["db_path"] = str(base / "work" / "db" / "nepal.sqlite")
    cfg_path = base / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_data))

    cfg = Config.load(cfg_path)
    report = s01_probe.run(cfg)
    return truth, report, cfg


@needs_tools
def test_fov_recovers_the_projected_lens_fov(probed):
    """The clips were projected to dual fisheye at a known FOV; S01.4 must find it."""
    truth, report, _ = probed
    assert report["fov"]["fov_deg"] == truth["true_fov"]
    assert not report["fov"]["used_fallback"]
    assert report["fov"]["confidence"] > 0.05


@needs_tools
def test_reference_clock_is_the_gps_validated_phone(probed):
    """The spec nominates the camera as reference. On material where the
    camera's battery went flat that is the one clock that is provably wrong,
    while phone photos carry satellite time. The evidence must decide."""
    truth, report, cfg = probed
    from nepal import db
    conn = db.init(cfg.db_path)
    assert db.get_decision(conn, "clock_reference") == truth["reference_clock"]
    assert db.get_decision_float(conn, "clock_vs_gps_phone_keller_s") == pytest.approx(0.0, abs=2)
    conn.close()


@needs_tools
def test_reference_device_takes_zero_offset(probed):
    _, report, _ = probed
    assert report["clock"]["keller"]["offset_s"] == pytest.approx(0.0, abs=1.0)


@needs_tools
def test_second_phone_offset_recovered(probed):
    """Device timestamps are whole-second, so 1 s is the honest tolerance --
    the correlation resolves far finer than the metadata does."""
    truth, report, _ = probed
    got = report["clock"]["kulikov"]["offset_s"]
    assert abs(got - truth["true_kulikov_offset_s"]) <= 1.0
    assert not report["clock"]["kulikov"]["needs_manual"]


@needs_tools
def test_fourteen_day_camera_clock_error_is_recovered(probed):
    """The headline case. A flat battery resets an action camera's clock, and
    the resulting error is thousands of times wider than the +/-600 s window
    GCC-PHAT searches. Coincidence voting proposes candidates, audio picks
    the right one and refines it to sub-second."""
    truth, report, _ = probed
    got = report["clock"]["camera"]["offset_s"]
    want = truth["true_camera_offset_s"]
    assert abs(got - want) <= 1.0, f"got {got}, want {want}"
    assert abs(got - want) < 86400, "must not be a whole-day shift out"
    assert not report["clock"]["camera"]["needs_manual"]
    assert report["clock"]["camera"]["pairs_accepted"] >= 3


@needs_tools
def test_camera_corrected_time_lands_inside_the_trek(probed):
    """The point of the correction: after it, camera recordings sit inside the
    window the phones describe, which is what every downstream join needs."""
    from nepal import db
    _, _, cfg = probed
    conn = db.init(cfg.db_path)
    row = conn.execute(
        "SELECT MIN(created_at_utc) lo, MAX(created_at_utc) hi FROM assets "
        "WHERE source='camera' AND created_at_utc IS NOT NULL").fetchone()
    phone = conn.execute(
        "SELECT MIN(created_at_utc) lo, MAX(created_at_utc) hi FROM assets "
        "WHERE source LIKE 'phone_%' AND created_at_utc IS NOT NULL").fetchone()
    conn.close()
    assert row["lo"] and phone["lo"]
    from datetime import datetime, timedelta
    cam_lo = datetime.fromisoformat(row["lo"])
    ph_lo = datetime.fromisoformat(phone["lo"])
    ph_hi = datetime.fromisoformat(phone["hi"])
    assert ph_lo - timedelta(days=1) <= cam_lo <= ph_hi + timedelta(days=1), \
        f"camera {cam_lo} outside phone window {ph_lo}..{ph_hi}"


@needs_tools
def test_unrelated_audio_pairs_are_rejected(probed):
    """kulikov has a decoy clip sharing no sound with any camera take."""
    _, report, _ = probed
    k = report["clock"]["kulikov"]
    assert k["pairs_accepted"] < k["pairs_total"]


@needs_tools
def test_chapters_collapse_into_one_recording(probed):
    """Two .insv chapters plus their .lrv proxy must be a single recording."""
    from nepal import db
    _, _, cfg = probed
    conn = db.init(cfg.db_path)
    row = conn.execute(
        "SELECT recording_id, asset_count FROM recordings "
        "WHERE source='camera' ORDER BY asset_count DESC LIMIT 1").fetchone()
    assert row["asset_count"] == 3
    conn.close()


@needs_tools
def test_camera_reports_no_gps(probed):
    _, report, _ = probed
    assert report["gps_check"]["camera_has_gps"] is False


@needs_tools
def test_timezone_offset_tag_is_actually_requested(probed):
    """Regression guard for a bug that unit tests could not catch: the code
    read OffsetTimeOriginal but exiftool was never asked for it, so every
    Nepal photo was parsed 5h45m off. The GPS comparison exposes it -- a
    +20700 s median is exactly that timezone leaking through."""
    from nepal import db
    _, _, cfg = probed
    conn = db.init(cfg.db_path)
    delta = db.get_decision_float(conn, "clock_vs_gps_phone_keller_s")
    conn.close()
    assert delta is not None, "no GPS/clock comparison was recorded"
    assert abs(delta) < 60, f"device clock sits {delta}s from GPS -- timezone leak?"


@needs_tools
def test_phone_photos_carry_gps_and_corrected_utc(probed):
    from nepal import db
    _, _, cfg = probed
    conn = db.init(cfg.db_path)
    n = conn.execute("SELECT COUNT(*) c FROM assets WHERE kind='photo' "
                     "AND has_gps=1 AND source LIKE 'phone_%'").fetchone()["c"]
    assert n > 0
    missing = conn.execute("SELECT COUNT(*) c FROM assets WHERE created_at IS NOT NULL "
                           "AND created_at_utc IS NULL").fetchone()["c"]
    assert missing == 0, "every dated asset must get a corrected UTC time"
    conn.close()


@needs_tools
def test_telegram_media_gets_the_telegram_quality_curve(probed):
    from nepal import db
    _, _, cfg = probed
    conn = db.init(cfg.db_path)
    rows = conn.execute("SELECT quality_curve FROM assets WHERE source='telegram'").fetchall()
    assert rows and all(r["quality_curve"] == "telegram" for r in rows)
    conn.close()


@needs_tools
def test_rerun_is_idempotent_and_resumable(probed):
    """A Spot interruption must re-run only what is missing."""
    from nepal.stages import s01_probe
    _, _, cfg = probed
    again = s01_probe.run(cfg)
    assert again["manifest"] == {"skipped": "already done"}
    assert again["fov"]["skipped"] == "already done"
