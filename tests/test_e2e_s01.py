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
@pytest.mark.parametrize("device", ["keller", "kulikov"])
def test_clock_offsets_recover_the_planted_skew(probed, device):
    """Device timestamps are whole-second, so 1 s is the honest tolerance --
    the correlation itself resolves far finer than the metadata does."""
    truth, report, _ = probed
    got = report["clock"][device]["offset_s"]
    want = truth[f"true_{device}_offset_s"]
    assert abs(got - want) <= 1.0, f"{device}: got {got}, want {want}"
    assert not report["clock"][device]["needs_manual"]


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
