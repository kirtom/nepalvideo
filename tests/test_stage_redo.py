"""--redo re-runs a named sub-step without paying for the others."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest

from nepal import db
from nepal.config import Config
from nepal.stages import s01_probe


def _cfg(tmp_path):
    data = tmp_path / "data"
    (data / "media_from_phones" / "kulikov").mkdir(parents=True)
    (data / "media_from_phones" / "kulikov" / "note.txt").write_text("not media")
    return Config({
        "project": {"data_root": str(data), "work_root": str(tmp_path / "work"),
                    "db_path": str(tmp_path / "work" / "db" / "nepal.sqlite")},
        "probe": {"max_chapter_gap_s": 1.0, "exclude_dirs": [], "ignore_globs": [],
                  "capture_time_overrides": {}},
    })


def test_s01_redo_reruns_only_the_named_unit(tmp_path):
    cfg = _cfg(tmp_path)
    first = s01_probe.run(cfg, skip_fov=True, skip_clock=True)
    assert "n_assets" in first["manifest"] and "n_recordings" in first["chapters"]
    second = s01_probe.run(cfg, skip_fov=True, skip_clock=True)
    assert second["manifest"] == {"skipped": "already done"}
    third = s01_probe.run(cfg, skip_fov=True, skip_clock=True, redo={"manifest"})
    assert "n_assets" in third["manifest"]
    assert third["chapters"] == {"skipped": "already done"}


def test_s01_redo_rejects_an_unknown_unit(tmp_path):
    with pytest.raises(SystemExit):
        s01_probe.run(_cfg(tmp_path), skip_fov=True, skip_clock=True, redo={"nonsense"})


def test_capture_time_override_sets_created_at_utc_by_filename(tmp_path):
    conn = db.init(tmp_path / "t.sqlite")
    conn.execute("INSERT INTO assets(asset_id, s3_key, source, kind, created_at) "
                 "VALUES ('x', 'raw/media_from_phones/kulikov/video_1.mp4', "
                 "'phone_kulikov', 'video_flat', '2025-11-22T23:46:56+00:00')")
    conn.commit()
    n = s01_probe.apply_offsets(conn, {"kulikov": 0.0},
                                overrides={"video_1.mp4": "2024-05-12T09:30:00+05:45"})
    assert n == 1
    got = conn.execute("SELECT created_at_utc FROM assets WHERE asset_id='x'").fetchone()[0]
    assert got == "2024-05-12T03:45:00+00:00"
