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


def test_s03_measuring_steps_always_run_resumably_even_when_marked_done(tmp_path, monkeypatch):
    """A re-detection replaced 1,060 shot rows, and metrics, audio, asr and
    faces stayed behind a "done" marker: no metrics, no transcript, no face.
    Each of them resumes through the data, so each always runs, with
    force=False, and costs nothing when nothing is pending."""
    from nepal.stages import s03_process as s3
    work = tmp_path / "work"
    cfg = Config({"project": {"data_root": str(tmp_path / "data"), "work_root": str(work),
                              "db_path": str(work / "db" / "nepal.sqlite")}})
    conn = db.init(cfg.db_path)
    for unit in ("proxies", "shots", "photos", "metrics", "audio", "asr", "faces"):
        db.mark_unit(conn, "S03", unit)
    conn.close()
    calls: dict[str, dict] = {}

    def spy(name):
        def fn(*a, **k):
            calls[name] = k
            return {}
        return fn
    for name in ("build_proxies", "detect_shots", "build_photo_shots", "place_shots",
                 "measure_shots", "measure_audio", "transcribe_shots", "detect_faces",
                 "recluster_faces", "apply_gate"):
        monkeypatch.setattr(s3, name, spy(name))
    monkeypatch.setattr(s3.freshness, "warn_if_stale", lambda *a, **k: [])
    rep = s3.run(cfg)
    for name in ("detect_shots", "measure_shots", "measure_audio", "transcribe_shots",
                 "detect_faces"):
        assert name in calls, f"{name} must run even when marked done"
    assert calls["measure_shots"]["force"] is False and calls["detect_faces"]["force"] is False
    assert calls["detect_shots"]["redo_all"] is False
    assert "build_proxies" not in calls and "build_photo_shots" not in calls
    assert rep["proxies"] == {"skipped": "already done"}


def test_shot_detection_resumes_per_recording(tmp_path):
    """Only recordings with a proxy and no shots yet; everything with an
    explicit redo."""
    from nepal.stages.s03_process import recordings_to_detect
    proxies = tmp_path / "proxies"
    proxies.mkdir()
    for rid in ("a", "b"):
        (proxies / f"{rid}_eq.mp4").write_bytes(b"x")
    recs = [{"recording_id": "a"}, {"recording_id": "b"}, {"recording_id": "c"}]
    todo = recordings_to_detect(recs, have_shots={"a"}, proxies=proxies, redo_all=False)
    assert [r["recording_id"] for r in todo] == ["b"]          # a has shots, c has no proxy
    everything = recordings_to_detect(recs, have_shots={"a"}, proxies=proxies, redo_all=True)
    assert [r["recording_id"] for r in everything] == ["a", "b"]
