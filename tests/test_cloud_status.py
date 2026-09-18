"""The status document: what is running, what has run, what it cost."""
import json
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from datetime import datetime, timezone

from nepal import db
from nepal.cloud import spend, status


def _seed(tmp_path):
    conn = db.init(tmp_path / "db" / "nepal.sqlite")
    conn.executemany("INSERT INTO assets(asset_id, s3_key, source, kind, created_at_utc) "
                     "VALUES (?,?,?,?,?)",
                     [("a", "k1", "phone_keller", "photo", "2024-05-01T00:00:00+00:00"),
                      ("b", "k2", "camera", "video360", None)])
    conn.execute("INSERT INTO recordings(recording_id, source, is_360) VALUES ('r', 'camera', 1)")
    conn.executemany("INSERT INTO shots(shot_id, recording_id, media_kind, start_s, end_s, status) "
                     "VALUES (?,?,?,?,?,?)",
                     [("r#0", "r", "video", 0, 5, "shortlisted"),
                      ("r#1", "r", "video", 5, 9, "rejected")])
    conn.execute("INSERT INTO timeline(slot_index, act, shot_id, t_in, t_out) "
                 "VALUES (0, 3, 'r#0', 0, 4.5)")
    db.mark_unit(conn, "S03", "faces", detail="x")
    db.mark_unit(conn, "S03", "place")
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "s03_process.json").write_text(json.dumps(
        {"stage": "S03", "finished_utc": "2026-09-18T05:00:00+00:00"}))
    (reports / "remote_jobs.log").write_text("--- s03 05:00:00\nS03.6 faces 10/10\n")
    led = spend.Ledger(reports / "spend")
    led.record("gce:cpu", 0.42)
    return conn, led, reports


def test_build_status_reads_everything_from_the_database_and_the_reports(tmp_path):
    conn, led, reports = _seed(tmp_path)
    st = status.build_status(conn, led, reports,
                             now=datetime(2026, 9, 18, 6, tzinfo=timezone.utc))
    assert st["counts"] == {"assets": 2, "dated_assets": 1, "recordings": 1, "shots": 2,
                            "surviving": 1, "shortlisted": 1, "slots": 1, "timeline_s": 4.5}
    assert {s["unit"] for s in st["stages"]} == {"faces", "place"}
    assert st["latest_stage"]["unit"] in ("faces", "place")
    assert st["spend"]["total_usd"] == 0.42 and st["spend"]["entries"][-1]["what"] == "gce:cpu"
    assert st["last_report"] == {"name": "s03_process.json",
                                 "finished_utc": "2026-09-18T05:00:00+00:00"}
    assert st["draft"] is None
    assert st["remote_jobs_tail"][-1] == "S03.6 faces 10/10"
    assert st["generated_utc"] == "2026-09-18T06:00:00+00:00"


def test_render_html_is_self_contained_and_names_the_stage(tmp_path):
    conn, led, reports = _seed(tmp_path)
    page = status.render_html(status.build_status(conn, led, reports))
    assert page.startswith("<!doctype html>") and "<script" not in page
    assert "place" in page and "0.42" in page and "S03.6 faces 10/10" in page


def test_write_status_writes_both_files(tmp_path):
    from nepal.config import Config
    conn, led, reports = _seed(tmp_path)
    conn.close()
    cfg = Config({"project": {"data_root": str(tmp_path / "data"), "work_root": str(tmp_path),
                              "db_path": str(tmp_path / "db" / "nepal.sqlite")}})
    out = status.write_status(cfg)
    assert out.name == "index.html" and (tmp_path / "status" / "status.json").exists()
    assert json.loads((tmp_path / "status" / "status.json").read_text())["counts"]["assets"] == 2
