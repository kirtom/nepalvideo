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
    conn.executemany("INSERT INTO shots(shot_id, recording_id, asset_id, media_kind, start_s, end_s, "
                     "act, status) VALUES (?,?,?,?,?,?,?,?)",
                     [("r#0", "r", None, "video", 0, 5, 3, "shortlisted"),
                      ("r#1", "r", None, "video", 5, 9, 3, "rejected"),
                      ("r#2", "r", None, "video", 9, 14, 3, "candidate"),
                      ("photo_a", None, "a", "photo", 0, 4, 3, "shortlisted")])
    conn.execute("INSERT INTO timeline(slot_index, act, kind, shot_id, t_in, t_out) "
                 "VALUES (0, 3, 'video', 'r#0', 0, 4.5)")
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
    assert st["counts"] == {"assets": 2, "dated_assets": 1, "recordings": 1, "shots": 4,
                            "surviving": 3, "shortlisted": 2, "slots": 1, "timeline_s": 4.5}
    # per act, per source: what the cut took against what survived -- the
    # rejected shot is not material, the photo counts under its asset's source
    # and is told apart from the clips, which the share rule is about
    assert st["per_act_sources"] == {
        "3": {"camera": {"slots": 1, "available": 2, "videos": 2, "video_slots": 1},
              "phone_keller": {"slots": 0, "available": 1, "videos": 0, "video_slots": 0}}}
    assert {s["unit"] for s in st["stages"]} == {"faces", "place"}
    assert st["latest_stage"]["unit"] in ("faces", "place")
    assert st["spend"]["total_usd"] == 0.42 and st["spend"]["entries"][-1]["what"] == "gce:cpu"
    assert st["spend"]["booked_usd"] == 0.42 and st["spend"]["running_usd"] == 0.0
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
    assert "<h2>sources per act</h2>" in page
    assert "<td>3</td><td>camera</td><td>1</td><td>2</td><td>100%</td>" in page
    assert "<td>3</td><td>phone_keller</td><td>0</td><td>1</td><td>0%</td>" in page


def test_a_report_with_no_finished_stamp_is_running_not_unknown(tmp_path):
    """S05 checkpoints its report after every sub-step, so the file exists
    for the whole of a cut with `finished_utc` null. The page said nothing
    at all for the hours that mattered most."""
    conn, led, reports = _seed(tmp_path)
    (reports / "s05_cut.json").write_text(json.dumps(
        {"stage": "S05", "started_utc": "2026-09-18T05:30:00+00:00",
         "skipped_stale": False, "score": {"n": 3}, "timeline": {"slots": 1}}))
    st = status.build_status(conn, led, reports)
    rep = st["last_report"]
    assert rep["name"] == "s05_cut.json" and rep["finished_utc"] is None
    assert rep["running_since"] == "2026-09-18T05:30:00+00:00"
    assert rep["steps_done"] == ["score", "timeline"]      # not cues, not draft
    assert rep["updated_utc"].endswith("+00:00")           # the last checkpoint
    assert "running since 2026-09-18T05:30:00+00:00" in status.render_html(st)
    # and a finished report still reads as finished
    (reports / "s05_cut.json").write_text(json.dumps(
        {"stage": "S05", "started_utc": "2026-09-18T05:30:00+00:00",
         "finished_utc": "2026-09-18T06:30:00+00:00", "score": {"n": 3}}))
    done = status.build_status(conn, led, reports)["last_report"]
    assert done["finished_utc"] == "2026-09-18T06:30:00+00:00" and "steps_done" not in done


def test_the_page_counts_a_box_that_is_up_and_not_booked(tmp_path):
    """The ledger books VM hours at `down`; `up` leaves its stamp and its
    price in remote_state.json, and until `down` clears them the meter is
    running. Showing only the booked total is how 104 idle hours hid."""
    conn, led, reports = _seed(tmp_path)
    (reports / "remote_state.json").write_text(json.dumps(
        {"cpu": {"up_since": "2026-09-18T04:00:00+00:00", "usd_per_h": 0.3}}))
    st = status.build_status(conn, led, reports,
                             now=datetime(2026, 9, 18, 6, tzinfo=timezone.utc))
    assert st["spend"] == {**st["spend"], "booked_usd": 0.42, "running_usd": 0.6,
                           "running_since": "2026-09-18T04:00:00+00:00", "total_usd": 1.02}
    assert "booked 0.42 + running 0.60" in status.render_html(st)


def test_write_status_writes_both_files(tmp_path):
    from nepal.config import Config
    conn, led, reports = _seed(tmp_path)
    conn.close()
    cfg = Config({"project": {"data_root": str(tmp_path / "data"), "work_root": str(tmp_path),
                              "db_path": str(tmp_path / "db" / "nepal.sqlite")}})
    out = status.write_status(cfg)
    assert out.name == "index.html" and (tmp_path / "status" / "status.json").exists()
    assert json.loads((tmp_path / "status" / "status.json").read_text())["counts"]["assets"] == 2
