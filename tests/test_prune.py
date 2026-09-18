"""nepal prune: db.upsert never removes, so something must."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from nepal import db, prune


def _seed(tmp_path):
    conn = db.init(tmp_path / "db" / "nepal.sqlite")
    work = tmp_path / "work"
    for d in ("proxies", "audio", "transcripts/shots"):
        (work / d).mkdir(parents=True)
    conn.executemany(
        "INSERT INTO assets(asset_id, s3_key, source, kind, container, duration_s, "
        "created_at, created_at_utc, recording_id) VALUES (?,?,?,?,?,?,?,?,?)",
        [("a1", "raw/media_from_phones/kulikov/IMG_0001.MOV", "phone_kulikov",
          "video_flat", "mov", 4.0, "2024-05-01T01:00:00+00:00",
          "2024-05-01T01:00:00+00:00", "phone_kulikov_IMG_0001"),
         ("a2", "raw/media_from_phones/kulikov/IMG_0002.MOV", "phone_kulikov",
          "video_flat", "mov", 3.0, "2024-05-01T02:00:00+00:00",
          "2024-05-01T02:00:00+00:00", "phone_kulikov_IMG_0002")])
    conn.executemany(
        "INSERT INTO recordings(recording_id, source, is_360, start_utc, duration_s, "
        "asset_count) VALUES (?,?,?,?,?,?)",
        [("phone_kulikov_IMG_0001", "phone_kulikov", 0, "2024-05-01T01:00:00+00:00", 4.0, 1),
         ("phone_kulikov_IMG_0002", "phone_kulikov", 0, "2024-05-01T02:00:00+00:00", 3.0, 1),
         # the ghost: produced by the old grouping rule, owns no asset any more
         ("phone_kulikov_IMG", "phone_kulikov", 0, "2024-04-27T15:12:57+00:00", 1554.0, 302)])
    conn.executemany(
        "INSERT INTO shots(shot_id, recording_id, media_kind, start_s, end_s, status) "
        "VALUES (?,?,?,?,?,?)",
        [("phone_kulikov_IMG#0000", "phone_kulikov_IMG", "video", 0.0, 20.0, "shortlisted"),
         ("phone_kulikov_IMG_0001#0000", "phone_kulikov_IMG_0001", "video", 0.0, 4.0, "candidate")])
    conn.execute("INSERT INTO timeline(slot_index, act, kind, shot_id, t_in, t_out) "
                 "VALUES (0, 1, 'video', 'phone_kulikov_IMG#0000', 0.0, 4.0)")
    conn.execute("INSERT INTO stage_units(stage, unit_id, status) "
                 "VALUES ('S03', 'proxy:phone_kulikov_IMG', 'done')")
    conn.commit()
    for name in ("proxies/phone_kulikov_IMG_eq.mp4", "audio/phone_kulikov_IMG.wav",
                 "transcripts/shots/phone_kulikov_IMG_0005.json",
                 "proxies/phone_kulikov_IMG_0001_eq.mp4",
                 "proxies/camera_20240101_000000_eq.mp4"):      # a proxy the DB never heard of
        (work / name).write_bytes(b"x")
    return conn, work


def _plan(conn, work):
    recordings = [dict(r) for r in conn.execute("SELECT * FROM recordings")]
    assets = [dict(r) for r in conn.execute("SELECT * FROM assets")]
    shots = [dict(r) for r in conn.execute("SELECT shot_id, recording_id, asset_id FROM shots")]
    return prune.plan_prune(recordings=recordings, assets=assets, shots=shots, work_root=work)


def test_plan_finds_the_ghost_and_its_artefacts(tmp_path):
    conn, work = _seed(tmp_path)
    plan = _plan(conn, work)
    assert plan.recordings == ["phone_kulikov_IMG"]
    assert plan.shots == ["phone_kulikov_IMG#0000"]
    names = {p.name for p in plan.files}
    assert names == {"phone_kulikov_IMG_eq.mp4", "phone_kulikov_IMG.wav",
                     "phone_kulikov_IMG_0005.json", "camera_20240101_000000_eq.mp4"}
    assert "phone_kulikov_IMG_0001_eq.mp4" not in names
    assert "phone_kulikov_IMG" in plan.reasons


def test_execute_removes_rows_and_files_and_keeps_the_rest(tmp_path):
    conn, work = _seed(tmp_path)
    report = prune.execute(conn, _plan(conn, work))
    assert report["n_recordings"] == 1 and report["n_shots"] == 1 and report["n_files"] == 4
    assert conn.execute("SELECT COUNT(*) FROM recordings").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM shots").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM timeline").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM stage_units").fetchone()[0] == 0
    assert not (work / "proxies" / "phone_kulikov_IMG_eq.mp4").exists()
    assert (work / "proxies" / "phone_kulikov_IMG_0001_eq.mp4").exists()


def test_dry_run_changes_nothing(tmp_path):
    conn, work = _seed(tmp_path)
    report = prune.execute(conn, _plan(conn, work), dry_run=True)
    assert report["dry_run"] is True
    assert conn.execute("SELECT COUNT(*) FROM recordings").fetchone()[0] == 3
    assert (work / "proxies" / "phone_kulikov_IMG_eq.mp4").exists()


def test_nothing_to_prune_is_an_empty_plan(tmp_path):
    conn, work = _seed(tmp_path)
    prune.execute(conn, _plan(conn, work))
    again = _plan(conn, work)
    assert again.is_empty
