"""S03.0 -- photographs become shots, so stills can reach the timeline.

Writes real JPEGs and runs the stage against them: the interesting failures are
in decoding and in the act lookup, neither of which a mocked image exercises.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from nepal import db
from nepal.config import Config
from nepal.stages.s03_process import build_photo_shots

UTC = timezone.utc
TREK = datetime(2024, 4, 28, tzinfo=UTC)
BOUNDS = [
    {"act": 1, "start_utc": (TREK - timedelta(days=180)).isoformat(),
     "end_utc": TREK.isoformat(), "method": "message-phase:planning"},
    {"act": 2, "start_utc": TREK.isoformat(),
     "end_utc": (TREK + timedelta(days=3)).isoformat(), "method": "alt"},
    {"act": 3, "start_utc": (TREK + timedelta(days=3)).isoformat(),
     "end_utc": (TREK + timedelta(days=9)).isoformat(), "method": "alt"},
    {"act": 4, "start_utc": (TREK + timedelta(days=9)).isoformat(),
     "end_utc": (TREK + timedelta(days=10)).isoformat(), "method": "alt"},
    {"act": 5, "start_utc": (TREK + timedelta(days=10)).isoformat(),
     "end_utc": (TREK + timedelta(days=43)).isoformat(), "method": "alt"},
]


def write_jpeg(path, kind="detailed", size=(320, 240)):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(path.name)) % 2**32)
    if kind == "detailed":
        arr = (rng.random((size[1], size[0], 3)) * 255).astype(np.uint8)
    elif kind == "blown":
        arr = np.full((size[1], size[0], 3), 253, dtype=np.uint8)
    else:                                   # flat grey: no detail at all
        arr = np.full((size[1], size[0], 3), 128, dtype=np.uint8)
    Image.fromarray(arr).save(path, quality=95)
    return path


@pytest.fixture()
def project(tmp_path):
    data, work = tmp_path / "data", tmp_path / "work"
    (data / "media_from_phones" / "keller").mkdir(parents=True)
    cfg_path = tmp_path / "pipeline.yaml"
    cfg_path.write_text(
        f"project:\n"
        f"  name: t\n  data_root: {data}\n  work_root: {work}\n"
        f"  db_path: {work}/db/nepal.sqlite\n"
        f"quality_curves:\n"
        f"  camera:   {{min_sharpness: 4.0, max_exposure_pen: 0.15, "
        f"min_stability: 0.35, min_duration_s: 1.5}}\n"
        f"  phone:    {{min_sharpness: 3.5, max_exposure_pen: 0.20, "
        f"min_stability: 0.30, min_duration_s: 1.5}}\n"
        f"  telegram: {{min_sharpness: 2.0, max_exposure_pen: 0.35, "
        f"min_stability: 0.15, min_duration_s: 1.0}}\n"
        f"process:\n  photo_analysis_px: 512\n  photo_slot_s: 3.0\n")
    cfg = Config.load(cfg_path)
    conn = db.init(cfg.db_path)
    db.set_decision(conn, "act_boundaries", json.dumps(BOUNDS))
    yield cfg, conn, data
    conn.close()


def add_photo(conn, data, name, when, *, kind="detailed", curve="phone"):
    rel = f"media_from_phones/keller/{name}"
    write_jpeg(data / rel, kind)
    conn.execute(
        "INSERT INTO assets(asset_id, s3_key, source, kind, quality_curve, "
        "created_at_utc, lat, lon, alt_dem_m, place_name, width, height) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (name, f"raw/{rel}", "phone_keller", "photo", curve, when.isoformat(),
         28.5, 84.6, 3500.0, "Samagaun", 320, 240))
    conn.commit()
    return rel


def test_a_good_photo_becomes_a_shot_in_the_right_act(project):
    cfg, conn, data = project
    add_photo(conn, data, "good.jpg", TREK + timedelta(days=4))   # act 3

    rep = build_photo_shots(cfg, conn)

    assert rep["n_shots"] == 1
    row = conn.execute("SELECT * FROM shots").fetchone()
    assert row["media_kind"] == "photo"
    assert row["act"] == 3
    assert row["recording_id"] is None
    assert row["asset_id"] == "good.jpg"
    assert row["end_s"] == pytest.approx(3.0)
    assert row["place_name"] == "Samagaun"


def test_a_still_has_no_stability_or_motion(project):
    """A photograph beats every clip on stability by not moving, which is not
    a fact about its quality. NULL says so; a flattering default would lie."""
    cfg, conn, data = project
    add_photo(conn, data, "good.jpg", TREK + timedelta(days=4))
    build_photo_shots(cfg, conn)
    row = conn.execute("SELECT stability, motion_mag, audio_lufs FROM shots").fetchone()
    assert row["stability"] is None
    assert row["motion_mag"] is None
    assert row["audio_lufs"] is None


def test_a_blown_out_photo_is_rejected(project):
    cfg, conn, data = project
    add_photo(conn, data, "blown.jpg", TREK + timedelta(days=4), kind="blown")
    rep = build_photo_shots(cfg, conn)
    assert rep["n_shots"] == 0
    assert "below the quality gate" in rep["rejected"]


def test_a_photo_with_no_detail_is_rejected(project):
    cfg, conn, data = project
    add_photo(conn, data, "flat.jpg", TREK + timedelta(days=4), kind="flat")
    assert build_photo_shots(cfg, conn)["n_shots"] == 0


def test_a_photo_outside_every_act_is_skipped_not_measured(project):
    cfg, conn, data = project
    add_photo(conn, data, "late.jpg", TREK + timedelta(days=400))
    rep = build_photo_shots(cfg, conn)
    assert rep["n_shots"] == 0 and rep["n_unplaced"] == 1


def test_a_missing_file_is_counted_rather_than_crashing(project):
    cfg, conn, data = project
    rel = add_photo(conn, data, "gone.jpg", TREK + timedelta(days=4))
    (data / rel).unlink()
    rep = build_photo_shots(cfg, conn)
    assert rep["rejected"].get("missing file") == 1


def test_photos_spread_across_the_acts_they_belong_to(project):
    cfg, conn, data = project
    for i, days in enumerate((-30, 1, 4, 9, 12)):
        add_photo(conn, data, f"p{i}.jpg", TREK + timedelta(days=days, hours=3))
    rep = build_photo_shots(cfg, conn)
    assert rep["n_shots"] == 5
    assert sorted(rep["per_act"]) == [1, 2, 3, 4, 5]


def test_rerunning_updates_rather_than_duplicates(project):
    cfg, conn, data = project
    add_photo(conn, data, "good.jpg", TREK + timedelta(days=4))
    build_photo_shots(cfg, conn)
    build_photo_shots(cfg, conn)
    assert conn.execute("SELECT COUNT(*) FROM shots").fetchone()[0] == 1


def test_undated_photos_are_not_considered(project):
    cfg, conn, data = project
    write_jpeg(data / "media_from_phones/keller/undated.jpg")
    conn.execute("INSERT INTO assets(asset_id, s3_key, source, kind, created_at_utc) "
                 "VALUES ('u','raw/media_from_phones/keller/undated.jpg',"
                 "'phone_keller','photo',NULL)")
    conn.commit()
    assert build_photo_shots(cfg, conn)["n_photos"] == 0


# -- the schema change that lets a photo be a shot at all ---------------

def test_a_shot_must_be_a_recording_or_an_asset_but_not_both(tmp_path):
    conn = db.init(tmp_path / "n.sqlite")
    conn.execute("INSERT INTO recordings(recording_id, source, is_360) "
                 "VALUES ('r1','camera',1)")
    conn.execute("INSERT INTO assets(asset_id, s3_key, source, kind) "
                 "VALUES ('a1','raw/a.jpg','phone_keller','photo')")
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO shots(shot_id, recording_id, asset_id, "
                     "start_s, end_s) VALUES ('s','r1','a1',0,3)")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO shots(shot_id, start_s, end_s) VALUES ('s',0,3)")
    conn.close()


def test_an_old_database_is_migrated_in_place(tmp_path):
    """recording_id was NOT NULL, which SQLite cannot relax with ALTER TABLE."""
    p = tmp_path / "old.sqlite"
    raw = sqlite3.connect(p)
    raw.executescript(
        "CREATE TABLE recordings(recording_id TEXT PRIMARY KEY);"
        "CREATE TABLE shots(shot_id TEXT PRIMARY KEY,"
        " recording_id TEXT NOT NULL REFERENCES recordings(recording_id),"
        " start_s REAL NOT NULL, end_s REAL NOT NULL, status TEXT,"
        " act INTEGER, start_utc TEXT);")
    raw.commit()
    raw.close()

    conn = db.init(p)
    cols = {r[1]: r[3] for r in conn.execute("PRAGMA table_info(shots)")}
    assert not cols["recording_id"], "recording_id must become nullable"
    assert "asset_id" in cols and "media_kind" in cols
    conn.close()


def test_migration_refuses_to_discard_existing_shots(tmp_path):
    """Rebuilding an old table with rows in it would throw away a cut."""
    p = tmp_path / "old.sqlite"
    raw = sqlite3.connect(p)
    raw.executescript(
        "CREATE TABLE shots(shot_id TEXT PRIMARY KEY,"
        " recording_id TEXT NOT NULL, start_s REAL NOT NULL, end_s REAL NOT NULL);"
        "INSERT INTO shots VALUES ('s1','r1',0,4);")
    raw.commit()
    raw.close()
    with pytest.raises(RuntimeError, match="would discard"):
        db.init(p)
