"""The SQLite layer: schema, migrations, and the generic upsert."""
import sqlite3
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest

from nepal import db


def test_upsert_refuses_rows_with_different_keys(tmp_path):
    """Columns come from the first row. A row that lacks a key would have
    that column silently dropped for every row -- which is how 930 video
    shots lost their position: the first row had no `lat`."""
    conn = db.init(tmp_path / "t.sqlite")
    rows = [{"key": "a", "value": "1", "confidence": 1.0, "method": "m"},
            {"key": "b", "value": "2", "method": "m"}]          # no confidence
    with pytest.raises(ValueError) as exc:
        db.upsert(conn, "decisions", ["key"], rows)
    assert "confidence" in str(exc.value)


def test_upsert_writes_every_column_of_uniform_rows(tmp_path):
    conn = db.init(tmp_path / "t.sqlite")
    rows = [{"key": "a", "value": "1", "confidence": 1.0, "method": "m"},
            {"key": "b", "value": "2", "confidence": 0.5, "method": "n"}]
    assert db.upsert(conn, "decisions", ["key"], rows) == 2
    got = {r["key"]: r["confidence"] for r in conn.execute("SELECT key, confidence FROM decisions")}
    assert got == {"a": 1.0, "b": 0.5}


def test_gps_points_and_activities_carry_the_strava_columns(tmp_path):
    conn = db.init(tmp_path / "t.sqlite")
    gcols = {r[1] for r in conn.execute("PRAGMA table_info(gps_points)")}
    assert {"hr_bpm", "alt_baro_m", "activity_id"} <= gcols
    acols = {r[1] for r in conn.execute("PRAGMA table_info(activities)")}
    assert {"activity_id", "name", "start_utc", "end_utc", "hr_max", "n_points"} <= acols


def test_delete_shots_takes_the_slots_with_them(tmp_path):
    """timeline.shot_id references shots and foreign keys are on: a bare
    DELETE on a shot in the cut is refused. Two remote runs died on it."""
    conn = db.init(tmp_path / "t.sqlite")
    conn.execute("INSERT INTO recordings(recording_id, source, is_360) VALUES ('r1', 'camera', 0)")
    conn.execute("INSERT INTO assets(asset_id, s3_key, source, kind) VALUES ('a1', 'k', 'phone_keller', 'photo')")
    conn.executemany("INSERT INTO shots(shot_id, recording_id, asset_id, media_kind, start_s, end_s) "
                     "VALUES (?,?,?,?,0,3)", [("r1#0", "r1", None, "video"),
                                             ("photo_a1", None, "a1", "photo")])
    conn.executemany("INSERT INTO timeline(slot_index, act, kind, shot_id, t_in, t_out) "
                     "VALUES (?,1,?,?,0,3)", [(0, "video", "r1#0"), (1, "photo", "photo_a1")])
    conn.commit()
    assert db.delete_shots(conn, recording_id="r1") == 1
    assert db.delete_shots(conn, asset_id="a1") == 1
    assert conn.execute("SELECT COUNT(*) FROM shots").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM timeline").fetchone()[0] == 0
    with pytest.raises(ValueError):
        db.delete_shots(conn)


def test_assets_carry_the_heading_columns(tmp_path):
    conn = db.init(tmp_path / "t.sqlite")
    cols = {r[1] for r in conn.execute("PRAGMA table_info(assets)")}
    assert {"heading_deg", "heading_ref", "pos_error_m", "focal_35mm"} <= cols


def test_a_fresh_db_has_the_voice_spine_columns_and_table(tmp_path):
    """Film v2 section 3: segment times live in the row, a hallucinated shot
    says so, and the beat sheet has a table of its own -- `story_beats`,
    because `beats` is the music grid S02.7 writes and S05 reads."""
    conn = db.init(tmp_path / "t.sqlite")
    cols = {r[1] for r in conn.execute("PRAGMA table_info(shots)")}
    assert {"transcript_json", "hallucinated"} <= cols
    beat_cols = {r[1] for r in conn.execute("PRAGMA table_info(story_beats)")}
    assert {"beat_id", "kind", "act", "shot_id", "msg_id", "src_in", "src_out",
            "text", "levity", "effect", "rationale", "rank"} <= beat_cols
    music = {r[1] for r in conn.execute("PRAGMA table_info(beats)")}
    assert music == {"track_id", "t_s", "is_downbeat"}          # untouched


def test_an_older_db_gains_the_voice_spine_columns(tmp_path):
    import sqlite3
    path = tmp_path / "old.sqlite"
    raw = sqlite3.connect(path)
    # the columns the schema's indexes need, and nothing from Film v2
    raw.executescript("CREATE TABLE shots (shot_id TEXT PRIMARY KEY, recording_id TEXT, "
                      "asset_id TEXT, media_kind TEXT, start_s REAL, end_s REAL, "
                      "start_utc TEXT, act INTEGER, status TEXT);")
    raw.execute("INSERT INTO shots VALUES ('r#0001','r',NULL,'video',0,5,NULL,2,'candidate')")
    raw.commit()
    raw.close()
    conn = db.init(path)
    row = conn.execute("SELECT transcript_json, hallucinated FROM shots").fetchone()
    assert row["transcript_json"] is None and row["hallucinated"] is None
    assert conn.execute("SELECT COUNT(*) FROM story_beats").fetchone()[0] == 0


def test_a_fresh_db_has_the_picture_and_audio_tables(tmp_path):
    conn = db.init(tmp_path / "n.sqlite")
    ordered_cols = tuple(r[1] for r in conn.execute("PRAGMA table_info(timeline)"))
    cols = set(ordered_cols)
    assert {"kind", "secondary_shot_id", "secondary_src_in", "motion", "speed",
            "transition", "beat_id", "scene_id"} <= cols
    # TIMELINE_V2_COLUMNS is read by writers as the insert's column order; it
    # must not drift from the DDL it is meant to describe.
    assert db.TIMELINE_V2_COLUMNS == ordered_cols
    assert {r[1] for r in conn.execute("PRAGMA table_info(audio_cues)")} >= {
        "cue_id", "track", "t_in", "t_out", "source", "src_in", "src_out",
        "gain_lufs", "fade_in_s", "fade_out_s", "beat_id"}
    assert {r[1] for r in conn.execute("PRAGMA table_info(overlays)")} >= {
        "overlay_id", "kind", "t_in", "t_out", "payload", "asset_path"}


def test_an_older_timeline_is_rebuilt_because_it_is_derived(tmp_path):
    """timeline is DELETEd and rewritten by every S05 run; a v1 table is
    dropped and recreated rather than migrated column by column."""
    p = tmp_path / "old.sqlite"
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE timeline (slot_index INTEGER PRIMARY KEY, act INTEGER, "
              "shot_id TEXT, msg_id TEXT, t_in REAL, t_out REAL, src_in REAL, src_out REAL, "
              "yaw REAL, transition TEXT)")
    c.execute("INSERT INTO timeline(slot_index, act) VALUES (0, 1)")
    c.commit(); c.close()
    conn = db.init(p)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(timeline)")}
    assert "kind" in cols and "beat_id" in cols
    assert conn.execute("SELECT COUNT(*) FROM timeline").fetchone()[0] == 0
