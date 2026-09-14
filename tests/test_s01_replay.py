"""A manifest-only fix must not cost an hour of clock solving.

build_manifest() clears created_at_utc, because the corrected time depends on
offsets it does not know. If the clock step is skipped those stay NULL and every
asset drops out of the film, so S01 replays the offsets it already recorded.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from datetime import datetime, timezone

import pytest

from nepal import db
from nepal.stages.s01_probe import apply_offsets, stored_offsets


@pytest.fixture()
def conn(tmp_path):
    c = db.init(tmp_path / "nepal.sqlite")
    yield c
    c.close()


def add(conn, asset_id, source, created_at):
    conn.execute("INSERT INTO assets(asset_id, s3_key, source, kind, created_at, "
                 "created_at_utc) VALUES (?,?,?,?,?,NULL)",
                 (asset_id, f"raw/{asset_id}", source, "photo", created_at))
    conn.commit()


def test_stored_offsets_reads_back_what_the_clock_step_solved(conn):
    db.set_decision(conn, "clock_offset_camera_s", -1234.5, 0.9, "gcc-phat")
    db.set_decision(conn, "clock_offset_kulikov_s", 0.0, 1.0, "GPS-derived")
    db.set_decision(conn, "fov_deg", 196.0, 0.8, "solve")
    assert stored_offsets(conn) == {"camera": -1234.5, "kulikov": 0.0}


def test_stored_offsets_is_empty_before_the_clocks_are_solved(conn):
    assert stored_offsets(conn) == {}


def test_offsets_tolerate_a_non_numeric_decision(conn):
    """decisions.value is TEXT, so a key matching the pattern need not parse."""
    db.set_decision(conn, "clock_offset_camera_s", "needs manual review")
    db.set_decision(conn, "clock_offset_keller_s", 60.0)
    assert stored_offsets(conn) == {"keller": 60.0}


def test_replaying_the_offsets_refills_created_at_utc(conn):
    db.set_decision(conn, "clock_offset_camera_s", 3600.0)
    add(conn, "cam", "camera", "2024:05:06 04:00:00+05:45")
    add(conn, "phone", "phone_kulikov", "2024:05:06 04:00:00+05:45")

    assert apply_offsets(conn, stored_offsets(conn)) == 2

    got = dict(conn.execute("SELECT source, created_at_utc FROM assets").fetchall())
    # 04:00 Nepal is 22:15 UTC the day before; the camera runs an hour fast
    assert got["phone_kulikov"] == "2024-05-05T22:15:00+00:00"
    assert got["camera"] == "2024-05-05T23:15:00+00:00"


def test_telegram_stamps_are_never_shifted(conn):
    """Telegram timestamps come from the server, not a device clock."""
    db.set_decision(conn, "clock_offset_camera_s", 3600.0)
    add(conn, "tg", "telegram", "2024:05:06 04:00:00+05:45")
    apply_offsets(conn, stored_offsets(conn))
    assert conn.execute("SELECT created_at_utc FROM assets").fetchone()[0] \
        == "2024-05-05T22:15:00+00:00"


def test_an_asset_with_no_device_stamp_is_left_for_s02(conn):
    """Telegram strips EXIF, so those assets take their time from the message."""
    add(conn, "stripped", "telegram", None)
    assert apply_offsets(conn, {}) == 0
    assert conn.execute("SELECT created_at_utc FROM assets").fetchone()[0] is None


# -- rows for files that no longer exist --------------------------------

def test_orphan_asset_rows_are_removed_and_their_message_link_cleared(conn):
    """Upsert never removes, so a deleted file is counted in every report
    forever. messages.media_asset references assets, so the link goes first."""
    from nepal import db as _db
    _db.upsert(conn, "assets", ["asset_id"],
               [{"asset_id": "gone", "s3_key": "raw/gone.jpg", "source": "telegram",
                 "kind": "photo"},
                {"asset_id": "kept", "s3_key": "raw/kept.jpg", "source": "telegram",
                 "kind": "photo"}])
    conn.execute("INSERT INTO messages(msg_id, ts_utc, media_asset) VALUES "
                 "('m1','2024-05-06T00:00:00+00:00','gone')")
    conn.commit()

    keep = {"kept"}
    orphans = [r["asset_id"] for r in conn.execute("SELECT asset_id FROM assets")
               if r["asset_id"] not in keep]
    rows = [(a,) for a in orphans]
    conn.executemany("UPDATE messages SET media_asset=NULL WHERE media_asset=?", rows)
    conn.executemany("DELETE FROM assets WHERE asset_id=?", rows)
    conn.commit()

    assert orphans == ["gone"]
    assert [r[0] for r in conn.execute("SELECT asset_id FROM assets")] == ["kept"]
    # the message survives; only its broken link is dropped
    assert conn.execute("SELECT media_asset FROM messages").fetchone()[0] is None
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


# -- S01.2 stale recordings ------------------------------------------
def _cfg():
    from nepal.config import Config
    return Config({"probe": {"max_chapter_gap_s": 1.0}})


def _video(conn, asset_id, name, rid=None, when="2024-05-06T08:00:00+00:00"):
    conn.execute("INSERT INTO assets(asset_id, s3_key, source, kind, container, "
                 "duration_s, created_at, created_at_utc, recording_id) "
                 "VALUES (?,?,?,?,?,?,?,?,?)",
                 (asset_id, f"raw/media_from_camera/{name}", "camera",
                  "video360", "mp4", 10.0, when, when, rid))
    conn.commit()


def test_a_recording_the_grouping_no_longer_produces_is_removed(conn):
    """An upsert never removes. Three recordings from the old rule -- the one
    that collapsed every phone clip into a single take -- outlived it and were
    still being reported as S03.1 failures two runs later."""
    from nepal.stages.s01_probe import group_chapters
    conn.execute("INSERT INTO recordings(recording_id, source, is_360, "
                 "start_utc, duration_s, asset_count) VALUES "
                 "('phone_keller_IMG','phone',0,'2024-05-06T08:00:00+00:00',1.0,7)")
    conn.commit()
    _video(conn, "a1", "VID_20240506_080000_00_001.mp4")

    rep = group_chapters(_cfg(), conn)
    assert "phone_keller_IMG" in rep["removed"]
    assert [r[0] for r in conn.execute("SELECT recording_id FROM recordings")] \
        == ["camera_20240506_080000"]
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_a_stale_recording_that_still_owns_assets_is_kept_and_reported(conn):
    """Deleting it would take its assets' only link to a moment with it. That
    is a bug in the grouping, not something to paper over silently."""
    from nepal.stages.s01_probe import group_chapters
    conn.execute("INSERT INTO recordings(recording_id, source, is_360, "
                 "start_utc, duration_s, asset_count) VALUES "
                 "('legacy','phone',0,'2024-05-06T08:00:00+00:00',1.0,1)")
    conn.commit()
    _video(conn, "a1", "VID_20240506_080000_00_001.mp4")
    conn.execute("INSERT INTO assets(asset_id, s3_key, source, kind, recording_id) "
                 "VALUES ('orphan','raw/x.jpg','phone','photo','legacy')")
    conn.commit()

    rep = group_chapters(_cfg(), conn)
    assert rep["stale_still_used"] == ["legacy"]
    assert rep["removed"] == []


def test_shots_of_a_removed_recording_go_with_it(conn):
    from nepal.stages.s01_probe import group_chapters
    conn.execute("INSERT INTO recordings(recording_id, source, is_360, "
                 "start_utc, duration_s, asset_count) VALUES "
                 "('gone','phone',0,'2024-05-06T08:00:00+00:00',1.0,1)")
    conn.execute("INSERT INTO shots(shot_id, recording_id, media_kind, start_s, "
                 "end_s) VALUES ('gone#0000','gone','video',0.0,3.0)")
    conn.commit()
    _video(conn, "a1", "VID_20240506_080000_00_001.mp4")

    group_chapters(_cfg(), conn)
    assert conn.execute("SELECT COUNT(*) FROM shots").fetchone()[0] == 0
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
