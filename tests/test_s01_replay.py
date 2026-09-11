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
