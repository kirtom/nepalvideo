"""A table built by the previous version of the code must say so.

Three rounds of debugging turned on one question -- is this output from the code
now installed, or from the code it replaced -- and nothing in the report could
answer it.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from datetime import datetime, timedelta, timezone

import pytest

from nepal import db
from nepal.stages.s02_spine import code_mtime, stale_units


@pytest.fixture()
def conn(tmp_path):
    c = db.init(tmp_path / "nepal.sqlite")
    yield c
    c.close()


def mark(conn, stage, unit, when):
    conn.execute("INSERT INTO stage_units(stage, unit_id, status, updated_at) "
                 "VALUES (?,?,'done',?)", (stage, unit, when.isoformat()))
    conn.commit()


def test_code_mtime_finds_the_installed_package():
    got = code_mtime()
    assert got is not None and got.tzinfo is not None


def test_a_unit_that_ran_before_the_code_was_written_is_stale(conn):
    code = code_mtime()
    mark(conn, "S01", "manifest", code - timedelta(hours=3))
    mark(conn, "S02", "acts", code - timedelta(hours=3))
    assert stale_units(conn) == ["S01.manifest", "S02.acts"]


def test_a_unit_that_ran_after_the_code_was_written_is_current(conn):
    code = code_mtime()
    mark(conn, "S01", "manifest", code + timedelta(minutes=5))
    assert stale_units(conn) == []


def test_only_the_stale_units_are_named(conn):
    code = code_mtime()
    mark(conn, "S01", "manifest", code - timedelta(hours=3))
    mark(conn, "S02", "acts", code + timedelta(minutes=1))
    assert stale_units(conn) == ["S01.manifest"]


def test_a_failed_unit_is_not_reported_as_merely_stale(conn):
    code = code_mtime()
    conn.execute("INSERT INTO stage_units(stage, unit_id, status, updated_at) "
                 "VALUES ('S01','fov','failed',?)",
                 ((code - timedelta(hours=3)).isoformat(),))
    conn.commit()
    assert stale_units(conn) == []


def test_nothing_recorded_means_nothing_stale(conn):
    assert stale_units(conn) == []
