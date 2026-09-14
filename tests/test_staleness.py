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
from nepal.freshness import (code_mtime, stale_units, warn_if_stale,
                             rerun_command, EXPENSIVE)


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
    assert stale_units(conn) == [("S01", "manifest"), ("S02", "acts")]


def test_stale_units_can_be_narrowed_to_one_stage(conn):
    code = code_mtime()
    mark(conn, "S01", "manifest", code - timedelta(hours=3))
    mark(conn, "S02", "acts", code - timedelta(hours=3))
    assert stale_units(conn, "S02") == [("S02", "acts")]


def test_a_unit_that_ran_after_the_code_was_written_is_current(conn):
    code = code_mtime()
    mark(conn, "S01", "manifest", code + timedelta(minutes=5))
    assert stale_units(conn) == []


def test_only_the_stale_units_are_named(conn):
    code = code_mtime()
    mark(conn, "S01", "manifest", code - timedelta(hours=3))
    mark(conn, "S02", "acts", code + timedelta(minutes=1))
    assert stale_units(conn) == [("S01", "manifest")]


def test_a_failed_unit_is_not_reported_as_merely_stale(conn):
    code = code_mtime()
    conn.execute("INSERT INTO stage_units(stage, unit_id, status, updated_at) "
                 "VALUES ('S01','fov','failed',?)",
                 ((code - timedelta(hours=3)).isoformat(),))
    conn.commit()
    assert stale_units(conn) == []


def test_nothing_recorded_means_nothing_stale(conn):
    assert stale_units(conn) == []


# -- the warning a re-run without --force has to carry ------------------

class _Log:
    def __init__(self): self.messages = []
    def warning(self, fmt, *args): self.messages.append(fmt % args)


def test_a_rerun_without_force_says_it_will_skip_the_stale_work(conn):
    """`nepal s01` after a fix prints a full report and changes nothing: every
    unit is already done, so every unit is skipped. Silence there cost three
    rounds of debugging the wrong layer."""
    code = code_mtime()
    mark(conn, "S01", "manifest", code - timedelta(hours=3))
    mark(conn, "S01", "clock", code - timedelta(hours=3))
    log = _Log()

    got = warn_if_stale(log, conn, "S01", force=False, rerun_hint="nepal s01 --force")

    assert got == ["clock", "manifest"]
    assert len(log.messages) == 1
    text = log.messages[0]
    assert "SKIPPED" in text and "will not change them" in text
    assert "nepal s01 --force" in text
    assert "clock" in text and "manifest" in text


def test_force_needs_no_warning_because_nothing_is_skipped(conn):
    code = code_mtime()
    mark(conn, "S01", "manifest", code - timedelta(hours=3))
    log = _Log()
    assert warn_if_stale(log, conn, "S01", force=True, rerun_hint="x") == []
    assert log.messages == []


def test_current_work_is_not_warned_about(conn):
    code = code_mtime()
    mark(conn, "S01", "manifest", code + timedelta(minutes=1))
    log = _Log()
    assert warn_if_stale(log, conn, "S01", force=False, rerun_hint="x") == []
    assert log.messages == []


def test_another_stages_stale_work_is_not_this_stages_warning(conn):
    code = code_mtime()
    mark(conn, "S02", "acts", code - timedelta(hours=3))
    log = _Log()
    assert warn_if_stale(log, conn, "S01", force=False, rerun_hint="x") == []
    assert log.messages == []


def test_an_unparseable_timestamp_is_skipped_not_crashed(conn):
    conn.execute("INSERT INTO stage_units(stage, unit_id, status, updated_at) "
                 "VALUES ('S01','manifest','done','not a date')")
    conn.commit()
    assert stale_units(conn) == []


# -- the command the warning names has to redo the work ------------------

def test_a_stale_manifest_takes_the_cheap_path():
    assert rerun_command([("S01", "manifest")]) == \
        "nepal s01 --force --skip-fov --skip-clock"


def test_a_stale_clock_solve_must_not_be_told_to_skip_the_clock_solve():
    """The cheap path skips exactly fov and clock. Naming it for those units
    sends the operator in a circle: run this, see the same warning, repeat."""
    cmd = rerun_command([("S01", "clock"), ("S01", "fov")])
    assert cmd == "nepal s01 --force"
    assert "--skip-clock" not in cmd and "--skip-fov" not in cmd


def test_a_mix_still_covers_the_expensive_unit():
    cmd = rerun_command([("S01", "manifest"), ("S01", "clock")])
    assert cmd == "nepal s01 --force"


def test_both_stages_are_named_when_both_are_stale():
    cmd = rerun_command([("S01", "manifest"), ("S02", "acts")])
    assert cmd == "nepal s01 --force --skip-fov --skip-clock && nepal s02 --force"


def test_only_s02_stale_does_not_rerun_s01():
    assert rerun_command([("S02", "acts")]) == "nepal s02 --force"


def test_nothing_stale_names_no_command():
    assert rerun_command([]) == ""


def test_the_expensive_units_are_the_ones_the_skip_flags_skip():
    assert EXPENSIVE["S01"] == {"fov", "clock"}
