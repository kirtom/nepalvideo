"""Progress reporting: a twenty-minute stage must not run in silence."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import io
import logging
import time

import pytest

from nepal.util import progress as prog
from nepal.util.progress import (Progress, ProgressAwareHandler, fmt_duration,
                                 heartbeat, render_bar, track)


class FakeTTY(io.StringIO):
    def isatty(self):
        return True


class NotATTY(io.StringIO):
    def isatty(self):
        return False


# -- formatting ---------------------------------------------------------

def test_durations_are_readable_not_raw_floats():
    assert fmt_duration(7) == "0:07"
    assert fmt_duration(325) == "5:25"
    assert fmt_duration(4000) == "1:06:40"


def test_an_unknowable_eta_says_so_rather_than_printing_inf():
    assert fmt_duration(float("inf")) == "--:--"
    assert fmt_duration(-1) == "--:--"


def test_the_bar_reports_position_rate_and_eta():
    line = render_bar("hashing", 300, 1364, 90.0, width=100)
    assert "300/1364" in line and "22%" in line
    assert "3.3/s" in line
    assert "eta 5:19" in line


def test_the_bar_fills_as_the_work_completes():
    early = render_bar("x", 1, 100, 1.0, width=100)
    late = render_bar("x", 99, 100, 1.0, width=100)
    assert early.count("#") < late.count("#")
    assert early.count(".") > late.count(".")


def test_an_unknown_total_reports_a_count_and_elapsed():
    line = render_bar("scanning", 88, None, 12.0, width=100)
    assert "88" in line and "0:12" in line and "%" not in line


def test_the_line_never_exceeds_the_terminal_width():
    for width in (30, 52, 80, 200):
        assert len(render_bar("a long label here", 300, 1364, 90.0,
                              width=width, note="some_file.heic")) < width


def test_nothing_divides_by_zero_at_the_very_start():
    assert render_bar("x", 0, 100, 0.0, width=80)
    assert render_bar("x", 0, 0, 0.0, width=80)


# -- drawing on a terminal ---------------------------------------------

def test_a_terminal_gets_one_redrawn_line():
    out = FakeTTY()
    with Progress("hashing", 10, stream=out, min_interval_s=0) as p:
        for _ in range(10):
            p.step()
    text = out.getvalue()
    assert "\r" in text, "a terminal should be redrawn in place"
    assert text.count("\n") == 0, "redrawing must not scroll the terminal"
    assert "10/10" in text


def test_the_line_is_wiped_when_the_work_finishes():
    out = FakeTTY()
    with Progress("hashing", 3, stream=out, min_interval_s=0) as p:
        for _ in range(3):
            p.step()
    assert out.getvalue().endswith("\r\033[2K"), "a stale bar must not be left behind"


def test_a_pipe_gets_periodic_lines_and_no_escape_codes(caplog):
    out = NotATTY()
    with caplog.at_level(logging.INFO):
        with Progress("hashing", 100, stream=out, log_every_s=0.0) as p:
            for _ in range(5):
                p.step()
    assert out.getvalue() == "", "a log file must not receive redraw codes"
    assert any("hashing" in r.message for r in caplog.records)


def test_the_finished_total_is_logged_even_on_a_terminal(caplog):
    """The bar is transient; a run scrolled back should still say what it did."""
    out = FakeTTY()
    with caplog.at_level(logging.INFO):
        with Progress("hashing", 4, stream=out, min_interval_s=0) as p:
            for _ in range(4):
                p.step()
    assert any("4/4" in r.getMessage() for r in caplog.records)


def test_closing_twice_is_harmless():
    p = Progress("x", 2, stream=FakeTTY())
    p.close()
    p.close()


def test_progress_can_be_disabled_by_environment(monkeypatch):
    monkeypatch.setenv("NEPAL_NO_PROGRESS", "1")
    out = FakeTTY()
    with Progress("hashing", 3, stream=out, min_interval_s=0) as p:
        p.step()
    assert out.getvalue() == ""


def test_track_wraps_an_iterable_and_yields_everything():
    out = FakeTTY()
    got = list(track(range(6), "counting", stream=out, min_interval_s=0))
    assert got == list(range(6))
    assert "6/6" in out.getvalue()


def test_track_handles_a_generator_with_no_length():
    out = FakeTTY()
    assert list(track((i for i in range(4)), "gen", stream=out)) == [0, 1, 2, 3]


# -- logging through a live bar ----------------------------------------

def test_a_log_record_does_not_land_inside_the_bar():
    """Both go to stderr. Without this the message is embedded in the bar."""
    bar_out, log_out = FakeTTY(), io.StringIO()
    handler = ProgressAwareHandler(log_out)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("test.progress.interleave")
    logger.handlers = [handler]
    logger.propagate = False

    with Progress("hashing", 10, stream=bar_out, min_interval_s=0) as p:
        p.step()
        before = bar_out.getvalue()
        logger.warning("a file went missing")
        after = bar_out.getvalue()

    assert "a file went missing" in log_out.getvalue()
    assert after.startswith(before), "the bar must not be rewritten over"
    # the handler wiped the line, then redrew it
    assert after[len(before):].startswith("\r\033[2K")
    assert "1/10" in after[len(before):]


def test_the_handler_is_harmless_with_no_bar_running():
    log_out = io.StringIO()
    handler = ProgressAwareHandler(log_out)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("test.progress.nobar")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)     # else the root level filters it out
    prog._active = None
    logger.info("plain")
    assert log_out.getvalue().strip() == "plain"


# -- the long opaque call ----------------------------------------------

def test_heartbeat_reports_elapsed_while_something_blocks():
    """exiftool takes fifteen minutes and offers nothing to count."""
    out = FakeTTY()
    with heartbeat("exiftool", interval_s=0.01, stream=out):
        time.sleep(0.06)
    assert "exiftool" in out.getvalue()


def test_heartbeat_clears_up_after_itself():
    out = FakeTTY()
    with heartbeat("exiftool", interval_s=0.01, stream=out):
        time.sleep(0.03)
    assert out.getvalue().endswith("\r\033[2K")


def test_heartbeat_logs_the_total_time(caplog):
    with caplog.at_level(logging.INFO):
        with heartbeat("exiftool", interval_s=5.0, stream=NotATTY()):
            pass
    assert any("exiftool" in r.getMessage() for r in caplog.records)


def test_heartbeat_stops_its_thread_when_the_body_raises():
    import threading
    before = threading.active_count()
    with pytest.raises(ValueError):
        with heartbeat("boom", interval_s=0.01, stream=FakeTTY()):
            raise ValueError("boom")
    time.sleep(0.05)
    assert threading.active_count() <= before
