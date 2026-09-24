"""The idle watchdog: the decision, and the timer that carries it."""
import os
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from datetime import datetime, timedelta, timezone

from nepal.cloud import watchdog

NOW = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
BOOT = NOW - timedelta(hours=8)
WRAP = ("bash -c . ~/.profile 2>/dev/null; cd /data/projects/nepalvideo && git fetch -q "
        "&& .venv/bin/pip install -q -e '.[dev]' && (.venv/bin/nepal s03)")
SELF = "/data/projects/nepalvideo/.venv/bin/python -m nepal.cloud.watchdog"


def _decide(**kw):
    args = dict(jobs_running=0, last_activity=None, boot=BOOT, now=NOW,
                idle_stop_min=30, stuck_job_min=240)
    return watchdog.decide(**{**args, **kw})


def test_a_running_job_keeps_the_box_up_however_old_the_stamp_is():
    """A job launched detached returns at once: its stamp is hours old
    while it works, so the process is what answers -- up to the wedge
    bound, which the next test is about."""
    stop, why = _decide(jobs_running=1, last_activity=NOW - timedelta(hours=3))
    assert stop is False and "1 job" in why


def test_a_job_that_has_written_nothing_for_hours_is_wedged_not_working():
    """S03.1 is the longest run here and logs per recording. Without this
    bound a hung process is an unbounded bill with a reassuring cause."""
    stop, why = _decide(jobs_running=2, last_activity=NOW - timedelta(minutes=241))
    assert stop is True and "wedged" in why and "2 job" in why


def test_the_box_stops_only_once_the_timeout_has_passed():
    """Between two sub-steps of a stage there is no venv process for a
    second, so the stamp is what answers."""
    inside = _decide(last_activity=NOW - timedelta(minutes=29))
    assert inside[0] is False and "29 min of 30" in inside[1]
    stop, why = _decide(last_activity=NOW - timedelta(minutes=31))
    assert stop is True and "31 min" in why and "2026-09-24T11:29:00" in why


def test_the_quiet_is_measured_from_this_boot_at_the_earliest():
    """A box restarted with `up` inherits a stamp from the session before
    it -- days old -- and the timer's first tick lands while the bootstrap
    is still inside the raw/ rsync, with no venv process to see. Without
    the floor the watchdog stops the box it is booting, `up` waits 1500 s
    for a READY that never comes, and the retry races it again."""
    booting = _decide(last_activity=NOW - timedelta(days=4), boot=NOW - timedelta(minutes=2))
    assert booting[0] is False and "2 min of 30" in booting[1]
    # and a box that has never run anything is still bounded: the boot is
    # the activity it has
    assert _decide(last_activity=None, boot=NOW - timedelta(minutes=2))[0] is False
    assert _decide(last_activity=None, boot=NOW - timedelta(minutes=31))[0] is True


def test_the_watchdog_does_not_count_itself_as_a_job():
    """It runs from the same venv as everything it looks for."""
    assert watchdog.jobs([WRAP, SELF, "/usr/bin/sshd -D", "sleep 60"]) == [WRAP]


def test_boot_time_is_before_now_and_reads_the_machine():
    assert watchdog.boot_time(NOW) <= NOW
    assert watchdog.boot_time().tzinfo is timezone.utc


def test_last_activity_is_the_newest_of_the_stamp_and_the_job_logs(tmp_path):
    reports = tmp_path / "reports"
    (reports / "remote_jobs").mkdir(parents=True)
    assert watchdog.last_activity(tmp_path) is None
    watchdog.touch(tmp_path)
    old = NOW.timestamp() - 3600
    os.utime(watchdog.stamp_path(tmp_path), (old, old))
    assert watchdog.last_activity(tmp_path) == NOW - timedelta(hours=1)
    # a three-hour stage writes its own log as it goes, and that is activity
    (reports / "remote_jobs" / "s03.log").write_text("S03.6 faces 10/10\n")
    assert watchdog.last_activity(tmp_path) > NOW - timedelta(hours=1)


def test_the_installer_writes_a_timer_that_survives_the_session():
    sh = watchdog.install_sh(repo="/data/projects/nepalvideo", period_min=5)
    assert "/etc/systemd/system/nepal-idle.service" in sh
    assert "/etc/systemd/system/nepal-idle.timer" in sh
    # ExecStart is parsed into words, so its path is quoted; a quoted
    # WorkingDirectory is "not absolute" to systemd and the unit refuses to
    # start -- to the journal, where nothing was looking
    assert "WorkingDirectory=/data/projects/nepalvideo\n" in sh
    assert 'ExecStart="/data/projects/nepalvideo/.venv/bin/python" ' \
           '-m nepal.cloud.watchdog' in sh
    assert "OnUnitActiveSec=5min" in sh and "OnBootSec=5min" in sh
    assert "systemctl daemon-reload" in sh
    assert sh.strip().endswith("systemctl enable --now nepal-idle.timer")
    # the timeouts are read from the config at every tick, not baked in here
    assert "idle_stop_min" not in sh and "stuck_job_min" not in sh
    assert "OnUnitActiveSec=15min" in watchdog.install_sh(repo="/r", period_min=15)


def test_the_stamp_the_shell_touches_is_the_one_the_watchdog_reads(tmp_path):
    sh = watchdog.touch_sh("/data/projects/nepal_work")
    assert str(watchdog.stamp_path(pathlib.Path("/data/projects/nepal_work"))) in sh
    assert sh.startswith("mkdir -p /data/projects/nepal_work/reports")
    assert watchdog.touch(tmp_path) == watchdog.stamp_path(tmp_path)
