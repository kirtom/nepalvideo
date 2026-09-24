"""The idle watchdog: the decision, and the timer that carries it."""
import os
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from datetime import datetime, timedelta, timezone

from nepal.cloud import watchdog

NOW = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
WRAP = ("bash -c . ~/.profile 2>/dev/null; cd /data/projects/nepalvideo && git fetch -q "
        "&& .venv/bin/pip install -q -e '.[dev]' && (.venv/bin/nepal s03)")
SELF = "/data/projects/nepalvideo/.venv/bin/python -m nepal.cloud.watchdog"


def test_a_running_job_keeps_the_box_up_however_old_the_stamp_is():
    """A job launched detached returns at once: its stamp is hours old
    while it works, so the process is what answers."""
    stop, why = watchdog.decide(jobs_running=1, last_activity=NOW - timedelta(hours=5),
                                now=NOW, idle_stop_min=30)
    assert stop is False and "1 job" in why


def test_the_box_stops_only_once_the_timeout_has_passed():
    """Between two sub-steps of a stage there is no venv process for a
    second, so the stamp is what answers."""
    inside = watchdog.decide(jobs_running=0, last_activity=NOW - timedelta(minutes=29),
                             now=NOW, idle_stop_min=30)
    assert inside[0] is False and "29 min of 30" in inside[1]
    stop, why = watchdog.decide(jobs_running=0, last_activity=NOW - timedelta(minutes=31),
                                now=NOW, idle_stop_min=30)
    assert stop is True and "31 min" in why and "2026-09-24T11:29:00" in why
    # nothing has ever run here: not a box to stop on a guess
    assert watchdog.decide(jobs_running=0, last_activity=None, now=NOW,
                           idle_stop_min=30)[0] is False


def test_the_watchdog_does_not_count_itself_as_a_job():
    """It runs from the same venv as everything it looks for."""
    assert watchdog.jobs([WRAP, SELF, "/usr/bin/sshd -D", "sleep 60"]) == [WRAP]


def test_last_activity_is_the_newest_of_the_stamp_and_the_job_logs(tmp_path):
    reports = tmp_path / "reports"
    (reports / "remote_jobs").mkdir(parents=True)
    assert watchdog.last_activity(tmp_path) is None
    watchdog.stamp_path(tmp_path).write_text("")
    old = NOW.timestamp() - 3600
    os.utime(watchdog.stamp_path(tmp_path), (old, old))
    assert watchdog.last_activity(tmp_path) == NOW - timedelta(hours=1)
    # a three-hour stage writes its own log as it goes, and that is activity
    (reports / "remote_jobs" / "s03.log").write_text("S03.6 faces 10/10\n")
    assert watchdog.last_activity(tmp_path) > NOW - timedelta(hours=1)


def test_the_installer_writes_a_timer_that_survives_the_session():
    sh = watchdog.install_sh(repo="/data/projects/nepalvideo")
    assert "/etc/systemd/system/nepal-idle.service" in sh
    assert "/etc/systemd/system/nepal-idle.timer" in sh
    assert "ExecStart=/data/projects/nepalvideo/.venv/bin/python -m nepal.cloud.watchdog" in sh
    assert "OnUnitActiveSec=5min" in sh and "OnBootSec=5min" in sh
    assert "systemctl daemon-reload" in sh
    assert sh.strip().endswith("systemctl enable --now nepal-idle.timer")
    # the timeout is read from the config at every tick, not baked in here
    assert "idle_stop_min" not in sh


def test_the_stamp_the_shell_touches_is_the_one_the_watchdog_reads():
    sh = watchdog.touch_sh("/data/projects/nepal_work")
    assert str(watchdog.stamp_path(pathlib.Path("/data/projects/nepal_work"))) in sh
    assert sh.startswith("mkdir -p /data/projects/nepal_work/reports")
