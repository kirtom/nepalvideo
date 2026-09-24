"""The box stops itself when nobody is using it.

The ledger makes an abandoned box visible; this makes it cheap. On
2026-09-20 a session died on a rate limit with nepal-cpu RUNNING and
nothing stopped it for four days: 104 idle hours, ~31 USD, past a ceiling
of 25. The thing that failed was the session, so the fix cannot live in
one -- it runs on the box, as a systemd timer, and outlives whatever
started it.

Three bounds, because each of them alone is wrong.

**No job process**: every command arrives as `bash -c ... .venv/bin/...`,
so a command line naming the venv is work in progress (this module runs
from that venv too, hence the exclusion -- which `remote exec` leans on
as well: the `bash -c` carrying its wrapper script names the venv a dozen
times, and names this module once, so the probe does not see itself).
But a stage between two sub-steps has no venv process for a second.

**No recent output**: the stamp `remote exec` touches around every
command, and the log a long job appends to as it goes. But a job launched
detached returns at once and leaves a stamp that is hours old while it
works -- and a box that has just restarted carries a stamp from the
session before it, days old, while the bootstrap is still inside the raw/
rsync with no venv process to see. So the quiet is measured from this
boot at the earliest.

**A job that writes nothing for hours is wedged, not working**: S03.1
takes three and a half hours and logs per recording. Without that bound a
hung process is an unbounded bill with a reassuring cause.

`shutdown -h now` inside the guest leaves the instance TERMINATED, which
is exactly what `remote up` starts again; it stops the CPU bill and keeps
the disk, which is the whole point of stopping rather than deleting. It
needs no credentials and no compute scope on the box -- calling the GCE
API from here would need both. The hours reach the ledger at the next
`nepal remote` command, which reads GCP's `lastStopTimestamp`.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

log = logging.getLogger(__name__)

UNIT = "nepal-idle"
STAMP_NAME = ".last_activity"
# A command line that names the venv is a nepal job; this module runs from
# the same venv, so it has to be told apart from the work it looks for.
JOB_MARK = ".venv/bin/"
SELF_MARK = "nepal.cloud.watchdog"
# Long enough that an interrupted push cannot hold the box up, short enough
# that the operator is not waiting on it.
PUSH_TIMEOUT_S = 600


def stamp_path(work_root: Path) -> Path:
    return Path(work_root) / "reports" / STAMP_NAME


def touch_sh(remote_work: str) -> str:
    """The shell `remote exec` runs around every command to say `now`."""
    d = f"{remote_work}/reports"
    return f"mkdir -p {d} && touch {d}/{STAMP_NAME}"


def jobs(cmdlines: Iterable[str]) -> list[str]:
    return [c for c in cmdlines if JOB_MARK in c and SELF_MARK not in c]


def decide(*, jobs_running: int, last_activity: datetime | None, boot: datetime,
           now: datetime, idle_stop_min: float,
           stuck_job_min: float) -> tuple[bool, str]:
    """Stop, and why -- or stay up, and why. The whole decision, no I/O.

    The quiet is measured from the newest of the last activity and this
    boot: a restarted box inherits a stamp from the session before it, and
    an activity stamp older than the machine it is on means nothing.
    """
    last = max(last_activity, boot) if last_activity else boot
    quiet_min = (now - last).total_seconds() / 60
    if jobs_running:
        if quiet_min >= float(stuck_job_min):
            return True, (f"{jobs_running} job process(es) but nothing written for "
                          f"{quiet_min:.0f} min (limit {float(stuck_job_min):.0f}): "
                          f"wedged, not working")
        return False, f"{jobs_running} job process(es) running"
    if quiet_min < float(idle_stop_min):
        return False, f"idle {quiet_min:.0f} min of {float(idle_stop_min):.0f}"
    return True, (f"idle {quiet_min:.0f} min (limit {float(idle_stop_min):.0f}), no job "
                  f"running, last activity {last.isoformat(timespec='seconds')}")


def install_sh(*, repo: str, period_min: int) -> str:
    """The shell that installs the timer, rendered rather than shipped as a
    file: `remote up` pipes it over ssh to a box that is already running,
    and the bootstrap runs the same text at boot. The timeouts are not in
    here -- the watchdog reads them from the config in the checkout at
    every tick, which is also why the service names a WorkingDirectory:
    `Config.load()` finds pipeline.yaml by walking up from where it is
    run, and a systemd unit starts in /.

    ExecStart is quoted word by word and WorkingDirectory is not, because
    that is what systemd parses: quoting the directory made the unit
    refuse to start at all ("path is not absolute"), which the timer
    reported to the journal and nowhere else.
    """
    return f"""set -eu
cat > /etc/systemd/system/{UNIT}.service <<'UNIT_EOF'
[Unit]
Description=nepal: stop this box when nobody is using it

[Service]
Type=oneshot
WorkingDirectory={repo}
ExecStart="{repo}/.venv/bin/python" -m nepal.cloud.watchdog
UNIT_EOF
cat > /etc/systemd/system/{UNIT}.timer <<'UNIT_EOF'
[Unit]
Description=nepal idle check, every {int(period_min)} min

[Timer]
OnBootSec={int(period_min)}min
OnUnitActiveSec={int(period_min)}min

[Install]
WantedBy=timers.target
UNIT_EOF
systemctl daemon-reload
systemctl enable --now {UNIT}.timer
"""


# -- the box side, a thin shell around decide() ---------------------------

def _cmdlines() -> list[str]:
    out = []
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            try:
                out.append(p.joinpath("cmdline").read_bytes()
                           .replace(b"\0", b" ").decode(errors="replace").strip())
            except OSError:      # the process ended while we looked at it
                pass
    return out


def boot_time(now: datetime | None = None) -> datetime:
    """When this boot started. Unreadable uptime reads as "just now", which
    keeps the box up: a watchdog that cannot tell how old the machine is
    must not be the one to stop it."""
    now = now or datetime.now(timezone.utc)
    try:
        return now - timedelta(seconds=float(Path("/proc/uptime").read_text().split()[0]))
    except (OSError, ValueError, IndexError):
        return now


def _activity_files(work_root: Path) -> list[Path]:
    reports = Path(work_root) / "reports"
    return [stamp_path(work_root), reports / "remote_jobs.log",
            *sorted((reports / "remote_jobs").glob("*.log"))]


def last_activity(work_root: Path) -> datetime | None:
    times = [p.stat().st_mtime for p in _activity_files(work_root) if p.exists()]
    return datetime.fromtimestamp(max(times), timezone.utc) if times else None


def touch(work_root: Path) -> Path:
    p = stamp_path(work_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()
    return p


def _note(work_root: Path, line: str) -> None:
    """Into the log the next `remote pull` shows. Written by root, so the
    ownership is put back: the log is appended to by the box's user at
    every run, and a root-owned one would break the next."""
    path = Path(work_root) / "reports" / "remote_jobs.log"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(line + "\n")
        st = path.parent.stat()
        os.chown(path, st.st_uid, st.st_gid)
    except OSError as exc:
        log.warning("watchdog: could not write %s: %s", path, exc)


def _push(cfg, work_root: Path) -> None:
    """The bucket, before the box goes: a detached job's output and the
    reason above are only real once they are off this disk.

    The snap gcloud is on the login PATH and not on a systemd unit's, so
    the binary is resolved rather than named -- a bare `gcloud` here was a
    FileNotFoundError swallowed into silence, and 340 MB of a stage's
    output would have gone with the box.
    """
    from nepal.cloud import sync
    bucket = str(cfg.get("cloud.gcp.bucket")).rstrip("/")
    user = str(cfg.get("cloud.gcp.ssh_user", "nepal"))
    exe = shutil.which("gcloud") or "/snap/bin/gcloud"
    args = [exe, *sync.rsync_args(str(work_root), f"{bucket}/work")]
    if os.geteuid() == 0:
        args = ["sudo", "-u", user, *args]      # the user whose gcloud is logged in
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=PUSH_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError) as exc:
        log.error("watchdog: push before stop failed: %s", exc)
        return
    if proc.returncode != 0:
        log.error("watchdog: push before stop failed (rc %s): %s",
                  proc.returncode, (proc.stderr or proc.stdout or "")[-400:].strip())


def _shutdown() -> tuple[int, str]:
    exe = shutil.which("shutdown") or "/sbin/shutdown"
    try:
        proc = subprocess.run([exe, "-h", "now"], capture_output=True, text=True)
    except OSError as exc:
        return 127, str(exc)
    return proc.returncode, (proc.stderr or proc.stdout or "").strip()


def main(argv: Sequence[str] | None = None) -> int:
    from nepal.config import Config
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    argv = list(sys.argv[1:] if argv is None else argv)
    cfg = Config.load()
    work_root = cfg.work_root
    # The bootstrap asks for these two rather than spelling them out, so the
    # unit it installs and the stamp it writes cannot drift from the ones
    # this module renders and reads.
    if argv[:1] == ["install"]:
        print(install_sh(repo=str(cfg.get("cloud.gcp.remote_repo")),
                         period_min=int(cfg.get("cloud.gcp.idle_check_min"))))
        return 0
    if argv[:1] == ["touch"]:
        print(touch(work_root))
        return 0
    if argv[:1] == ["jobs"]:
        # What `remote exec` asks the box before it rsyncs work/ over what a
        # job is writing. Printed rather than counted: the caller logs which
        # job held the pull off, and "nothing" is an empty line.
        print("\n".join(jobs(_cmdlines())))
        return 0
    now = datetime.now(timezone.utc)
    stop, why = decide(jobs_running=len(jobs(_cmdlines())),
                       last_activity=last_activity(work_root), boot=boot_time(now), now=now,
                       idle_stop_min=float(cfg.get("cloud.gcp.idle_stop_min")),
                       stuck_job_min=float(cfg.get("cloud.gcp.stuck_job_min")))
    log.info("watchdog: %s (%s)", "stopping" if stop else "staying up", why)
    if not stop:
        return 0
    # Requested, not done: the box is claimed to be stopped only once
    # something has actually stopped it.
    _note(work_root, f"--- idle stop requested at {now.isoformat(timespec='seconds')}: {why}")
    _push(cfg, work_root)
    rc, err = _shutdown()
    if rc == 0:
        log.info("watchdog: shutdown accepted, the box is going down")
        return 0
    log.error("watchdog: shutdown failed (rc %s): %s", rc, err)
    _note(work_root, f"--- idle stop FAILED at "
                     f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}: "
                     f"rc {rc} {err}")
    _push(cfg, work_root)
    return 1


if __name__ == "__main__":
    sys.exit(main())
