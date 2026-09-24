"""The box stops itself when nobody is using it.

The ledger makes an abandoned box visible; this makes it cheap. On
2026-09-20 a session died on a rate limit with nepal-cpu RUNNING and
nothing stopped it for four days: 104 idle hours, ~31 USD, past a ceiling
of 25. The thing that failed was the session, so the fix cannot live in
one -- it runs on the box, as a systemd timer, and outlives whatever
started it.

Two conditions, both required, because either alone is wrong. **No job
process**: every command arrives as `bash -c ... .venv/bin/...`, so a
command line naming the venv is work in progress (this module runs from
that venv too, hence the exclusion). But a stage between two sub-steps has
no venv process for a second. **No recent output**: the stamp `remote
exec` touches around every command, and the log a long job appends to as
it goes. But a job launched detached returns at once and leaves a stamp
that is hours old while it works. Together they are right in both cases.

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
import subprocess
import sys
from datetime import datetime, timezone
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


def decide(*, jobs_running: int, last_activity: datetime | None, now: datetime,
           idle_stop_min: float) -> tuple[bool, str]:
    """Stop, and why -- or stay up, and why. The whole decision, no I/O."""
    if jobs_running:
        return False, f"{jobs_running} job process(es) running"
    if last_activity is None:
        # Nothing has ever run here and nothing says when the box woke: an
        # unexplained box is not one to stop from a guess.
        return False, "no activity stamp yet"
    idle_min = (now - last_activity).total_seconds() / 60
    if idle_min < float(idle_stop_min):
        return False, f"idle {idle_min:.0f} min of {float(idle_stop_min):.0f}"
    return True, (f"idle {idle_min:.0f} min (limit {float(idle_stop_min):.0f}), no job "
                  f"running, last activity {last_activity.isoformat(timespec='seconds')}")


def install_sh(*, repo: str, period_min: int = 5) -> str:
    """The shell that installs the timer, rendered rather than shipped as a
    file: `remote up` pipes it over ssh to a box that is already running,
    and the bootstrap runs the same text at boot. The timeout is not in
    here -- the watchdog reads it from the config in the checkout, so
    changing it needs no reinstall."""
    return f"""set -eu
cat > /etc/systemd/system/{UNIT}.service <<'UNIT_EOF'
[Unit]
Description=nepal: stop this box when nobody is using it

[Service]
Type=oneshot
WorkingDirectory={repo}
ExecStart={repo}/.venv/bin/python -m nepal.cloud.watchdog
UNIT_EOF
cat > /etc/systemd/system/{UNIT}.timer <<'UNIT_EOF'
[Unit]
Description=nepal idle check, every {period_min} min

[Timer]
OnBootSec={period_min}min
OnUnitActiveSec={period_min}min

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


def _activity_files(work_root: Path) -> list[Path]:
    reports = Path(work_root) / "reports"
    return [stamp_path(work_root), reports / "remote_jobs.log",
            *sorted((reports / "remote_jobs").glob("*.log"))]


def last_activity(work_root: Path) -> datetime | None:
    times = [p.stat().st_mtime for p in _activity_files(work_root) if p.exists()]
    return datetime.fromtimestamp(max(times), timezone.utc) if times else None


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
    """The bucket, before the box goes: the reason above is only useful if
    the next `remote pull` can read it."""
    from nepal.cloud import sync
    bucket = str(cfg.get("cloud.gcp.bucket")).rstrip("/")
    user = str(cfg.get("cloud.gcp.ssh_user", "nepal"))
    args = ["gcloud", *sync.rsync_args(str(work_root), f"{bucket}/work")]
    if os.geteuid() == 0:
        args = ["sudo", "-u", user, *args]      # the user whose gcloud is logged in
    try:
        subprocess.run(args, capture_output=True, timeout=PUSH_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("watchdog: push before stop failed: %s", exc)


def main(argv: Sequence[str] | None = None) -> int:
    from nepal.config import Config
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = Config.load()
    work_root = cfg.work_root
    now = datetime.now(timezone.utc)
    stop, why = decide(jobs_running=len(jobs(_cmdlines())),
                       last_activity=last_activity(work_root), now=now,
                       idle_stop_min=float(cfg.get("cloud.gcp.idle_stop_min")))
    log.info("watchdog: %s (%s)", "stopping" if stop else "staying up", why)
    if not stop:
        return 0
    _note(work_root, f"--- idle-stopped at {now.isoformat(timespec='seconds')}: {why}")
    _push(cfg, work_root)
    subprocess.run(["shutdown", "-h", "now"], capture_output=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
