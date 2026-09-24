"""`nepal remote`: one Spot box, the bucket as the hub, a ledger that refuses.

`up` creates the box if it is absent and starts it if it is stopped, then
waits for the startup script's READY marker. `run` wraps a `nepal` command
so that the box first tracks the branch, pulls `work/` from the bucket, runs,
and pushes `work/` back whatever the exit code -- a killed run still leaves
its checkpoints in the bucket. `down` stops (keeping the disk) and writes the
hours to the ledger at the profile's price.
"""
from __future__ import annotations

import json
import logging
import shlex
import time
from datetime import datetime, timezone
from typing import Sequence

from nepal.cloud import gce, spend, sync, watchdog
from nepal.cloud.gcloud import Gcloud

log = logging.getLogger(__name__)

EXTRAS = "vision,music,asr,faces,semantic,api,dev"

# What the box is asked before a command touches work/. It asks the
# watchdog's own module, so "what a job looks like" has one definition --
# and jobs() drops any command line naming that module, which is what keeps
# the `bash -c` carrying this whole wrapper (it names the venv a dozen
# times) from reading as the job it is looking for. Empty output means the
# box is idle; the first line names the job that is not.
JOB_PROBE = "job=$(.venv/bin/python -m nepal.cloud.watchdog jobs 2>/dev/null | head -1)"
# Held back from the push while a job runs: the draft is the one output
# written over minutes, so a push mid-render would publish a truncated
# file over the good one in the bucket. Everything else under work/ is a
# checkpoint, and pushing those mid-run is the point.
JOB_PUSH_EXCLUDE = "gates/.*"


class Remote:
    def __init__(self, cfg, profile: str = "cpu", gcloud: Gcloud | None = None):
        self.cfg = cfg
        g = cfg.get("cloud.gcp")
        self.profile_key = profile
        self.profile = gce.Profile.from_cfg(g["profiles"][profile])
        self.project = str(g["project"])
        self.zone = str(g["zone"])
        self.bucket = str(g["bucket"]).rstrip("/")
        self.branch = str(g["branch"])
        self.remote_repo = str(g["remote_repo"])
        self.remote_work = str(g["remote_work_root"])
        self.remote_data = str(g["remote_data_root"])
        self.ssh_user = str(g.get("ssh_user", "nepal"))
        self.ready_timeout = float(g.get("ready_timeout_s", 1500))
        self.gcloud = gcloud or Gcloud(g.get("gcloud", "gcloud"))
        self.state_path = cfg.work_root / "reports" / "remote_state.json"
        self.startup_script = cfg.base_dir / "tools" / "cloud" / "bootstrap-gcp.sh"

    # -- state ------------------------------------------------------------
    def _state(self) -> dict:
        return json.loads(self.state_path.read_text()) if self.state_path.exists() else {}

    def _save(self, st: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(st, indent=2))

    def status(self) -> gce.Status:
        doc = self.gcloud.json(gce.describe_args(self.profile.name, project=self.project,
                                                 zone=self.zone))
        return gce.parse_describe(doc or {})

    # -- money ------------------------------------------------------------
    def account(self, st: gce.Status | None = None) -> dict:
        """What the box has cost: what the ledger holds, plus the meter.

        The ledger books VM hours at `down`. A session that ends without one
        -- a rate limit, a crash, a closed laptop -- leaves the box running
        and the total unchanged: that is how nepal-cpu billed 104 idle hours
        while `remote status` read 5.21 USD for four days. So a running box's
        time is counted here from GCP's own `lastStartTimestamp`, which is
        what the bill is made of; the stamp `up` wrote locally is the
        fallback for a session that cannot describe the instance. And a box
        that has stopped since the last command -- the idle watchdog, a
        preemption -- is booked now, from `lastStopTimestamp`, or its hours
        are lost to the ledger entirely.
        """
        st = st or self.status()
        state = self._state()
        prof = state.setdefault(self.profile_key, {})
        local_since = gce.stamp(prof.get("up_since"))
        start = st.last_start or local_since
        led = spend.ledger(self.cfg)
        hours = 0.0
        if start and st.state in ("RUNNING", "STAGING", "PROVISIONING"):
            hours = max(0.0, (datetime.now(timezone.utc) - start).total_seconds() / 3600)
        elif start and local_since and st.last_stop and st.last_stop > start:
            if not self._booked(led, start):
                self._book(led, (st.last_stop - start).total_seconds() / 3600, start,
                           note=f"stopped at {st.last_stop.isoformat(timespec='seconds')} "
                                f"without `down`")
            prof.pop("up_since", None)
            self._save(state)
        running = hours * self.profile.usd_per_h
        return {"status": st, "ledger": led, "booked": led.total(), "unbooked_h": hours,
                "unbooked_usd": running, "since": start, "total": led.total() + running}

    @staticmethod
    def _session(start: datetime) -> str:
        return f"session from {start.isoformat(timespec='seconds')}"

    def _booked(self, led: spend.Ledger, start: datetime) -> bool:
        """Has this instance start been booked already?

        remote_state.json lives under work/reports, which rsyncs in both
        directions, so a `pull` copies back the `up_since` that booking
        popped and the same span would be booked again after every pull.
        The ledger is append-only and travels the same way by union, so it
        is the only record that cannot be resurrected: one entry per
        instance start, and the start is named in the entry.
        """
        return any(self._session(start) in e.detail for e in led.entries)

    def _book(self, led: spend.Ledger, hours: float, start: datetime,
              *, note: str = "") -> None:
        led.record(f"gce:{self.profile_key}", hours * self.profile.usd_per_h,
                   detail=f"{hours:.2f} h {self.profile.machine_type} at "
                          f"{self.profile.usd_per_h} USD/h (estimate), "
                          f"{self._session(start)}" + (f", {note}" if note else ""))
        # The box guards the API spend against this total, and it reads
        # the ledger from the bucket at its next run: push the entry now,
        # not at the next `remote push` somebody remembers to make.
        self.gcloud.run(sync.rsync_args(str(led.path), f"{self.bucket}/work/reports/spend"),
                        check=False)

    def _guard(self, estimate_usd: float) -> dict:
        """Refuse against the live total, not the booked one: a box that has
        already eaten the ceiling must be refused now, not at the `down`
        that may never come."""
        acct = self.account()
        acct["ledger"].guard(estimate_usd, unbooked_usd=acct["unbooked_usd"],
                             ceiling_usd=float(self.cfg.get("cloud.spend_ceiling_usd")))
        return acct

    # -- lifecycle --------------------------------------------------------
    def up(self, *, wait: bool = True) -> gce.Status:
        # A box that runs for an hour costs one hour; refuse if even that
        # would cross the line.
        st = self._guard(self.profile.usd_per_h)["status"]
        if st.state == "ABSENT":
            meta = {"nepal-profile": self.profile_key, "nepal-branch": self.branch,
                    "nepal-bucket": self.bucket}
            log.info("remote: creating %s (%s, %s)", self.profile.name,
                     self.profile.machine_type, "spot" if self.profile.spot else "on-demand")
            self.gcloud.run(gce.create_args(self.profile, project=self.project, zone=self.zone,
                                            startup_script=self.startup_script, metadata=meta))
        elif st.state in ("TERMINATED", "STOPPING"):
            # The checkout's bootstrap, every start: the box boots the
            # startup script in its metadata, not the one in the branch.
            self.gcloud.run(gce.startup_script_args(self.profile.name, project=self.project,
                                                    zone=self.zone,
                                                    startup_script=self.startup_script))
            log.info("remote: starting %s", self.profile.name)
            self.gcloud.run(gce.start_args(self.profile.name, project=self.project,
                                           zone=self.zone))
        state = self._state()
        # The price alongside the stamp: the status page prices the unbooked
        # hours from this file, and it runs on the box, which has no config
        # of its own to look the profile up in.
        state.setdefault(self.profile_key, {}).update(
            up_since=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            usd_per_h=self.profile.usd_per_h)
        self._save(state)
        if wait:
            self._wait_ready()
        st = self.status()
        if st.state == "RUNNING":
            self.install_watchdog()
        return st

    def install_watchdog(self) -> None:
        """Put the idle watchdog on the box, or refresh it.

        The startup script installs it at every boot, which covers a box that
        `up` creates or starts. This covers the one that is already running --
        every box created before the watchdog existed, including the nepal-cpu
        that billed the 104 idle hours.
        """
        script = watchdog.install_sh(
            repo=self.remote_repo,
            period_min=int(self.cfg.get("cloud.gcp.idle_check_min")))
        # A heredoc rather than a quoted argument: the script is multi-line
        # and ssh hands the whole --command string to the login shell anyway.
        cmd = f"sudo bash -s <<'NEPAL_WATCHDOG_EOF'\n{script}\nNEPAL_WATCHDOG_EOF\n"
        proc = self.gcloud.run(gce.ssh_args(self.profile.name, project=self.project,
                                            user=self.ssh_user, zone=self.zone, command=cmd),
                               check=False)
        log.info("remote: idle watchdog %s",
                 "installed" if proc.returncode == 0 else "install failed (see the boot log)")

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.ready_timeout
        probe = "test -f /data/projects/READY"
        while time.monotonic() < deadline:
            proc = self.gcloud.run(gce.ssh_args(self.profile.name, project=self.project, user=self.ssh_user,
                                                zone=self.zone, command=probe), check=False)
            if proc.returncode == 0:
                log.info("remote: %s is ready", self.profile.name)
                return
            time.sleep(15)
        raise TimeoutError(f"{self.profile.name} did not become ready in "
                           f"{self.ready_timeout:.0f}s; read /var/log/nepal-bootstrap.log")

    def down(self, *, delete: bool = False) -> None:
        # account() books a box that stopped without us; what is left to book
        # here is a box still running, from the start GCP reports.
        acct = self.account()
        state = self._state()
        state.setdefault(self.profile_key, {}).pop("up_since", None)
        # Not conditional on the local stamp: account() counted these hours
        # from GCP's own start, and a state file that lost `up_since` (or
        # never had it, on a box started from another machine) would
        # otherwise stop this box for free. Booking twice is what the
        # ledger check prevents.
        if acct["unbooked_h"] and not self._booked(acct["ledger"], acct["since"]):
            self._book(acct["ledger"], acct["unbooked_h"], acct["since"])
        self._save(state)
        args = (gce.delete_args if delete else gce.stop_args)(
            self.profile.name, project=self.project, zone=self.zone)
        self.gcloud.run(args, check=False)
        log.info("remote: %s %s", self.profile.name, "deleted" if delete else "stopped")

    # -- data -------------------------------------------------------------
    def push(self) -> None:
        for src, dst in sync.push_plan(self.cfg):
            if src.exists():
                self.gcloud.run(sync.rsync_args(str(src), dst))

    def pull(self) -> None:
        for src, dst in sync.pull_plan(self.cfg):
            dst.mkdir(parents=True, exist_ok=True)
            self.gcloud.run(sync.rsync_args(src, str(dst)), check=False)

    # -- commands ---------------------------------------------------------
    def _wrap(self, command: str) -> str:
        # raw/ is refreshed unconditionally: material lands in the bucket
        # after the box was first booted (the upload outlives the
        # bootstrap), it is read-only media, and an rsync of an unchanged
        # tree costs a listing.
        #
        # work/ is not. rsync writes a temp and renames it over the target,
        # so pulling the bucket's older copy of a file a running job holds
        # open leaves that job writing into `draft.mp4 (deleted)`: one
        # `exec` from another session did exactly that to a 40-minute
        # ffmpeg, which was rescued through /proc/<pid>/fd. A job on the box
        # owns work/ until it finishes; a concurrent command reads what is
        # there and is told why it got no refresh.
        pull_work = f"gcloud storage rsync --recursive {self.bucket}/work {self.remote_work}"
        pull = (f"gcloud storage rsync --recursive {self.bucket}/raw {self.remote_data} && "
                f"{{ {JOB_PROBE}; if [ -n \"$job\" ]; then "
                f"echo \"skipping work/ pull: job running ($job)\"; else {pull_work}; fi; }}")
        # The same hazard at the other end, in the other direction: half a
        # file pushed over a good one in the bucket. The probe runs again
        # rather than reusing the pull's answer -- this command may have
        # started the job itself, detached.
        push_all = f"gcloud storage rsync --recursive {self.remote_work} {self.bucket}/work"
        push_safe = (f"gcloud storage rsync --recursive --exclude='{JOB_PUSH_EXCLUDE}' "
                     f"{self.remote_work} {self.bucket}/work")
        push = (f"{{ {JOB_PROBE}; if [ -n \"$job\" ]; then "
                f"echo \"pushing work/ without gates/: job running ($job)\"; {push_safe}; "
                f"else {push_all}; fi; }}")
        # Before and after: the idle watchdog reads this stamp, and a job
        # launched detached (`nohup ... &`) returns at once, so the stamp at
        # the end is when the box was last *asked* for something, not when it
        # stopped working -- which is why the watchdog reads the job logs too.
        touch = watchdog.touch_sh(self.remote_work)
        # A --command runs in a non-login shell, so the profile (and the API
        # key the bootstrap put in it) is sourced by hand.
        return (f". ~/.profile 2>/dev/null; {touch}; cd {self.remote_repo} && "
                f"git fetch -q origin {self.branch} && "
                f"git reset -q --hard origin/{self.branch} && "
                f".venv/bin/pip install -q -e '.[{EXTRAS}]' && {pull} && "
                f"({command}); rc=$?; {touch}; "
                # the status page is rebuilt after every run, whatever happened
                f".venv/bin/nepal status-page >/dev/null 2>&1; {push}; exit $rc")

    def exec_cmd(self, command: str) -> int:
        # Same estimate as `up`: a command costs at least the hour of box
        # time it runs in, and the refusal has to arrive before the command,
        # not at a `down` that may never happen.
        self._guard(self.profile.usd_per_h)
        return self.gcloud.stream(gce.ssh_args(self.profile.name, project=self.project, user=self.ssh_user,
                                               zone=self.zone, command=self._wrap(command)))

    def run_nepal(self, args: Sequence[str]) -> int:
        return self.exec_cmd(".venv/bin/nepal " + " ".join(shlex.quote(a) for a in args))

    def ssh(self) -> int:
        return self.gcloud.stream(gce.ssh_args(self.profile.name, project=self.project, user=self.ssh_user,
                                               zone=self.zone))
