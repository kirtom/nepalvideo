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

from nepal.cloud import gce, spend, sync
from nepal.cloud.gcloud import Gcloud

log = logging.getLogger(__name__)

EXTRAS = "vision,music,asr,faces,semantic,dev"


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

    # -- lifecycle --------------------------------------------------------
    def up(self, *, wait: bool = True) -> gce.Status:
        led = spend.ledger(self.cfg)
        # A box that runs for an hour costs one hour; refuse if even that
        # would cross the line.
        led.guard(self.profile.usd_per_h,
                  ceiling_usd=float(self.cfg.get("cloud.spend_ceiling_usd")))
        st = self.status()
        if st.state == "ABSENT":
            meta = {"nepal-profile": self.profile_key, "nepal-branch": self.branch,
                    "nepal-bucket": self.bucket}
            log.info("remote: creating %s (%s, %s)", self.profile.name,
                     self.profile.machine_type, "spot" if self.profile.spot else "on-demand")
            self.gcloud.run(gce.create_args(self.profile, project=self.project, zone=self.zone,
                                            startup_script=self.startup_script, metadata=meta))
        elif st.state in ("TERMINATED", "STOPPING"):
            log.info("remote: starting %s", self.profile.name)
            self.gcloud.run(gce.start_args(self.profile.name, project=self.project,
                                           zone=self.zone))
        state = self._state()
        state.setdefault(self.profile_key, {})["up_since"] = \
            datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._save(state)
        if wait:
            self._wait_ready()
        return self.status()

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
        state = self._state()
        since = state.get(self.profile_key, {}).pop("up_since", None)
        if since:
            hours = (datetime.now(timezone.utc)
                     - datetime.fromisoformat(since)).total_seconds() / 3600
            spend.ledger(self.cfg).record(
                f"gce:{self.profile_key}", hours * self.profile.usd_per_h,
                detail=f"{hours:.2f} h {self.profile.machine_type} at "
                       f"{self.profile.usd_per_h} USD/h (estimate)")
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
        # raw/ is refreshed too: material lands in the bucket after the box
        # was first booted (the upload outlives the bootstrap), and an rsync
        # of an unchanged tree costs a listing.
        pull = (f"gcloud storage rsync --recursive {self.bucket}/raw {self.remote_data} && "
                f"gcloud storage rsync --recursive {self.bucket}/work {self.remote_work}")
        push = f"gcloud storage rsync --recursive {self.remote_work} {self.bucket}/work"
        # A --command runs in a non-login shell, so the profile (and the API
        # key the bootstrap put in it) is sourced by hand.
        return (f". ~/.profile 2>/dev/null; cd {self.remote_repo} && "
                f"git fetch -q origin {self.branch} && "
                f"git reset -q --hard origin/{self.branch} && "
                f".venv/bin/pip install -q -e '.[{EXTRAS}]' && {pull} && "
                f"({command}); rc=$?; "
                # the status page is rebuilt after every run, whatever happened
                f".venv/bin/nepal status-page >/dev/null 2>&1; {push}; exit $rc")

    def exec_cmd(self, command: str) -> int:
        return self.gcloud.stream(gce.ssh_args(self.profile.name, project=self.project, user=self.ssh_user,
                                               zone=self.zone, command=self._wrap(command)))

    def run_nepal(self, args: Sequence[str]) -> int:
        return self.exec_cmd(".venv/bin/nepal " + " ".join(shlex.quote(a) for a in args))

    def ssh(self) -> int:
        return self.gcloud.stream(gce.ssh_args(self.profile.name, project=self.project, user=self.ssh_user,
                                               zone=self.zone))
