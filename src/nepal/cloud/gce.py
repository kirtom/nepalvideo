"""gcloud compute, as argument lists.

Nothing here runs anything. Each function returns the arguments that follow
`gcloud`, so the command a stage is about to issue can be asserted in a test
and printed in a log before a single API call is made -- the same rule as
the ffmpeg graphs in S03 and S07.

Two decisions are baked in. Spot VMs stop rather than terminate on
preemption (`--instance-termination-action=STOP`), so the disk and everything
pulled onto it survive and `nepal remote up` simply starts it again. And the
box gets the `storage-rw` scope plus the two the Ops Agent needs to ship logs
and metrics, and nothing else: a box that can only read and write one bucket
and its own telemetry is a box whose compromise costs one bucket. The first
box had storage only, and the agent was refused for insufficient scopes while
every chart stayed empty.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class Profile:
    name: str
    machine_type: str
    disk_gb: int
    image_family: str
    image_project: str
    accelerator: str | None
    spot: bool
    usd_per_h: float

    @classmethod
    def from_cfg(cls, d: Mapping[str, Any]) -> "Profile":
        return cls(str(d["name"]), str(d["machine_type"]), int(d["disk_gb"]),
                   str(d["image_family"]), str(d["image_project"]),
                   (str(d["accelerator"]) if d.get("accelerator") else None),
                   bool(d.get("spot", True)), float(d.get("usd_per_h", 0.0)))


@dataclass(frozen=True)
class Status:
    name: str
    state: str          # RUNNING | TERMINATED | STOPPING | STAGING | PROVISIONING | ABSENT
    ip: str | None
    # What GCP bills from: when this instance last started and last stopped.
    # The local stamp `remote up` writes says only that a session meant to
    # start a box; these say what the meter did, including the starts and
    # stops no session of ours saw.
    last_start: datetime | None = None
    last_stop: datetime | None = None


def _base(name: str, verb: str, *, project: str, zone: str) -> list[str]:
    return ["compute", "instances", verb, name, f"--project={project}", f"--zone={zone}"]


def create_args(p: Profile, *, project: str, zone: str, startup_script: Path,
                metadata: Mapping[str, str]) -> list[str]:
    args = _base(p.name, "create", project=project, zone=zone) + [
        f"--machine-type={p.machine_type}",
        f"--image-family={p.image_family}", f"--image-project={p.image_project}",
        f"--boot-disk-size={p.disk_gb}GB", "--boot-disk-type=pd-balanced",
        "--scopes=storage-rw,logging-write,monitoring-write",
        f"--metadata-from-file=startup-script={startup_script}",
        "--labels=project=nepal",
    ]
    if metadata:
        args.append("--metadata=" + ",".join(f"{k}={v}" for k, v in metadata.items()))
    if p.spot:
        args += ["--provisioning-model=SPOT", "--instance-termination-action=STOP"]
    if p.accelerator:
        args += [f"--accelerator=type={p.accelerator},count=1",
                 "--maintenance-policy=TERMINATE"]
    return args


def describe_args(name: str, *, project: str, zone: str) -> list[str]:
    return _base(name, "describe", project=project, zone=zone) + ["--format=json"]


def startup_script_args(name: str, *, project: str, zone: str,
                        startup_script: Path) -> list[str]:
    # A box runs the startup script it was created with; nothing on it
    # pulls a newer one. The first fix to the bootstrap never reached the
    # box until this call existed.
    return _base(name, "add-metadata", project=project, zone=zone) + [
        f"--metadata-from-file=startup-script={startup_script}"]


def start_args(name: str, *, project: str, zone: str) -> list[str]:
    return _base(name, "start", project=project, zone=zone)


def stop_args(name: str, *, project: str, zone: str) -> list[str]:
    return _base(name, "stop", project=project, zone=zone)


def delete_args(name: str, *, project: str, zone: str) -> list[str]:
    return _base(name, "delete", project=project, zone=zone) + ["--quiet"]


def ssh_args(name: str, *, project: str, zone: str,
             command: str | None = None, user: str | None = None) -> list[str]:
    # Log in as the user that owns the checkout and the data on the box.
    # Without this gcloud uses the local login name, and git refuses to
    # touch a repository owned by somebody else ("dubious ownership").
    target = f"{user}@{name}" if user else name
    args = ["compute", "ssh", target, f"--project={project}", f"--zone={zone}",
            "--quiet"]
    if command is not None:
        args.append(f"--command={command}")
    return args


def stamp(value: Any) -> datetime | None:
    """A GCE RFC3339 stamp as UTC, or None.

    GCP writes them in the zone's own offset ("2026-09-20T09:12:33.123-07:00")
    and the box runs Python 3.10, whose `fromisoformat` takes an offset but
    not a trailing Z. Anything unparseable is None rather than an exception:
    a missing hour in a display must not stop a box from being described.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")) \
            .astimezone(timezone.utc)
    except ValueError:
        return None


def parse_describe(doc: Mapping[str, Any]) -> Status:
    if not doc:
        return Status("", "ABSENT", None)
    ip = None
    for nic in doc.get("networkInterfaces") or []:
        for ac in nic.get("accessConfigs") or []:
            if ac.get("natIP"):
                ip = str(ac["natIP"])
    return Status(str(doc.get("name", "")), str(doc.get("status", "ABSENT")), ip,
                  stamp(doc.get("lastStartTimestamp")), stamp(doc.get("lastStopTimestamp")))
