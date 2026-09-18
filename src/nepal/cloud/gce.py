"""gcloud compute, as argument lists.

Nothing here runs anything. Each function returns the arguments that follow
`gcloud`, so the command a stage is about to issue can be asserted in a test
and printed in a log before a single API call is made -- the same rule as
the ffmpeg graphs in S03 and S07.

Two decisions are baked in. Spot VMs stop rather than terminate on
preemption (`--instance-termination-action=STOP`), so the disk and everything
pulled onto it survive and `nepal remote up` simply starts it again. And the
box gets the `storage-rw` scope and nothing else: it needs the bucket, and a
box that can only read and write one bucket is a box whose compromise costs
one bucket.
"""
from __future__ import annotations

from dataclasses import dataclass
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


def _base(name: str, verb: str, *, project: str, zone: str) -> list[str]:
    return ["compute", "instances", verb, name, f"--project={project}", f"--zone={zone}"]


def create_args(p: Profile, *, project: str, zone: str, startup_script: Path,
                metadata: Mapping[str, str]) -> list[str]:
    args = _base(p.name, "create", project=project, zone=zone) + [
        f"--machine-type={p.machine_type}",
        f"--image-family={p.image_family}", f"--image-project={p.image_project}",
        f"--boot-disk-size={p.disk_gb}GB", "--boot-disk-type=pd-balanced",
        "--scopes=storage-rw",
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


def start_args(name: str, *, project: str, zone: str) -> list[str]:
    return _base(name, "start", project=project, zone=zone)


def stop_args(name: str, *, project: str, zone: str) -> list[str]:
    return _base(name, "stop", project=project, zone=zone)


def delete_args(name: str, *, project: str, zone: str) -> list[str]:
    return _base(name, "delete", project=project, zone=zone) + ["--quiet"]


def ssh_args(name: str, *, project: str, zone: str,
             command: str | None = None) -> list[str]:
    args = ["compute", "ssh", name, f"--project={project}", f"--zone={zone}",
            "--quiet"]
    if command is not None:
        args.append(f"--command={command}")
    return args


def parse_describe(doc: Mapping[str, Any]) -> Status:
    if not doc:
        return Status("", "ABSENT", None)
    ip = None
    for nic in doc.get("networkInterfaces") or []:
        for ac in nic.get("accessConfigs") or []:
            if ac.get("natIP"):
                ip = str(ac["natIP"])
    return Status(str(doc.get("name", "")), str(doc.get("status", "ABSENT")), ip)
