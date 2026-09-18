# Film v2 — Step 2: remote execution on GCP — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `nepal remote` runs any stage, test or command on a GCP Spot VM with the bucket as the hub, keeps a spend ledger under a ceiling, and lets the manifest run on a host that holds only part of the corpus — so that nothing computes on the operator's machine again.

**Architecture:** Pure command builders (`nepal/cloud/gce.py`, `nepal/cloud/sync.py`) produce `gcloud` argument lists; one thin runner (`nepal/cloud/gcloud.py`) executes them; `nepal/cloud/remote.py` orchestrates up/run/pull/down and the ledger; a startup script on the VM installs the toolchain idempotently and pulls the bucket. The manifest learns to keep assets whose source directory is absent on the current host and to skip hashing files it has already seen.

**Tech Stack:** Python 3.10+, `gcloud` CLI (installed at `~/google-cloud-sdk/bin/gcloud`, logged in), GCS via `gcloud storage rsync`, GCE Spot VMs, bash startup script, pytest with a fake `gcloud` on PATH.

**Spec:** `docs/superpowers/specs/2026-09-16-film-v2-voice-spine-design.md` §2.2 (execution model), §8 (hygiene), §11 step 2.

## Global Constraints

- **Nothing computes on the local machine.** The unit tests in this plan are pure and run in well under a second per file; run them one file at a time with `-q`, never the whole suite. The full and slow suites run on the box (Task 6).
- Every tunable lives in `config/pipeline.yaml` inside the existing blocks or one new `cloud:` block; **duplicate top-level keys raise**.
- Everything runs through the `nepal` console script. The gcloud binary path is `cloud.gcp.gcloud` (default `~/google-cloud-sdk/bin/gcloud`), never hard-coded.
- Modules that shell out keep the shelling in a thin wrapper; decision logic stays pure.
- Credentials never enter the repo: `gcloud` holds the operator's login; `ANTHROPIC_API_KEY` lives only in the box's environment (Task 3 reads it from a metadata attribute the operator sets, never from a file in git).
- The worktree guard on this machine refuses shell commands containing certain words (`enable`, `git` inside scripts). Put such commands in a script file under `$CLAUDE_JOB_DIR/tmp` and run the file.
- Commit after every task, message ending with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_013XaQhkq3CDL5Qy4AUKcX5M`.

**Facts fixed on 2026-09-18:** project ID `nepalvideo`, number `29922345852`, region `europe-west4`, zone `europe-west4-a`, bucket `gs://nepalvideo-29922345852`, budget "nepal" 60 USD. Quota: `NVIDIA_L4_GPUS=1`, `PREEMPTIBLE_NVIDIA_L4_GPUS=1` granted; `GPUS_ALL_REGIONS=0` (operator requesting). The upload of `raw/`, `work/`, `ref/` was started by `$CLAUDE_JOB_DIR/tmp/gcp_upload.sh`.

## File structure

| File | Responsibility |
|---|---|
| `config/pipeline.yaml` | new `cloud:` block (Task 1) |
| `src/nepal/cloud/__init__.py` (new) | package marker |
| `src/nepal/cloud/spend.py` (new) | the ledger: `Entry`, `Ledger`, `load`, `record`, `total`, `guard`, `SpendCeiling` |
| `src/nepal/cloud/gce.py` (new) | pure `gcloud compute` argument builders and the `Profile` dataclass |
| `src/nepal/cloud/sync.py` (new) | pure `gcloud storage rsync` builders and the push/pull manifests |
| `src/nepal/cloud/gcloud.py` (new) | the one thin runner: `run(args) -> CompletedProcess`, `json(args)`, `stream(args)` |
| `src/nepal/cloud/remote.py` (new) | orchestration: `up`, `down`, `status`, `push`, `pull`, `run_nepal`, `exec_cmd`, `ssh` |
| `src/nepal/cli.py` | `nepal remote …` sub-commands |
| `tools/cloud/bootstrap-gcp.sh` (new) | GCE startup script, idempotent, profile-aware |
| `src/nepal/probe/manifest.py`, `src/nepal/stages/s01_probe.py`, `src/nepal/db.py` | partial-host manifest and incremental hashing (Task 5) |
| `tests/test_cloud_spend.py`, `tests/test_cloud_gce.py`, `tests/test_cloud_sync.py`, `tests/test_cloud_remote.py`, `tests/test_bootstrap_gcp.py`, `tests/test_manifest_partial.py` (all new) | one file per unit |
| `README.md`, `docs/STATE.md`, `CLAUDE.md` | the cloud section rewritten for GCP; the new trap |

---

### Task 1: The `cloud:` config block and the spend ledger

**Files:**
- Modify: `config/pipeline.yaml` (append a top-level `cloud:` block; check `grep -n '^cloud:' config/pipeline.yaml` finds nothing first)
- Create: `src/nepal/cloud/__init__.py` (empty), `src/nepal/cloud/spend.py`
- Test: `tests/test_cloud_spend.py`

**Interfaces:**
- Produces:
  - config keys listed in Step 3 below.
  - `spend.Entry(when: str, what: str, usd: float, detail: str = "")` (frozen dataclass).
  - `spend.Ledger(path: Path)` with `.entries -> list[Entry]`, `.total() -> float`, `.record(what, usd, detail="") -> Entry`, `.guard(estimate_usd, ceiling_usd)` raising `SpendCeiling(total, estimate, ceiling)`.
  - `spend.ledger(cfg) -> Ledger` at `cfg.work_root / "reports" / "spend.json"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cloud_spend.py
"""The spend ledger: what the project has paid for, and the ceiling."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import json
import pytest

from nepal.cloud import spend


def test_a_fresh_ledger_is_empty(tmp_path):
    led = spend.Ledger(tmp_path / "spend.json")
    assert led.entries == [] and led.total() == 0.0


def test_record_persists_and_totals(tmp_path):
    led = spend.Ledger(tmp_path / "spend.json")
    led.record("gce:cpu", 0.42, detail="3.8 h e2-standard-8 spot")
    led.record("api:captions", 1.31)
    again = spend.Ledger(tmp_path / "spend.json")
    assert [e.what for e in again.entries] == ["gce:cpu", "api:captions"]
    assert again.total() == pytest.approx(1.73)
    raw = json.loads((tmp_path / "spend.json").read_text())
    assert raw["entries"][0]["usd"] == 0.42 and raw["entries"][0]["when"]


def test_guard_refuses_when_the_estimate_would_cross_the_ceiling(tmp_path):
    led = spend.Ledger(tmp_path / "spend.json")
    led.record("api:beats", 13.0)
    led.guard(1.5, ceiling_usd=15.0)                  # 14.5: fine
    with pytest.raises(spend.SpendCeiling) as exc:
        led.guard(2.5, ceiling_usd=15.0)              # 15.5: not fine
    assert exc.value.total == 13.0 and exc.value.ceiling == 15.0


def test_a_corrupt_ledger_is_an_error_not_a_zero(tmp_path):
    (tmp_path / "spend.json").write_text("{not json")
    with pytest.raises(ValueError):
        spend.Ledger(tmp_path / "spend.json")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /data/projects/nepalvideo/.claude/worktrees/film-v2-design && /data/projects/nepalvideo/.venv/bin/python -m pytest tests/test_cloud_spend.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'nepal.cloud'`.

- [ ] **Step 3: Config and module**

Append to `config/pipeline.yaml`:

```yaml
# Remote execution (Film v2 section 2.2). The local machine computes nothing;
# everything runs on one GCP Spot VM with the bucket as the hub.
cloud:
  provider: gcp
  # Hard ceiling on what the pipeline may spend, in USD, across API calls and
  # VM hours as the ledger (work/reports/spend.json) records them. The
  # operator approved 15 on 2026-09-16; the GCP billing budget is 60 and
  # alerts at 50% and 80%, but that is a warning, this is a refusal.
  spend_ceiling_usd: 15
  gcp:
    gcloud: ~/google-cloud-sdk/bin/gcloud
    project: nepalvideo
    region: europe-west4
    zone: europe-west4-a
    bucket: gs://nepalvideo-29922345852
    # The branch the box checks out. The box always resets to origin/<branch>,
    # so push before `nepal remote run`.
    branch: worktree-film-v2-design
    repo: https://github.com/kirtom/nepalvideo.git
    # Two profiles: everything that is CPU, and the GPU minutes. Spot prices
    # in europe-west4 as of 2026-09-18, for the ledger's estimate only -- the
    # bill is what GCP says it is.
    profiles:
      cpu:
        name: nepal-cpu
        machine_type: e2-standard-8
        disk_gb: 60
        image_family: ubuntu-2204-lts
        image_project: ubuntu-os-cloud
        accelerator: null
        spot: true
        usd_per_h: 0.11
      gpu:
        name: nepal-gpu
        machine_type: g2-standard-4
        disk_gb: 100
        image_family: ubuntu-2204-lts
        image_project: ubuntu-os-cloud
        accelerator: nvidia-l4
        spot: true
        usd_per_h: 0.30
    # Where the box keeps things: the same paths the config names, so
    # pipeline.yaml needs no change on the box.
    remote_data_root: /data/projects/nepal_data
    remote_work_root: /data/projects/nepal_work
    remote_repo: /data/projects/nepalvideo
    # How long `up` waits for the startup script to write its READY marker.
    ready_timeout_s: 1500
```

```python
# src/nepal/cloud/spend.py
"""What the project has paid for, and the line it must not cross.

One JSON file, appended by whatever spends: `remote down` with the VM hours,
the API wrappers with their `usage`. The ceiling is a refusal, not a
warning -- the GCP budget already warns. Kept as a file rather than a
`decisions` row so it survives a database that is rebuilt or pulled from
the bucket, and so a person can read it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Entry:
    when: str
    what: str
    usd: float
    detail: str = ""


class SpendCeiling(RuntimeError):
    def __init__(self, total: float, estimate: float, ceiling: float):
        self.total, self.estimate, self.ceiling = total, estimate, ceiling
        super().__init__(
            f"spend ceiling: {total:.2f} USD spent, this step is estimated at "
            f"{estimate:.2f}, and the ceiling is {ceiling:.2f}. Raise "
            f"cloud.spend_ceiling_usd deliberately or skip the step.")


class Ledger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.entries: list[Entry] = []
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text())
            except json.JSONDecodeError as exc:
                raise ValueError(f"{self.path} is not valid JSON: {exc}") from exc
            self.entries = [Entry(**e) for e in raw.get("entries", [])]

    def total(self) -> float:
        return float(sum(e.usd for e in self.entries))

    def record(self, what: str, usd: float, detail: str = "") -> Entry:
        e = Entry(datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  what, round(float(usd), 4), detail)
        self.entries.append(e)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"entries": [asdict(x) for x in self.entries],
                                   "total_usd": round(self.total(), 4)}, indent=2))
        tmp.replace(self.path)
        return e

    def guard(self, estimate_usd: float, *, ceiling_usd: float) -> None:
        if self.total() + float(estimate_usd) > float(ceiling_usd):
            raise SpendCeiling(self.total(), float(estimate_usd), float(ceiling_usd))


def ledger(cfg) -> Ledger:
    return Ledger(cfg.work_root / "reports" / "spend.json")
```

Create `src/nepal/cloud/__init__.py` empty.

- [ ] **Step 4: Run the test**

Run: `/data/projects/nepalvideo/.venv/bin/python -m pytest tests/test_cloud_spend.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add config/pipeline.yaml src/nepal/cloud/__init__.py src/nepal/cloud/spend.py tests/test_cloud_spend.py
git commit -m "A spend ledger with a ceiling that refuses, and the cloud config block

The GCP budget warns at 50 and 80 percent; the pipeline itself must
refuse. One JSON file records every VM hour and API call, and every paid
step asks it before starting.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013XaQhkq3CDL5Qy4AUKcX5M"
```

---

### Task 2: Pure `gcloud compute` builders

**Files:**
- Create: `src/nepal/cloud/gce.py`
- Test: `tests/test_cloud_gce.py`

**Interfaces:**
- Produces:
  - `gce.Profile(name, machine_type, disk_gb, image_family, image_project, accelerator, spot, usd_per_h)` with `Profile.from_cfg(d: dict) -> Profile`.
  - `gce.create_args(p: Profile, *, project, zone, startup_script: Path, metadata: dict[str, str]) -> list[str]`
  - `gce.describe_args(name, *, project, zone) -> list[str]`, `gce.start_args`, `gce.stop_args`, `gce.delete_args` (same signature).
  - `gce.ssh_args(name, *, project, zone, command: str | None = None) -> list[str]`
  - `gce.parse_describe(doc: dict) -> Status` where `Status(name, state, ip: str | None)`; `state` is one of `RUNNING`, `TERMINATED`, `STOPPING`, `STAGING`, `PROVISIONING`, `ABSENT`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cloud_gce.py
"""gcloud compute argument lists, built as data."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from nepal.cloud import gce

CPU = gce.Profile("nepal-cpu", "e2-standard-8", 60, "ubuntu-2204-lts",
                  "ubuntu-os-cloud", None, True, 0.11)
GPU = gce.Profile("nepal-gpu", "g2-standard-4", 100, "ubuntu-2204-lts",
                  "ubuntu-os-cloud", "nvidia-l4", True, 0.30)


def test_profile_from_config_dict():
    p = gce.Profile.from_cfg({"name": "x", "machine_type": "e2-small", "disk_gb": 20,
                              "image_family": "f", "image_project": "p",
                              "accelerator": None, "spot": False, "usd_per_h": 0.02})
    assert p.name == "x" and p.spot is False and p.accelerator is None


def test_create_args_for_a_spot_cpu_box(tmp_path):
    script = tmp_path / "boot.sh"
    args = gce.create_args(CPU, project="nepalvideo", zone="europe-west4-a",
                           startup_script=script, metadata={"nepal-profile": "cpu"})
    s = " ".join(args)
    assert args[:4] == ["compute", "instances", "create", "nepal-cpu"]
    assert "--project=nepalvideo" in args and "--zone=europe-west4-a" in args
    assert "--machine-type=e2-standard-8" in args
    assert "--provisioning-model=SPOT" in args
    assert "--instance-termination-action=STOP" in args      # keep the disk on preemption
    assert "--boot-disk-size=60GB" in args
    assert f"--metadata-from-file=startup-script={script}" in args
    assert "--metadata=nepal-profile=cpu" in args
    assert "--scopes=storage-rw" in s                        # the bucket, nothing more
    assert "--accelerator" not in s and "--maintenance-policy" not in s


def test_create_args_for_the_gpu_box(tmp_path):
    args = gce.create_args(GPU, project="p", zone="z", startup_script=tmp_path / "b.sh",
                           metadata={})
    assert "--accelerator=type=nvidia-l4,count=1" in args
    assert "--maintenance-policy=TERMINATE" in args          # GPUs cannot live-migrate
    assert "--provisioning-model=SPOT" in args


def test_on_demand_profile_has_no_spot_flags(tmp_path):
    p = gce.Profile("n", "e2-small", 20, "f", "pr", None, False, 0.02)
    args = gce.create_args(p, project="p", zone="z", startup_script=tmp_path / "b.sh",
                           metadata={})
    assert "--provisioning-model=SPOT" not in args
    assert "--instance-termination-action=STOP" not in args


def test_lifecycle_args():
    assert gce.describe_args("nepal-cpu", project="p", zone="z") == \
        ["compute", "instances", "describe", "nepal-cpu", "--project=p", "--zone=z",
         "--format=json"]
    assert gce.start_args("nepal-cpu", project="p", zone="z")[:4] == \
        ["compute", "instances", "start", "nepal-cpu"]
    assert gce.stop_args("nepal-cpu", project="p", zone="z")[2] == "stop"
    assert gce.delete_args("nepal-cpu", project="p", zone="z")[2:4] == ["delete", "nepal-cpu"]
    assert "--quiet" in gce.delete_args("nepal-cpu", project="p", zone="z")


def test_ssh_args_pass_a_command_through():
    args = gce.ssh_args("nepal-cpu", project="p", zone="z", command="nepal doctor")
    assert args[:3] == ["compute", "ssh", "nepal-cpu"]
    assert "--command=nepal doctor" in args
    assert "--" not in args
    plain = gce.ssh_args("nepal-cpu", project="p", zone="z")
    assert not any(a.startswith("--command") for a in plain)


def test_parse_describe_reads_state_and_ip():
    doc = {"name": "nepal-cpu", "status": "RUNNING",
           "networkInterfaces": [{"accessConfigs": [{"natIP": "34.1.2.3"}]}]}
    st = gce.parse_describe(doc)
    assert st.state == "RUNNING" and st.ip == "34.1.2.3" and st.name == "nepal-cpu"
    assert gce.parse_describe({"name": "n", "status": "TERMINATED"}).ip is None
    assert gce.parse_describe({}).state == "ABSENT"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `/data/projects/nepalvideo/.venv/bin/python -m pytest tests/test_cloud_gce.py -q`
Expected: FAIL with `ImportError: cannot import name 'gce'`.

- [ ] **Step 3: Write the module**

```python
# src/nepal/cloud/gce.py
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
```

- [ ] **Step 4: Run the test**

Run: `/data/projects/nepalvideo/.venv/bin/python -m pytest tests/test_cloud_gce.py -q`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nepal/cloud/gce.py tests/test_cloud_gce.py
git commit -m "gcloud compute as argument lists: create, lifecycle, ssh, status

Pure builders, so the command a stage is about to issue is asserted in a
test and printed before any API call. Spot boxes stop rather than
terminate on preemption, and the box gets the storage-rw scope and nothing
else.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013XaQhkq3CDL5Qy4AUKcX5M"
```

---

### Task 3: The sync builders, the thin runner, and the startup script

**Files:**
- Create: `src/nepal/cloud/sync.py`, `src/nepal/cloud/gcloud.py`, `tools/cloud/bootstrap-gcp.sh`
- Test: `tests/test_cloud_sync.py`, `tests/test_bootstrap_gcp.py`

**Interfaces:**
- Produces:
  - `sync.PUSH: tuple[tuple[str, str], ...]` — `(local subdir relative to data_root/work_root/ref, bucket path)` pairs; `sync.PULL: tuple[str, ...]` — bucket `work/` subdirs that a run may change.
  - `sync.rsync_args(src: str, dst: str, *, exclude: str = sync.EXCLUDE) -> list[str]`
  - `sync.push_plan(cfg) -> list[tuple[Path, str]]`, `sync.pull_plan(cfg) -> list[tuple[str, Path]]`
  - `gcloud.Gcloud(binary: Path)` with `.run(args, *, check=True, timeout=None) -> subprocess.CompletedProcess`, `.json(args) -> Any`, `.stream(args) -> int` (inherits stdout, returns the exit code).
  - Startup script contract: reads metadata `nepal-profile` (`cpu`|`gpu`), `nepal-branch`, `nepal-bucket`; installs the toolchain once (marker `/data/projects/.toolchain`), installs the NVIDIA driver on `gpu` when `nvidia-smi` is absent, clones or resets the repo, `pip install -e '.[vision,music,asr,faces,semantic,dev]'`, pulls `raw/`, `work/`, `ref/` from the bucket, writes `/data/projects/READY` with a timestamp; logs to `/var/log/nepal-bootstrap.log`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cloud_sync.py
"""What travels, and the rsync commands that move it."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from nepal.cloud import sync
from nepal.config import Config


def _cfg(tmp_path):
    return Config({"project": {"data_root": str(tmp_path / "data"),
                               "work_root": str(tmp_path / "work"),
                               "db_path": str(tmp_path / "work" / "db" / "nepal.sqlite")},
                   "cloud": {"gcp": {"bucket": "gs://b"}}}, path=tmp_path / "config" / "pipeline.yaml")


def test_rsync_args_are_recursive_and_exclude_junk():
    args = sync.rsync_args("/x", "gs://b/y")
    assert args[:3] == ["storage", "rsync", "--recursive"]
    assert any(a.startswith("--exclude=") and "lock" in a for a in args)
    assert args[-2:] == ["/x", "gs://b/y"]


def test_push_plan_covers_the_working_set_and_not_the_originals(tmp_path):
    plan = sync.push_plan(_cfg(tmp_path))
    dsts = {d for _, d in plan}
    assert "gs://b/raw/media_from_phones" in dsts and "gs://b/raw/strava" in dsts
    assert "gs://b/work/proxies" in dsts and "gs://b/work/db" in dsts
    assert "gs://b/ref/data" in dsts
    assert not any("media_from_camera" in d for d in dsts)
    ref = [s for s, d in plan if d == "gs://b/ref/data"][0]
    assert ref == tmp_path / "data_ref" or ref.name == "data"   # <repo>/data


def test_pull_plan_is_what_a_run_can_change(tmp_path):
    plan = sync.pull_plan(_cfg(tmp_path))
    srcs = {s for s, _ in plan}
    assert "gs://b/work/db" in srcs and "gs://b/work/reports" in srcs
    assert "gs://b/work/semantic" in srcs and "gs://b/work/gates" in srcs
    assert "gs://b/work/proxies" not in srcs                    # never changes after S03.1
    assert all(d.is_relative_to(tmp_path / "work") for _, d in plan)
```

```python
# tests/test_bootstrap_gcp.py
"""The startup script: syntax, idempotence markers, and the contract."""
import pathlib
import shutil
import subprocess

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "tools" / "cloud" / "bootstrap-gcp.sh"


@pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")
def test_script_parses():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_script_reads_its_metadata_and_writes_the_markers():
    s = SCRIPT.read_text()
    for attr in ("nepal-profile", "nepal-branch", "nepal-bucket"):
        assert f"attributes/{attr}" in s
    assert "/data/projects/.toolchain" in s          # toolchain once
    assert "/data/projects/READY" in s               # what `up` waits for
    assert "nvidia-smi" in s and "install_gpu_driver.py" in s
    assert "gcloud storage rsync" in s
    assert "git fetch" in s and "reset --hard" in s   # the box tracks the branch
    assert "ANTHROPIC_API_KEY" in s                   # from metadata, into the environment
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/data/projects/nepalvideo/.venv/bin/python -m pytest tests/test_cloud_sync.py tests/test_bootstrap_gcp.py -q`
Expected: FAIL (ImportError, then `FileNotFoundError` on the script).

- [ ] **Step 3: The sync module**

```python
# src/nepal/cloud/sync.py
"""What travels between the machine, the bucket and the box.

The bucket is the hub. Local pushes the working set once and new material
as it lands; the box pulls before a run and pushes what a run may have
changed; local pulls those to look at them. Nothing before conform reads
the camera originals, so they are not in the plan.
"""
from __future__ import annotations

from pathlib import Path

EXCLUDE = r".*\.~lock.*|.*\.DS_Store|.*Thumbs\.db"

# (kind, local subdir, bucket path); kind says which root the subdir is under
PUSH = (
    ("data", "media_from_phones", "raw/media_from_phones"),
    ("data", "chat_export", "raw/chat_export"),
    ("data", "music", "raw/music"),
    ("data", "strava", "raw/strava"),
    ("work", "db", "work/db"),
    ("work", "reports", "work/reports"),
    ("work", "proxies", "work/proxies"),
    ("work", "audio", "work/audio"),
    ("work", "transcripts", "work/transcripts"),
    ("work", "faces", "work/faces"),
    ("work", "music", "work/music"),
    ("work", "gpx", "work/gpx"),
    ("work", "vocab", "work/vocab"),
    ("work", "semantic", "work/semantic"),
    ("ref", "data", "ref/data"),
)

# work/ subdirs a run may change and local wants back
PULL = ("db", "reports", "semantic", "faces", "transcripts", "gates", "overlays", "beats")


def rsync_args(src: str, dst: str, *, exclude: str = EXCLUDE) -> list[str]:
    return ["storage", "rsync", "--recursive", f"--exclude={exclude}", src, dst]


def _bucket(cfg) -> str:
    return str(cfg.get("cloud.gcp.bucket")).rstrip("/")


def push_plan(cfg) -> list[tuple[Path, str]]:
    roots = {"data": cfg.data_root, "work": cfg.work_root, "ref": cfg.base_dir}
    return [(roots[kind] / sub, f"{_bucket(cfg)}/{dst}") for kind, sub, dst in PUSH]


def pull_plan(cfg) -> list[tuple[str, Path]]:
    return [(f"{_bucket(cfg)}/work/{sub}", cfg.work_root / sub) for sub in PULL]
```

- [ ] **Step 4: The runner**

```python
# src/nepal/cloud/gcloud.py
"""The one place that runs `gcloud`."""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Sequence

log = logging.getLogger(__name__)


class GcloudFailed(RuntimeError):
    pass


class Gcloud:
    def __init__(self, binary: str | Path):
        self.binary = str(Path(binary).expanduser())

    def run(self, args: Sequence[str], *, check: bool = True,
            timeout: float | None = None) -> subprocess.CompletedProcess:
        cmd = [self.binary, *args]
        log.debug("gcloud %s", " ".join(args))
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if check and proc.returncode != 0:
            raise GcloudFailed(f"gcloud {' '.join(args[:4])} failed ({proc.returncode}): "
                               f"{(proc.stderr or proc.stdout)[-800:]}")
        return proc

    def json(self, args: Sequence[str]) -> Any:
        proc = self.run(args, check=False)
        if proc.returncode != 0:
            return None
        return json.loads(proc.stdout or "null")

    def stream(self, args: Sequence[str]) -> int:
        """Run with the terminal attached: the operator watches the log."""
        return subprocess.call([self.binary, *args])
```

- [ ] **Step 5: The startup script**

```bash
#!/bin/bash
# tools/cloud/bootstrap-gcp.sh -- GCE startup script for the nepal box.
#
# Runs on EVERY boot (that is what a GCE startup script does), so every step
# is idempotent: the toolchain installs once behind a marker, the repo is
# reset to the branch each time, the bucket pull is an rsync. It never starts
# a stage: `nepal remote run` does that over SSH once /data/projects/READY
# exists. Log: /var/log/nepal-bootstrap.log.
set -euxo pipefail
exec > >(tee -a /var/log/nepal-bootstrap.log) 2>&1

md() { curl -sf -H 'Metadata-Flavor: Google' \
        "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" || true; }
PROFILE="$(md nepal-profile)"; PROFILE="${PROFILE:-cpu}"
BRANCH="$(md nepal-branch)";   BRANCH="${BRANCH:-main}"
BUCKET="$(md nepal-bucket)"
REPO="${NEPAL_REPO:-https://github.com/kirtom/nepalvideo.git}"
USER_NAME=nepal
ROOT=/data/projects
rm -f $ROOT/READY

# -- toolchain, once ---------------------------------------------------
if [ ! -f $ROOT/.toolchain ]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -q
  apt-get install -y -q ffmpeg libimage-exiftool-perl git python3-venv python3-pip \
      sqlite3 build-essential
  id -u $USER_NAME >/dev/null 2>&1 || useradd -m -s /bin/bash $USER_NAME
  mkdir -p $ROOT && chown -R $USER_NAME:$USER_NAME /data
  touch $ROOT/.toolchain
fi

# -- GPU driver, once, only on the gpu profile ---------------------------
if [ "$PROFILE" = "gpu" ] && ! command -v nvidia-smi >/dev/null 2>&1; then
  curl -sSLo /tmp/install_gpu_driver.py \
    https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py
  python3 /tmp/install_gpu_driver.py || echo "driver install reported an error; continuing"
fi

# -- the repo, tracking the branch --------------------------------------
if [ ! -d $ROOT/nepalvideo/.git ]; then
  sudo -u $USER_NAME git clone --branch "$BRANCH" "$REPO" $ROOT/nepalvideo
fi
cd $ROOT/nepalvideo
sudo -u $USER_NAME git fetch -q origin "$BRANCH"
sudo -u $USER_NAME git reset -q --hard "origin/$BRANCH"
[ -d .venv ] || sudo -u $USER_NAME python3 -m venv .venv
sudo -u $USER_NAME .venv/bin/pip install -q --upgrade pip
sudo -u $USER_NAME .venv/bin/pip install -q -e '.[vision,music,asr,faces,semantic,dev]'

# -- the API key, from metadata into the user's environment -------------
KEY="$(md anthropic-api-key)"
if [ -n "$KEY" ]; then
  grep -q ANTHROPIC_API_KEY /home/$USER_NAME/.profile 2>/dev/null || \
    echo "export ANTHROPIC_API_KEY=$KEY" >> /home/$USER_NAME/.profile
fi

# -- the bucket -> the same paths the config names -----------------------
if [ -n "$BUCKET" ]; then
  mkdir -p $ROOT/nepal_data $ROOT/nepal_work $ROOT/nepalvideo/data
  chown -R $USER_NAME:$USER_NAME $ROOT/nepal_data $ROOT/nepal_work
  sudo -u $USER_NAME gcloud storage rsync --recursive "$BUCKET/raw"  $ROOT/nepal_data
  sudo -u $USER_NAME gcloud storage rsync --recursive "$BUCKET/work" $ROOT/nepal_work
  sudo -u $USER_NAME gcloud storage rsync --recursive "$BUCKET/ref/data" $ROOT/nepalvideo/data
fi

sudo -u $USER_NAME .venv/bin/nepal doctor > $ROOT/doctor.txt 2>&1 || true
nvidia-smi -L >> $ROOT/doctor.txt 2>&1 || echo "no GPU visible" >> $ROOT/doctor.txt
date -u +%FT%TZ > $ROOT/READY
```

`chmod +x tools/cloud/bootstrap-gcp.sh`.

- [ ] **Step 6: Run the tests**

Run: `/data/projects/nepalvideo/.venv/bin/python -m pytest tests/test_cloud_sync.py tests/test_bootstrap_gcp.py -q`
Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
git add src/nepal/cloud/sync.py src/nepal/cloud/gcloud.py tools/cloud/bootstrap-gcp.sh tests/test_cloud_sync.py tests/test_bootstrap_gcp.py
git commit -m "What travels, the one gcloud runner, and an idempotent GCE startup script

The bucket is the hub: the push plan is the working set without the
camera originals, the pull plan is what a run may change. The startup
script runs on every boot, so every step is a marker or an rsync, and it
never starts a stage.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013XaQhkq3CDL5Qy4AUKcX5M"
```

---

### Task 4: `nepal remote` — up, down, status, push, pull, run, exec, ssh

**Files:**
- Create: `src/nepal/cloud/remote.py`
- Modify: `src/nepal/cli.py`
- Test: `tests/test_cloud_remote.py`

**Interfaces:**
- Produces: `remote.Remote(cfg, profile: str = "cpu", gcloud: Gcloud | None = None)` with methods `status() -> gce.Status`, `up(wait: bool = True) -> gce.Status`, `down(delete: bool = False) -> None`, `push() -> None`, `pull() -> None`, `run_nepal(args: list[str]) -> int`, `exec_cmd(command: str) -> int`, `ssh() -> int`. `up` records `remote_up_since:<profile>` in the ledger's sidecar (`work/reports/remote_state.json`); `down` records the hours at the profile price into the ledger. `run_nepal` and `exec_cmd` wrap the command as `cd <remote_repo> && git fetch -q && git reset -q --hard origin/<branch> && .venv/bin/pip install -q -e '.[...]' && gcloud storage rsync ... work && <cmd>; rc=$?; gcloud storage rsync ... work back; exit $rc`.

- [ ] **Step 1: Write the failing test** — a fake `gcloud` that records its arguments and answers `describe` from a state file:

```python
# tests/test_cloud_remote.py
"""nepal remote against a fake gcloud that records what it was asked."""
import json
import os
import stat
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest

from nepal.cloud import remote, spend
from nepal.cloud.gcloud import Gcloud
from nepal.config import Config

FAKE = r'''#!/usr/bin/env python3
import json, os, sys
log = os.environ["FAKE_LOG"]; state = os.environ["FAKE_STATE"]
args = sys.argv[1:]
with open(log, "a") as fh: fh.write(json.dumps(args) + "\n")
st = json.load(open(state)) if os.path.exists(state) else {}
if args[:3] == ["compute", "instances", "describe"]:
    if not st.get("exists"): sys.exit(1)
    print(json.dumps({"name": args[3], "status": st.get("status", "RUNNING"),
                      "networkInterfaces": [{"accessConfigs": [{"natIP": "34.0.0.1"}]}]}))
elif args[:3] == ["compute", "instances", "create"]:
    st.update(exists=True, status="RUNNING")
elif args[:3] == ["compute", "instances", "start"]:
    st["status"] = "RUNNING"
elif args[:3] == ["compute", "instances", "stop"]:
    st["status"] = "TERMINATED"
elif args[:3] == ["compute", "instances", "delete"]:
    st.clear()
elif args[:2] == ["compute", "ssh"]:
    cmd = [a for a in args if a.startswith("--command=")]
    if cmd and "test -f /data/projects/READY" in cmd[0]:
        sys.exit(0 if st.get("ready", True) else 1)
json.dump(st, open(state, "w"))
'''


@pytest.fixture
def env(tmp_path, monkeypatch):
    fake = tmp_path / "gcloud"
    fake.write_text(FAKE)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "log"))
    monkeypatch.setenv("FAKE_STATE", str(tmp_path / "state"))
    (tmp_path / "config").mkdir()
    cfg = Config({
        "project": {"data_root": str(tmp_path / "data"), "work_root": str(tmp_path / "work"),
                    "db_path": str(tmp_path / "work" / "db" / "nepal.sqlite")},
        "cloud": {"spend_ceiling_usd": 15, "gcp": {
            "gcloud": str(fake), "project": "p", "zone": "z", "bucket": "gs://b",
            "branch": "br", "repo": "https://x/y.git",
            "remote_data_root": "/data/projects/nepal_data",
            "remote_work_root": "/data/projects/nepal_work",
            "remote_repo": "/data/projects/nepalvideo", "ready_timeout_s": 2,
            "profiles": {"cpu": {"name": "nepal-cpu", "machine_type": "e2-standard-8",
                                 "disk_gb": 60, "image_family": "f", "image_project": "ip",
                                 "accelerator": None, "spot": True, "usd_per_h": 0.5}}}},
    }, path=tmp_path / "config" / "pipeline.yaml")
    (tmp_path / "work" / "reports").mkdir(parents=True)
    return cfg, tmp_path


def calls(tmp_path):
    return [json.loads(l) for l in (tmp_path / "log").read_text().splitlines()]


def test_up_creates_when_absent_then_starts_when_stopped(env):
    cfg, tmp = env
    r = remote.Remote(cfg, "cpu")
    st = r.up(wait=True)
    assert st.state == "RUNNING" and st.ip == "34.0.0.1"
    verbs = [c[2] for c in calls(tmp) if c[:2] == ["compute", "instances"]]
    assert "create" in verbs and "start" not in verbs
    create = next(c for c in calls(tmp) if c[2:3] == ["create"])
    assert "--metadata=nepal-profile=cpu,nepal-branch=br,nepal-bucket=gs://b" in create
    r.down()
    (tmp / "log").write_text("")
    r.up(wait=True)
    verbs = [c[2] for c in calls(tmp) if c[:2] == ["compute", "instances"]]
    assert "start" in verbs and "create" not in verbs


def test_down_records_the_hours_in_the_ledger(env):
    cfg, tmp = env
    r = remote.Remote(cfg, "cpu")
    r.up(wait=True)
    r.down()
    led = spend.ledger(cfg)
    assert len(led.entries) == 1 and led.entries[0].what == "gce:cpu"
    assert 0.0 <= led.entries[0].usd < 0.01           # seconds at 0.5 USD/h
    assert not r.state_path.exists() or "up_since" not in json.loads(r.state_path.read_text()).get("cpu", {})


def test_up_refuses_when_the_ledger_is_at_the_ceiling(env):
    cfg, tmp = env
    spend.ledger(cfg).record("api:x", 15.0)
    with pytest.raises(spend.SpendCeiling):
        remote.Remote(cfg, "cpu").up(wait=False)


def test_run_wraps_the_command_with_sync_and_branch_reset(env):
    cfg, tmp = env
    r = remote.Remote(cfg, "cpu")
    r.up(wait=True)
    rc = r.run_nepal(["s03", "--redo", "place"])
    assert rc == 0
    ssh = [c for c in calls(tmp) if c[:2] == ["compute", "ssh"]]
    cmd = [a for a in ssh[-1] if a.startswith("--command=")][0]
    assert "reset -q --hard origin/br" in cmd
    assert "gcloud storage rsync --recursive gs://b/work /data/projects/nepal_work" in cmd
    assert ".venv/bin/nepal s03 --redo place" in cmd
    assert "gcloud storage rsync --recursive /data/projects/nepal_work gs://b/work" in cmd
    assert cmd.index("nepal s03") < cmd.index("/data/projects/nepal_work gs://b/work")


def test_push_and_pull_issue_one_rsync_per_plan_entry(env):
    cfg, tmp = env
    for kind, sub in (("data", "media_from_phones"), ("work", "db")):
        (tmp / kind / sub).mkdir(parents=True)
    r = remote.Remote(cfg, "cpu")
    r.push()
    rs = [c for c in calls(tmp) if c[:2] == ["storage", "rsync"]]
    assert any(c[-1] == "gs://b/raw/media_from_phones" for c in rs)
    assert any(c[-1] == "gs://b/work/db" for c in rs)
    assert not any("media_from_camera" in c[-1] for c in rs)
    (tmp / "log").write_text("")
    r.pull()
    rs = [c for c in calls(tmp) if c[:2] == ["storage", "rsync"]]
    assert any(c[-2] == "gs://b/work/db" and c[-1].endswith("/work/db") for c in rs)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `/data/projects/nepalvideo/.venv/bin/python -m pytest tests/test_cloud_remote.py -q`
Expected: FAIL with `ImportError: cannot import name 'remote'`.

- [ ] **Step 3: Write the module**

```python
# src/nepal/cloud/remote.py
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
from pathlib import Path
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
        led.guard(self.profile.usd_per_h, ceiling_usd=float(self.cfg.get("cloud.spend_ceiling_usd")))
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
            self.gcloud.run(gce.start_args(self.profile.name, project=self.project, zone=self.zone))
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
            proc = self.gcloud.run(gce.ssh_args(self.profile.name, project=self.project,
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
            hours = (datetime.now(timezone.utc) - datetime.fromisoformat(since)).total_seconds() / 3600
            spend.ledger(self.cfg).record(
                f"gce:{self.profile_key}", hours * self.profile.usd_per_h,
                detail=f"{hours:.2f} h {self.profile.machine_type} at {self.profile.usd_per_h} USD/h (estimate)")
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
        pull = f"gcloud storage rsync --recursive {self.bucket}/work {self.remote_work}"
        push = f"gcloud storage rsync --recursive {self.remote_work} {self.bucket}/work"
        return (f"cd {self.remote_repo} && git fetch -q origin {self.branch} && "
                f"git reset -q --hard origin/{self.branch} && "
                f".venv/bin/pip install -q -e '.[{EXTRAS}]' && {pull} && "
                f"({command}); rc=$?; {push}; exit $rc")

    def exec_cmd(self, command: str) -> int:
        return self.gcloud.stream(gce.ssh_args(self.profile.name, project=self.project,
                                               zone=self.zone, command=self._wrap(command)))

    def run_nepal(self, args: Sequence[str]) -> int:
        return self.exec_cmd(".venv/bin/nepal " + " ".join(shlex.quote(a) for a in args))

    def ssh(self) -> int:
        return self.gcloud.stream(gce.ssh_args(self.profile.name, project=self.project,
                                               zone=self.zone))
```

- [ ] **Step 4: CLI**

In `src/nepal/cli.py` add a parser after `prune`:

```python
    prem = sub.add_parser("remote", help="run things on the GCP box (Film v2 section 2.2)")
    prem.add_argument("action", choices=["up", "down", "status", "push", "pull", "run",
                                         "exec", "ssh"])
    prem.add_argument("--gpu", action="store_true", help="the GPU profile instead of cpu")
    prem.add_argument("--delete", action="store_true", help="down: delete rather than stop")
    prem.add_argument("--no-wait", action="store_true", help="up: do not wait for READY")
    prem.add_argument("rest", nargs=argparse.REMAINDER,
                      help="run: nepal arguments; exec: a shell command (after --)")
```

and the dispatch:

```python
    if args.cmd == "remote":
        from nepal.cloud import remote as remote_mod, spend
        r = remote_mod.Remote(cfg, "gpu" if args.gpu else "cpu")
        rest = [a for a in args.rest if a != "--"]
        if args.action == "status":
            st = r.status()
            led = spend.ledger(cfg)
            print(f"{r.profile.name}: {st.state}{' at ' + st.ip if st.ip else ''}; "
                  f"ledger {led.total():.2f} of {cfg.get('cloud.spend_ceiling_usd')} USD")
            return 0
        if args.action == "up":
            st = r.up(wait=not args.no_wait)
            print(f"{r.profile.name}: {st.state} at {st.ip}")
            return 0
        if args.action == "down":
            r.down(delete=args.delete)
            return 0
        if args.action == "push":
            r.push(); return 0
        if args.action == "pull":
            r.pull(); return 0
        if args.action == "run":
            return r.run_nepal(rest)
        if args.action == "exec":
            return r.exec_cmd(" ".join(rest))
        if args.action == "ssh":
            return r.ssh()
```

Add the `remote` lines to the docstring at the top of `cli.py`.

- [ ] **Step 5: Run the test**

Run: `/data/projects/nepalvideo/.venv/bin/python -m pytest tests/test_cloud_remote.py -q`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/nepal/cloud/remote.py src/nepal/cli.py tests/test_cloud_remote.py
git commit -m "nepal remote: up, run, pull, down on one Spot box with the bucket as the hub

up creates or starts and waits for READY; run tracks the branch, pulls
work/ from the bucket, runs, and pushes work/ back whatever the exit code,
so a killed run keeps its checkpoints; down stops, keeps the disk, and
writes the hours to the ledger. Tested against a fake gcloud that records
what it was asked.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013XaQhkq3CDL5Qy4AUKcX5M"
```

---

### Task 5: The manifest on a host that holds part of the corpus

The box has phones, chat, music and Strava but not the 60 GB of camera originals. Today `build_manifest` deletes every asset row whose file it cannot see, which would delete the camera recordings and their shots. It must instead leave a source alone when that source's directory is absent on this host. And it must not re-hash 8 GB on every re-probe.

**Files:**
- Modify: `src/nepal/probe/manifest.py` (add `absent_sources`), `src/nepal/stages/s01_probe.py:200-217` (orphan removal), `:144-167` (hash reuse), `src/nepal/db.py` (`assets.mtime` migration)
- Test: `tests/test_manifest_partial.py`

**Interfaces:**
- Produces: `manifest.SOURCE_DIRS = {"camera": "media_from_camera", "phone_keller": "media_from_phones/keller", "phone_kulikov": "media_from_phones/kulikov", "telegram": "chat_export", "music": "music"}`; `manifest.absent_sources(root: Path) -> set[str]`; `assets.mtime REAL`; `s01_probe.build_manifest` reuses `asset_id` for a file whose `(s3_key, bytes, mtime)` match an existing row and reports `n_hash_reused`; orphan removal skips sources in `absent_sources(root)` and reports `sources_absent`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_manifest_partial.py
"""A host that holds part of the corpus must not delete the rest."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from nepal import db
from nepal.config import Config
from nepal.probe import manifest
from nepal.stages import s01_probe


def _cfg(tmp_path):
    data = tmp_path / "data"
    (data / "media_from_phones" / "keller").mkdir(parents=True)
    (data / "media_from_phones" / "keller" / "IMG_1.jpg").write_bytes(b"\xff\xd8jpeg-ish")
    return Config({"project": {"data_root": str(data), "work_root": str(tmp_path / "work"),
                               "db_path": str(tmp_path / "work" / "db" / "nepal.sqlite")},
                   "probe": {"max_chapter_gap_s": 1.0, "exclude_dirs": [], "ignore_globs": [],
                             "capture_time_overrides": {}}}), data


def test_absent_sources_names_what_this_host_does_not_hold(tmp_path):
    _, data = _cfg(tmp_path)
    assert manifest.absent_sources(data) == {"camera", "phone_kulikov", "telegram", "music"}


def test_rows_of_an_absent_source_survive_a_re_probe(tmp_path):
    cfg, data = _cfg(tmp_path)
    conn = db.init(cfg.db_path)
    conn.execute("INSERT INTO assets(asset_id, s3_key, source, kind) VALUES "
                 "('cam1', 'raw/media_from_camera/VID_1.insv', 'camera', 'video360')")
    conn.execute("INSERT INTO assets(asset_id, s3_key, source, kind) VALUES "
                 "('gone', 'raw/media_from_phones/keller/IMG_9.jpg', 'phone_keller', 'photo')")
    conn.commit()
    rep = s01_probe.build_manifest(cfg, conn)
    ids = {r[0] for r in conn.execute("SELECT asset_id FROM assets")}
    assert "cam1" in ids                       # its source is absent here: untouched
    assert "gone" not in ids                   # its source is present and the file is not
    assert rep["sources_absent"] == sorted({"camera", "phone_kulikov", "telegram", "music"})


def test_an_unchanged_file_is_not_hashed_again(tmp_path, monkeypatch):
    cfg, data = _cfg(tmp_path)
    conn = db.init(cfg.db_path)
    first = s01_probe.build_manifest(cfg, conn)
    assert first["n_hash_reused"] == 0
    calls = []
    real = s01_probe.sha256_file
    monkeypatch.setattr(s01_probe, "sha256_file", lambda p: (calls.append(p), real(p))[1])
    second = s01_probe.build_manifest(cfg, conn)
    assert second["n_hash_reused"] == 1 and calls == []
    (data / "media_from_phones" / "keller" / "IMG_1.jpg").write_bytes(b"\xff\xd8changed!")
    third = s01_probe.build_manifest(cfg, conn)
    assert third["n_hash_reused"] == 0 and len(calls) == 1
```

- [ ] **Step 2: Run it to verify it fails**

Run: `/data/projects/nepalvideo/.venv/bin/python -m pytest tests/test_manifest_partial.py -q`
Expected: FAIL with `AttributeError: module 'nepal.probe.manifest' has no attribute 'absent_sources'`.

- [ ] **Step 3: Implement**

In `src/nepal/probe/manifest.py`, after `DEFAULT_EXCLUDE_DIRS`:

```python
# Where each source lives under data_root. A host that holds only part of the
# corpus -- the GCP box has the phones, the chat and the music but not 60 GB
# of camera originals -- must leave the rows of the sources it cannot see
# alone rather than treat every one of them as a deleted file.
SOURCE_DIRS = {
    "camera": "media_from_camera",
    "phone_keller": "media_from_phones/keller",
    "phone_kulikov": "media_from_phones/kulikov",
    "telegram": "chat_export",
    "music": "music",
}


def absent_sources(root: Path) -> set[str]:
    """Sources whose directory does not exist under ``root`` on this host."""
    return {src for src, sub in SOURCE_DIRS.items() if not (Path(root) / sub).is_dir()}
```

In `src/nepal/db.py` `MIGRATIONS` add `("assets", "mtime", "REAL"),` with the comment `# so a re-probe hashes only what changed`.

In `src/nepal/stages/s01_probe.py::build_manifest`: before the walk loop, load the known rows:

```python
    known = {r["s3_key"]: (r["asset_id"], r["bytes"], r["mtime"]) for r in conn.execute(
        "SELECT s3_key, asset_id, bytes, mtime FROM assets")}
    reused = 0
```

in the loop, replace `"asset_id": sha256_file(path),` and `"bytes": path.stat().st_size,` with:

```python
        st = path.stat()
        prev = known.get(f"raw/{rel}")
        if prev and prev[1] == st.st_size and prev[2] is not None and \
                abs(float(prev[2]) - st.st_mtime) < 1.0:
            asset_id = prev[0]                 # same bytes as last time, by size and mtime
            reused += 1
        else:
            asset_id = sha256_file(path)
```

and in the dict `"asset_id": asset_id, "bytes": st.st_size, "mtime": st.st_mtime,`. Replace the orphan block:

```python
    keep = {r["asset_id"] for r in unique_rows}
    absent = manifest.absent_sources(root)
    orphans = [r["asset_id"] for r in conn.execute("SELECT asset_id, source FROM assets")
               if r["asset_id"] not in keep and r["source"] not in absent]
    if absent:
        log.info("S01.1 this host does not hold %s; their rows are left untouched",
                 ", ".join(sorted(absent)))
```

and add `"n_hash_reused": reused, "sources_absent": sorted(absent),` to the return dict.

- [ ] **Step 4: Run the tests**

Run: `/data/projects/nepalvideo/.venv/bin/python -m pytest tests/test_manifest_partial.py tests/test_manifest.py -q`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add src/nepal/probe/manifest.py src/nepal/stages/s01_probe.py src/nepal/db.py tests/test_manifest_partial.py
git commit -m "The manifest on a host that holds part of the corpus

The box has the phones, the chat and the music but not 60 GB of camera
originals, and the manifest treated every file it could not see as
deleted. A source whose directory is absent on this host is now left
alone. And a file whose size and mtime match its row is not hashed again,
so a re-probe costs exiftool, not 8 GB of sha256.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013XaQhkq3CDL5Qy4AUKcX5M"
```

---

### Task 6: First remote runs, docs

**Files:**
- Modify: `README.md` ("Running it in the cloud"), `docs/STATE.md` ("The cloud move"), `CLAUDE.md` (Running stages)

- [ ] **Step 1: Push the branch and bring the box up**

```bash
git push origin worktree-film-v2-design
N=/data/projects/nepalvideo/.venv/bin/nepal
$N remote status
$N remote up                     # creates nepal-cpu, waits for READY (~10 min first time)
$N remote exec -- cat /data/projects/doctor.txt
```

Expected: the doctor output says `editable -- this checkout is what runs`, ffmpeg and exiftool present, every package present.

- [ ] **Step 2: The test suites, on the box**

```bash
$N remote exec -- .venv/bin/python -m pytest -q
$N remote exec -- .venv/bin/python -m pytest -m slow -q
```

Expected: both pass. Record the counts.

- [ ] **Step 3: The three jobs step 1 deferred**

```bash
$N remote run s01 --redo manifest,chapters --skip-fov --skip-clock   # heading tags + 35 new photos
$N remote run prune
$N remote run s02 --redo gps_track,geotag,acts
$N remote run s03 --redo place,faces                                  # the 667 unseen shots, on 8 cores
$N remote run cut                                                     # score, timeline, draft render
$N remote pull
$N remote down
$N remote status                                                      # the ledger line
```

Expected: `s01` reports `n_hash_reused` near 1,300 and `sources_absent: ["camera"]`; the 35 new photos become assets and photo shots; faces run in well under an hour on 8 cores; the draft lands in `work/gates/gate3/draft.mp4` and `remote pull` brings it here. Note the wall-clock of each in STATE.md.

- [ ] **Step 4: Docs**

`README.md` "Running it in the cloud": replace the AWS history with the GCP model (bucket hub, the six commands, the two profiles, where credentials live, the ledger) and move the AWS measurements into a short "what was learned on AWS" paragraph. `docs/STATE.md` "The cloud move": the GCP facts, the first remote timings, the ledger total. `CLAUDE.md` "Running stages": replace the first paragraph with "Nothing computes on the operator's machine: stages, tests and renders run through `nepal remote`. Locally, only edits, git and one-second queries."

- [ ] **Step 5: Commit and push**

```bash
git add README.md docs/STATE.md CLAUDE.md
git commit -m "Step 2 of Film v2 done: everything runs on the GCP box

First remote runs: the test suites, the manifest re-probe with the heading
tags and the 35 new photos, faces on the 667 shots the EC2 pass never saw,
and a fresh draft. The ledger records what it cost.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013XaQhkq3CDL5Qy4AUKcX5M"
git push origin worktree-film-v2-design
```

---

## Self-review

**Spec coverage:** §2.2 execution model → Tasks 1–4 and 6 (the six commands, bucket hub, two profiles, Spot, stop-not-delete, credentials outside the repo, the ledger and ceiling). §8 items deferred from step 1 that need the box → Task 6 step 3. The partial-host manifest is new, forced by "not the originals until conform" in §2.2 → Task 5. The serverless-GPU fallback is a decision, not code, and stays in the spec.

**Placeholder scan:** none.

**Type consistency:** `gce.Profile` fields match `Profile.from_cfg` and the config block; `Remote.up/down/push/pull/run_nepal/exec_cmd` match the CLI dispatch; `spend.ledger(cfg)` is used by `Remote` and the CLI; `sync.push_plan/pull_plan` signatures match `Remote.push/pull`; the fake gcloud answers exactly the argument shapes the builders produce.
