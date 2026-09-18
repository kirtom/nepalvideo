"""nepal remote against a fake gcloud that records what it was asked."""
import json
import stat
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest

from nepal.cloud import remote, spend
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
    assert "up_since" not in json.loads(r.state_path.read_text()).get("cpu", {})


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
    assert "gcloud storage rsync --recursive gs://b/raw /data/projects/nepal_data" in cmd
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
