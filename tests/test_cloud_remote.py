"""nepal remote against a fake gcloud that records what it was asked."""
import json
import stat
from datetime import datetime, timedelta, timezone
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest

from nepal.cloud import remote, spend, watchdog
from nepal.config import Config

FAKE = r'''#!/usr/bin/env python3
import datetime, json, os, sys
log = os.environ["FAKE_LOG"]; state = os.environ["FAKE_STATE"]
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
args = sys.argv[1:]
with open(log, "a") as fh: fh.write(json.dumps(args) + "\n")
st = json.load(open(state)) if os.path.exists(state) else {}
if args[:3] == ["compute", "instances", "describe"]:
    if not st.get("exists"): sys.exit(1)
    print(json.dumps({"name": args[3], "status": st.get("status", "RUNNING"),
                      "lastStartTimestamp": st.get("started"),
                      "lastStopTimestamp": st.get("stopped"),
                      "networkInterfaces": [{"accessConfigs": [{"natIP": "34.0.0.1"}]}]}))
elif args[:3] == ["compute", "instances", "create"]:
    st.update(exists=True, status="RUNNING", started=now())
elif args[:3] == ["compute", "instances", "start"]:
    st.update(status="RUNNING", started=now())
elif args[:3] == ["compute", "instances", "stop"]:
    st.update(status="TERMINATED", stopped=now())
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
            "idle_stop_min": 30, "idle_check_min": 5, "stuck_job_min": 240,
            "profiles": {"cpu": {"name": "nepal-cpu", "machine_type": "e2-standard-8",
                                 "disk_gb": 60, "image_family": "f", "image_project": "ip",
                                 "accelerator": None, "spot": True, "usd_per_h": 0.5}}}},
    }, path=tmp_path / "config" / "pipeline.yaml")
    (tmp_path / "work" / "reports").mkdir(parents=True)
    return cfg, tmp_path


def calls(tmp_path):
    return [json.loads(l) for l in (tmp_path / "log").read_text().splitlines()]


def commands(tmp_path):
    return [a for c in calls(tmp_path) if c[:2] == ["compute", "ssh"]
            for a in c if a.startswith("--command=")]


def _instance(tmp_path, **fields):
    """Rewrite what the fake gcloud reports for the instance."""
    st = json.loads((tmp_path / "state").read_text())
    st.update(fields)
    (tmp_path / "state").write_text(json.dumps(st))


def _ago(hours: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


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
    # the box boots the startup script in its metadata, so a start first
    # pushes the checkout's copy
    assert verbs.index("add-metadata") < verbs.index("start")
    meta = next(c for c in calls(tmp) if c[2:3] == ["add-metadata"])
    assert any(a.startswith("--metadata-from-file=startup-script=") and
               a.endswith("bootstrap-gcp.sh") for a in meta)


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


def test_the_running_box_is_counted_before_anybody_books_it(env):
    """`down` books the hours; a session that never reaches one leaves the
    meter running and the ledger frozen -- 104 idle hours in 2026-09."""
    cfg, tmp = env
    r = remote.Remote(cfg, "cpu")
    r.up(wait=True)
    _instance(tmp, started=_ago(4))
    acct = r.account()
    assert acct["booked"] == 0.0                       # nothing recorded yet
    assert acct["unbooked_h"] == pytest.approx(4.0, abs=0.01)
    assert acct["unbooked_usd"] == pytest.approx(2.0, abs=0.01)   # 4 h at 0.5
    assert acct["total"] == pytest.approx(2.0, abs=0.01)
    assert acct["since"].tzinfo is not None


def test_exec_refuses_when_the_running_box_has_eaten_the_ceiling(env):
    """The refusal has to arrive before the command, not at the `down`
    that the abandoned box never got."""
    cfg, tmp = env
    r = remote.Remote(cfg, "cpu")
    r.up(wait=True)
    spend.ledger(cfg).record("api:x", 10.0)
    _instance(tmp, started=_ago(10))                   # 5.00 USD unbooked, ceiling 15
    with pytest.raises(spend.SpendCeiling) as exc:
        r.exec_cmd("true")
    assert exc.value.total == pytest.approx(15.0, abs=0.01)


def test_a_box_that_stopped_without_down_is_booked_at_the_next_command(env):
    """The idle watchdog stops the box from inside the guest; nothing local
    ran `down`, so the hours come from GCP's own stamps -- once."""
    cfg, tmp = env
    r = remote.Remote(cfg, "cpu")
    r.up(wait=True)
    _instance(tmp, status="TERMINATED", started=_ago(3), stopped=_ago(1))
    r.account()
    led = spend.ledger(cfg)
    assert [e.what for e in led.entries] == ["gce:cpu"]
    assert led.entries[0].usd == pytest.approx(1.0, abs=0.01)     # 2 h at 0.5
    assert "without `down`" in led.entries[0].detail
    assert "up_since" not in json.loads(r.state_path.read_text())["cpu"]
    r.account()                                        # and not a second time
    assert len(spend.ledger(cfg).entries) == 1
    # remote_state.json rsyncs both ways, so a `pull` brings the popped
    # `up_since` back from the bucket. The ledger is what remembers.
    state = json.loads(r.state_path.read_text())
    state["cpu"]["up_since"] = "2026-09-24T00:00:00+00:00"
    r.state_path.write_text(json.dumps(state))
    r.account()
    assert len(spend.ledger(cfg).entries) == 1


def test_up_installs_the_idle_watchdog_on_a_box_that_is_already_running(env):
    """The bootstrap installs it at boot, which never reaches the box that
    is up already -- and that is the box the incident was about."""
    cfg, tmp = env
    remote.Remote(cfg, "cpu").up(wait=True)
    assert any("systemctl enable --now nepal-idle.timer" in c for c in commands(tmp))


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
    # the watchdog's stamp, before the command and after it: a detached job
    # leaves the second one hours behind, which is why the watchdog reads
    # the job logs too
    touch = "touch /data/projects/nepal_work/reports/.last_activity"
    assert cmd.count(touch) == 2
    assert cmd.index(touch) < cmd.index("nepal s03") < cmd.rindex(touch)


def test_the_wrapped_command_still_looks_like_a_job_to_the_watchdog(env):
    """The watchdog decides a box is busy by finding the venv in a command
    line. A `_wrap` that stopped naming it would leave every long run
    invisible to the thing that stops the box."""
    cfg, tmp = env
    assert watchdog.JOB_MARK in remote.Remote(cfg, "cpu")._wrap("x")


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
