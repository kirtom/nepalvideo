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
    assert "--scopes=storage-rw,logging-write,monitoring-write" in s   # the bucket, and the agent
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
    as_user = gce.ssh_args("nepal-cpu", project="p", zone="z", user="nepal")
    assert as_user[2] == "nepal@nepal-cpu"


def test_parse_describe_reads_state_and_ip():
    doc = {"name": "nepal-cpu", "status": "RUNNING",
           "networkInterfaces": [{"accessConfigs": [{"natIP": "34.1.2.3"}]}]}
    st = gce.parse_describe(doc)
    assert st.state == "RUNNING" and st.ip == "34.1.2.3" and st.name == "nepal-cpu"
    assert gce.parse_describe({"name": "n", "status": "TERMINATED"}).ip is None
    assert gce.parse_describe({}).state == "ABSENT"
