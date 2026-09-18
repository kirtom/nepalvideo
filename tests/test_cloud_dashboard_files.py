"""The checked-in dashboard artefacts: valid, and pointing at the right things."""
import pathlib

import yaml

CLOUD = pathlib.Path(__file__).resolve().parents[1] / "tools" / "cloud"


def test_ops_agent_config_tails_the_pipeline_logs():
    cfg = yaml.safe_load((CLOUD / "ops-agent.yaml").read_text())
    recv = cfg["logging"]["receivers"]["nepal_pipeline"]
    assert recv["type"] == "files"
    assert "/data/projects/nepal_work/reports/remote_jobs/*.log" in recv["include_paths"]
    assert "/var/log/nepal-bootstrap.log" in recv["include_paths"]
    pipe = cfg["logging"]["service"]["pipelines"]["nepal"]
    assert "nepal_pipeline" in pipe["receivers"]
    assert cfg["metrics"]["receivers"]["hostmetrics"]["type"] == "hostmetrics"


def test_bootstrap_installs_the_agent_once():
    s = (CLOUD / "bootstrap-gcp.sh").read_text()
    assert "add-google-cloud-ops-agent-repo.sh" in s
    assert "/etc/google-cloud-ops-agent/config.yaml" in s
    assert "$ROOT/.ops-agent" in s
