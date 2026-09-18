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


def test_dashboard_json_has_the_panels():
    import json
    d = json.loads((CLOUD / "dashboard.json").read_text())
    assert d["displayName"] == "nepal"
    titles = {t["widget"]["title"] for t in d["mosaicLayout"]["tiles"]}
    assert {"CPU", "Memory", "Disk", "Pipeline log"} <= titles
    logs = [t["widget"] for t in d["mosaicLayout"]["tiles"]
            if t["widget"]["title"] == "Pipeline log"][0]
    assert 'logName="projects/nepalvideo/logs/nepal_pipeline"' in logs["logsPanel"]["filter"]


def test_alert_policy_watches_the_error_metric():
    import json
    a = json.loads((CLOUD / "alert-errors.json").read_text())
    cond = a["conditions"][0]["conditionThreshold"]
    assert "logging.googleapis.com/user/nepal_errors" in cond["filter"]
    assert cond["comparison"] == "COMPARISON_GT" and cond["thresholdValue"] == 0


def test_setup_script_creates_everything_idempotently():
    s = (CLOUD / "monitoring-setup.sh").read_text()
    for needle in ("logging metrics create nepal_errors", "monitoring channels",
                   "monitoring policies create", "monitoring dashboards create",
                   "dashboards list", "policies list", "metrics describe"):
        assert needle in s
