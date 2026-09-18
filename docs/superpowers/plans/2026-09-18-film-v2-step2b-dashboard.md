# Film v2 — Step 2b: the dashboard — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One Cloud Monitoring dashboard the operator opens in the console to see whether the box is running, which stage it is at, whether anything failed, and what the project has spent, plus a static status page in the bucket regenerated after every stage.

**Architecture:** The Ops Agent on the box tails the per-stage logs into Cloud Logging under one log name; a dashboard (JSON, checked in, created from the command line) shows the box's CPU, memory and disk and a logs panel of the pipeline's stage lines; a log-based metric on ERROR/Traceback drives an alert that emails the operator. A pure `status` module turns the database, the reports and the ledger into one JSON document and one HTML page, written to `work/status/` at the end of every remote run and pushed with `work/`.

**Tech Stack:** Google Cloud Ops Agent, Cloud Logging, Cloud Monitoring dashboards and alerting (all via `gcloud`), Python for the status page.

**Spec:** `docs/superpowers/specs/2026-09-16-film-v2-voice-spine-design.md` §2.2; approved by the operator 2026-09-18 ("agree on the dashboard proposal").

## Global Constraints

- Nothing computes on the local machine; the unit tests here are pure and take under a second per file.
- Every tunable in `config/pipeline.yaml`; the dashboard and alert definitions live under `tools/cloud/` so they can be recreated.
- `gcloud` commands that the worktree guard refuses (words like `enable`) go into a script file under `$CLAUDE_JOB_DIR/tmp` and run from there.
- Commit per task, message ending with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

**Facts:** project `nepalvideo`, zone `europe-west4-a`, instance `nepal-cpu` (and `nepal-gpu` later), bucket `gs://nepalvideo-29922345852`, operator email `kirill.keller.tech@gmail.com`, per-stage logs at `/data/projects/nepal_work/reports/remote_jobs/*.log`, bootstrap log `/var/log/nepal-bootstrap.log`.

## File structure

| File | Responsibility |
|---|---|
| `tools/cloud/ops-agent.yaml` (new) | Ops Agent config: which files to tail, under which log name |
| `tools/cloud/bootstrap-gcp.sh` | installs the Ops Agent once, drops the config, restarts the agent |
| `src/nepal/cloud/status.py` (new) | `build_status(conn, ledger, reports_dir) -> dict` (pure), `render_html(status) -> str` (pure), `write_status(cfg) -> Path` |
| `src/nepal/cli.py` | `nepal status-page` |
| `src/nepal/cloud/remote.py` | `_wrap` runs `nepal status-page` after the command, before the push |
| `tools/cloud/jobs-step2.sh` | same, before its push |
| `tools/cloud/dashboard.json` (new) | the Cloud Monitoring dashboard |
| `tools/cloud/alert-errors.json` (new) | the alert policy on the error metric |
| `tools/cloud/monitoring-setup.sh` (new) | creates the log-based metric, the notification channel, the alert policy and the dashboard; idempotent |
| `tests/test_cloud_status.py`, `tests/test_cloud_dashboard_files.py` (new) | pure tests |
| `README.md`, `docs/STATE.md` | where to look |

---

### Task 1: The Ops Agent tails the pipeline's logs

**Files:**
- Create: `tools/cloud/ops-agent.yaml`
- Modify: `tools/cloud/bootstrap-gcp.sh`
- Test: `tests/test_cloud_dashboard_files.py` (first test)

- [ ] **Step 1: The test**

```python
# tests/test_cloud_dashboard_files.py
"""The checked-in dashboard artefacts: valid, and pointing at the right things."""
import json
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


def test_bootstrap_installs_the_agent_once():
    s = (CLOUD / "bootstrap-gcp.sh").read_text()
    assert "add-google-cloud-ops-agent-repo.sh" in s
    assert "/etc/google-cloud-ops-agent/config.yaml" in s
    assert "$ROOT/.ops-agent" in s
```

- [ ] **Step 2: Run it to verify it fails**

Run: `/data/projects/nepalvideo/.venv/bin/python -m pytest tests/test_cloud_dashboard_files.py -q -p no:cacheprovider`
Expected: FAIL, `FileNotFoundError` on `ops-agent.yaml`.

- [ ] **Step 3: The config and the bootstrap step**

```yaml
# tools/cloud/ops-agent.yaml -- what the box ships to Cloud Logging.
# Every stage's full log and the bootstrap log, under one log name, so the
# dashboard's logs panel and the error alert have one thing to filter on.
logging:
  receivers:
    nepal_pipeline:
      type: files
      include_paths:
        - /data/projects/nepal_work/reports/remote_jobs/*.log
        - /var/log/nepal-bootstrap.log
      record_log_file_path: true
  processors:
    nepal_severity:
      type: parse_regex
      field: message
      regex: '^(?<time>\d\d:\d\d:\d\d) (?<severity>[A-Z]+) +(?<logger>\S+) \| (?<message>.*)$'
  service:
    pipelines:
      nepal:
        receivers: [nepal_pipeline]
        processors: [nepal_severity]
metrics:
  receivers:
    hostmetrics:
      type: hostmetrics
      collection_interval: 60s
  service:
    pipelines:
      default_pipeline:
        receivers: [hostmetrics]
```

In `tools/cloud/bootstrap-gcp.sh`, after the toolchain block:

```bash
# -- the Ops Agent, once: ships the stage logs and host metrics ---------
if [ ! -f $ROOT/.ops-agent ]; then
  curl -sSo /tmp/add-ops-agent.sh https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh
  bash /tmp/add-ops-agent.sh --also-install || echo "ops agent install failed; continuing"
  touch $ROOT/.ops-agent
fi
# the config every boot: the repo may have changed it
cp $ROOT/nepalvideo/tools/cloud/ops-agent.yaml /etc/google-cloud-ops-agent/config.yaml 2>/dev/null \
  && systemctl restart google-cloud-ops-agent 2>/dev/null || true
```

Place the `cp` after the repo checkout (the file comes from the repo).

- [ ] **Step 4: Run the tests, commit**

```bash
git add tools/cloud/ops-agent.yaml tools/cloud/bootstrap-gcp.sh tests/test_cloud_dashboard_files.py
git commit -m "The Ops Agent ships the box's stage logs and host metrics

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: The status document and page

**Files:**
- Create: `src/nepal/cloud/status.py`
- Modify: `src/nepal/cli.py`, `src/nepal/cloud/remote.py`, `tools/cloud/jobs-step2.sh`, `src/nepal/cloud/sync.py` (add `status` to `PULL`)
- Test: `tests/test_cloud_status.py`

**Interfaces:**
- `status.build_status(conn, ledger: spend.Ledger, reports_dir: Path, *, now: datetime | None = None) -> dict` with keys `generated_utc`, `stages` (list of `{stage, unit, status, updated_at}` from `stage_units`), `latest_stage` (the most recently updated unit), `counts` (`assets`, `dated_assets`, `recordings`, `shots`, `surviving`, `shortlisted`, `slots`, `timeline_s`), `spend` (`total_usd`, `entries[-5:]`), `last_report` (name and `finished_utc` of the newest `reports/s0*.json`), `draft` (`path`, `bytes`, `mtime` if `gates/gate3/draft.mp4` exists), `remote_jobs_tail` (last 15 lines of `reports/remote_jobs.log` if present).
- `status.render_html(status: dict) -> str` — a single self-contained page, no scripts, dark background, one table per section.
- `status.write_status(cfg) -> Path` writes `work/status/status.json` and `work/status/index.html`.

- [ ] **Step 1: The test**

```python
# tests/test_cloud_status.py
"""The status document: what is running, what has run, what it cost."""
import json
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from datetime import datetime, timezone

from nepal import db
from nepal.cloud import spend, status


def _seed(tmp_path):
    conn = db.init(tmp_path / "db" / "nepal.sqlite")
    conn.executemany("INSERT INTO assets(asset_id, s3_key, source, kind, created_at_utc) "
                     "VALUES (?,?,?,?,?)", [("a", "k1", "phone_keller", "photo", "2024-05-01T00:00:00+00:00"),
                                            ("b", "k2", "camera", "video360", None)])
    conn.execute("INSERT INTO recordings(recording_id, source, is_360) VALUES ('r', 'camera', 1)")
    conn.executemany("INSERT INTO shots(shot_id, recording_id, media_kind, start_s, end_s, status) VALUES (?,?,?,?,?,?)",
                     [("r#0", "r", "video", 0, 5, "shortlisted"), ("r#1", "r", "video", 5, 9, "rejected")])
    conn.execute("INSERT INTO timeline(slot_index, act, shot_id, t_in, t_out) VALUES (0, 3, 'r#0', 0, 4.5)")
    db.mark_unit(conn, "S03", "faces", detail="x")
    db.mark_unit(conn, "S03", "place")
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "s03_process.json").write_text(json.dumps({"stage": "S03", "finished_utc": "2026-09-18T05:00:00+00:00"}))
    (reports / "remote_jobs.log").write_text("--- s03 05:00:00\nS03.6 faces 10/10\n")
    led = spend.Ledger(reports / "spend.json")
    led.record("gce:cpu", 0.42)
    return conn, led, reports


def test_build_status_reads_everything_from_the_database_and_the_reports(tmp_path):
    conn, led, reports = _seed(tmp_path)
    st = status.build_status(conn, led, reports, now=datetime(2026, 9, 18, 6, tzinfo=timezone.utc))
    assert st["counts"] == {"assets": 2, "dated_assets": 1, "recordings": 1, "shots": 2,
                            "surviving": 1, "shortlisted": 1, "slots": 1, "timeline_s": 4.5}
    assert {s["unit"] for s in st["stages"]} == {"faces", "place"}
    assert st["latest_stage"]["unit"] == "place"
    assert st["spend"]["total_usd"] == 0.42 and st["spend"]["entries"][-1]["what"] == "gce:cpu"
    assert st["last_report"] == {"name": "s03_process.json", "finished_utc": "2026-09-18T05:00:00+00:00"}
    assert st["draft"] is None
    assert st["remote_jobs_tail"][-1] == "S03.6 faces 10/10"
    assert st["generated_utc"] == "2026-09-18T06:00:00+00:00"


def test_render_html_is_self_contained_and_names_the_stage(tmp_path):
    conn, led, reports = _seed(tmp_path)
    html = status.render_html(status.build_status(conn, led, reports))
    assert html.startswith("<!doctype html>") and "<script" not in html
    assert "place" in html and "0.42" in html and "S03.6 faces 10/10" in html
```

- [ ] **Step 2: Run it to verify it fails**, then **Step 3: the module**

```python
# src/nepal/cloud/status.py
"""One document that says where the pipeline is.

Built from what already exists -- stage_units, the tables, the reports, the
ledger -- so it is never a second source of truth. Written to work/status/
at the end of every remote run and pushed with work/, so the bucket always
holds the latest, and the console's object viewer shows it to the operator
without a server.
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nepal.cloud import spend


def build_status(conn, ledger: spend.Ledger, reports_dir: Path, *,
                 now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    q = lambda sql: conn.execute(sql).fetchone()[0]
    stages = [dict(r) for r in conn.execute(
        "SELECT stage, unit_id AS unit, status, updated_at FROM stage_units "
        "WHERE unit_id NOT LIKE 'proxy:%' ORDER BY updated_at")]
    latest = stages[-1] if stages else None
    counts = {
        "assets": q("SELECT COUNT(*) FROM assets"),
        "dated_assets": q("SELECT COUNT(*) FROM assets WHERE created_at_utc IS NOT NULL"),
        "recordings": q("SELECT COUNT(*) FROM recordings"),
        "shots": q("SELECT COUNT(*) FROM shots"),
        "surviving": q("SELECT COUNT(*) FROM shots WHERE status <> 'rejected'"),
        "shortlisted": q("SELECT COUNT(*) FROM shots WHERE status = 'shortlisted'"),
        "slots": q("SELECT COUNT(*) FROM timeline"),
        "timeline_s": float(q("SELECT COALESCE(MAX(t_out), 0) FROM timeline")),
    }
    reports = sorted(Path(reports_dir).glob("s0*.json"), key=lambda p: p.stat().st_mtime)
    last_report = None
    if reports:
        try:
            last_report = {"name": reports[-1].name,
                           "finished_utc": json.loads(reports[-1].read_text()).get("finished_utc")}
        except (OSError, ValueError):
            last_report = {"name": reports[-1].name, "finished_utc": None}
    draft = Path(reports_dir).parent / "gates" / "gate3" / "draft.mp4"
    draft_info = None
    if draft.exists():
        st = draft.stat()
        draft_info = {"path": str(draft), "bytes": st.st_size,
                      "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds")}
    tail: list[str] = []
    log = Path(reports_dir) / "remote_jobs.log"
    if log.exists():
        tail = [l for l in log.read_text(errors="replace").splitlines() if l.strip()][-15:]
    return {
        "generated_utc": now.isoformat(timespec="seconds"),
        "stages": stages, "latest_stage": latest, "counts": counts,
        "spend": {"total_usd": round(ledger.total(), 2),
                  "entries": [e.__dict__ for e in ledger.entries[-5:]]},
        "last_report": last_report, "draft": draft_info, "remote_jobs_tail": tail,
    }


def render_html(st: dict[str, Any]) -> str:
    e = html.escape
    rows = lambda pairs: "".join(f"<tr><th>{e(str(k))}</th><td>{e(str(v))}</td></tr>" for k, v in pairs)
    latest = st.get("latest_stage") or {}
    stages = "".join(f"<tr><td>{e(s['stage'])}</td><td>{e(s['unit'])}</td><td>{e(s['status'])}</td>"
                     f"<td>{e(str(s['updated_at']))}</td></tr>" for s in st["stages"])
    spend_rows = "".join(f"<tr><td>{e(x['when'])}</td><td>{e(x['what'])}</td><td>{x['usd']:.2f}</td>"
                         f"<td>{e(x.get('detail', ''))}</td></tr>" for x in st["spend"]["entries"])
    tail = e("\n".join(st["remote_jobs_tail"]))
    draft = st.get("draft")
    draft_txt = (f"{draft['bytes'] / 1e6:.0f} MB, {draft['mtime']}" if draft else "none yet")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>nepal pipeline</title>
<style>body{{font:14px/1.4 system-ui,sans-serif;background:#111;color:#ddd;margin:2rem;max-width:60rem}}
h1,h2{{font-weight:600}}table{{border-collapse:collapse;margin:.5rem 0 1.5rem}}th,td{{text-align:left;padding:.2rem .8rem .2rem 0;vertical-align:top}}
th{{color:#999;font-weight:500}}pre{{background:#1a1a1a;padding:1rem;overflow:auto}}.big{{font-size:1.4rem}}</style></head>
<body><h1>nepal pipeline</h1>
<p class="big">latest: <b>{e(str(latest.get('stage', '-')))} {e(str(latest.get('unit', '-')))}</b> {e(str(latest.get('status', '')))} at {e(str(latest.get('updated_at', '')))}</p>
<p>generated {e(st['generated_utc'])} · draft: {e(draft_txt)} · last report: {e(str((st.get('last_report') or {}).get('name')))} {e(str((st.get('last_report') or {}).get('finished_utc')))}</p>
<h2>the funnel</h2><table>{rows(st['counts'].items())}</table>
<h2>spend</h2><p>total {st['spend']['total_usd']:.2f} USD</p><table>{spend_rows}</table>
<h2>stages</h2><table><tr><th>stage</th><th>unit</th><th>status</th><th>updated</th></tr>{stages}</table>
<h2>last remote run</h2><pre>{tail}</pre>
</body></html>
"""


def write_status(cfg) -> Path:
    from nepal import db
    conn = db.init(cfg.db_path)
    st = build_status(conn, spend.ledger(cfg), cfg.work_root / "reports")
    conn.close()
    out = cfg.workdir("status")
    (out / "status.json").write_text(json.dumps(st, indent=2, default=str))
    (out / "index.html").write_text(render_html(st))
    return out / "index.html"
```

- [ ] **Step 4: Wire it**

`cli.py`: `sub.add_parser("status-page", help="write work/status/{status.json,index.html}")` and `if args.cmd == "status-page": from nepal.cloud import status; print(status.write_status(cfg)); return 0`.

`remote.py::_wrap`: after `({command}); rc=$?;` insert `.venv/bin/nepal status-page >/dev/null 2>&1;` before the push. `jobs-step2.sh`: `$N status-page` before the push line. `sync.PULL`: add `"status"`.

- [ ] **Step 5: Run the tests, commit**

```bash
git add src/nepal/cloud/status.py src/nepal/cli.py src/nepal/cloud/remote.py src/nepal/cloud/sync.py tools/cloud/jobs-step2.sh tests/test_cloud_status.py tests/test_cloud_remote.py
git commit -m "A status page in the bucket, regenerated after every remote run

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: The dashboard, the error alert, and the setup script

**Files:**
- Create: `tools/cloud/dashboard.json`, `tools/cloud/alert-errors.json`, `tools/cloud/monitoring-setup.sh`
- Test: `tests/test_cloud_dashboard_files.py` (extend)

- [ ] **Step 1: The tests**

```python
def test_dashboard_json_has_the_four_panels():
    d = json.loads((CLOUD / "dashboard.json").read_text())
    assert d["displayName"] == "nepal"
    titles = [w["title"] for w in d["mosaicLayout"]["tiles"][0:0]] or \
             [t["widget"]["title"] for t in d["mosaicLayout"]["tiles"]]
    assert {"CPU", "Memory", "Disk", "Pipeline log"} <= set(titles)
    logs = [t["widget"] for t in d["mosaicLayout"]["tiles"] if t["widget"]["title"] == "Pipeline log"][0]
    assert 'logName="projects/nepalvideo/logs/nepal_pipeline"' in logs["logsPanel"]["filter"]


def test_alert_policy_watches_the_error_metric():
    a = json.loads((CLOUD / "alert-errors.json").read_text())
    cond = a["conditions"][0]["conditionThreshold"]
    assert "logging.googleapis.com/user/nepal_errors" in cond["filter"]
    assert cond["comparison"] == "COMPARISON_GT" and cond["thresholdValue"] == 0


def test_setup_script_creates_everything_idempotently():
    s = (CLOUD / "monitoring-setup.sh").read_text()
    for needle in ("logging metrics create nepal_errors", "monitoring channels",
                   "monitoring policies create", "monitoring dashboards create",
                   "dashboards list", "policies list"):
        assert needle in s
```

- [ ] **Step 2: Run to verify they fail**, then **Step 3: the files**

`tools/cloud/dashboard.json` (Cloud Monitoring `Dashboard` resource, mosaic layout, 12 columns):

```json
{
  "displayName": "nepal",
  "mosaicLayout": {
    "columns": 12,
    "tiles": [
      {"xPos": 0, "yPos": 0, "width": 4, "height": 4, "widget": {"title": "CPU",
        "xyChart": {"dataSets": [{"timeSeriesQuery": {"timeSeriesFilter": {
          "filter": "metric.type=\"compute.googleapis.com/instance/cpu/utilization\" resource.type=\"gce_instance\" metadata.user_labels.\"project\"=\"nepal\"",
          "aggregation": {"alignmentPeriod": "60s", "perSeriesAligner": "ALIGN_MEAN"}}}, "plotType": "LINE"}],
          "yAxis": {"scale": "LINEAR"}}}},
      {"xPos": 4, "yPos": 0, "width": 4, "height": 4, "widget": {"title": "Memory",
        "xyChart": {"dataSets": [{"timeSeriesQuery": {"timeSeriesFilter": {
          "filter": "metric.type=\"agent.googleapis.com/memory/percent_used\" resource.type=\"gce_instance\" metric.label.\"state\"=\"used\"",
          "aggregation": {"alignmentPeriod": "60s", "perSeriesAligner": "ALIGN_MEAN"}}}, "plotType": "LINE"}]}}},
      {"xPos": 8, "yPos": 0, "width": 4, "height": 4, "widget": {"title": "Disk",
        "xyChart": {"dataSets": [{"timeSeriesQuery": {"timeSeriesFilter": {
          "filter": "metric.type=\"agent.googleapis.com/disk/percent_used\" resource.type=\"gce_instance\" metric.label.\"state\"=\"used\" metric.label.\"device\"=\"sda1\"",
          "aggregation": {"alignmentPeriod": "60s", "perSeriesAligner": "ALIGN_MEAN"}}}, "plotType": "LINE"}]}}},
      {"xPos": 0, "yPos": 4, "width": 12, "height": 8, "widget": {"title": "Pipeline log",
        "logsPanel": {"filter": "logName=\"projects/nepalvideo/logs/nepal_pipeline\" AND (jsonPayload.logger:\"nepal\" OR textPayload:\"---\" OR severity>=WARNING)",
                      "resourceNames": ["projects/nepalvideo"]}}},
      {"xPos": 0, "yPos": 12, "width": 12, "height": 2, "widget": {"title": "Where else to look",
        "text": {"content": "Status page: https://console.cloud.google.com/storage/browser/_details/nepalvideo-29922345852/work/status/index.html  ·  Spend ledger: work/reports/spend.json in the bucket  ·  Billing: https://console.cloud.google.com/billing", "format": "MARKDOWN"}}}
    ]
  }
}
```

`tools/cloud/alert-errors.json`:

```json
{
  "displayName": "nepal pipeline error",
  "combiner": "OR",
  "conditions": [{"displayName": "an ERROR or a Traceback in the stage logs",
    "conditionThreshold": {
      "filter": "metric.type=\"logging.googleapis.com/user/nepal_errors\" resource.type=\"gce_instance\"",
      "comparison": "COMPARISON_GT", "thresholdValue": 0, "duration": "0s",
      "aggregations": [{"alignmentPeriod": "300s", "perSeriesAligner": "ALIGN_SUM"}]}}],
  "alertStrategy": {"autoClose": "1800s"}
}
```

`tools/cloud/monitoring-setup.sh`:

```bash
#!/bin/bash
# Creates the error metric, the email channel, the alert and the dashboard.
# Idempotent: each is looked up by name first. Run from the operator's
# machine with a live gcloud login; nothing here computes.
set -uo pipefail
G=${GCLOUD:-$HOME/google-cloud-sdk/bin/gcloud}
P=nepalvideo
EMAIL=${NEPAL_ALERT_EMAIL:-kirill.keller.tech@gmail.com}
HERE=$(cd "$(dirname "$0")" && pwd)

if ! $G logging metrics describe nepal_errors --project $P >/dev/null 2>&1; then
  $G logging metrics create nepal_errors --project $P \
    --description "ERROR or Traceback lines in the pipeline's stage logs" \
    --log-filter 'logName="projects/'$P'/logs/nepal_pipeline" AND (severity>=ERROR OR textPayload:"Traceback" OR jsonPayload.message:"Traceback")'
fi
CH=$($G beta monitoring channels list --project $P --filter "displayName='nepal email'" --format 'value(name)' | head -1)
if [ -z "$CH" ]; then
  CH=$($G beta monitoring channels create --project $P --display-name 'nepal email' \
       --type email --channel-labels "email_address=$EMAIL" --format 'value(name)')
fi
if [ -z "$($G alpha monitoring policies list --project $P --filter "displayName='nepal pipeline error'" --format 'value(name)' | head -1)" ]; then
  $G alpha monitoring policies create --project $P --policy-from-file "$HERE/alert-errors.json" \
     --notification-channels "$CH"
fi
if [ -z "$($G monitoring dashboards list --project $P --filter "displayName='nepal'" --format 'value(name)' | head -1)" ]; then
  $G monitoring dashboards create --project $P --config-from-file "$HERE/dashboard.json"
fi
echo "dashboard: https://console.cloud.google.com/monitoring/dashboards?project=$P"
```

- [ ] **Step 4: Run the tests; run the setup script from a copy under `$CLAUDE_JOB_DIR/tmp` (the worktree guard); open the dashboard URL and confirm the four panels show data once the box has run for a few minutes; commit.**

```bash
git add tools/cloud/dashboard.json tools/cloud/alert-errors.json tools/cloud/monitoring-setup.sh tests/test_cloud_dashboard_files.py
git commit -m "A Cloud Monitoring dashboard and an error alert for the pipeline

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Docs

README "Running it in the cloud" gains a short "Watching it" subsection with the dashboard URL, the status page location, and the alert; STATE.md's cloud section links both. Commit.

## Self-review

Coverage: every element of the approved proposal has a task (running state and resources → dashboard charts via the agent's host metrics; stage → logs panel and the status page's `latest_stage`; errors → metric plus alert plus email channel; spend → status page and dashboard text link). Types: `status.build_status` keys match both the tests and `render_html`; the log name `nepal_pipeline` matches between the agent receiver, the dashboard filter and the metric filter.
