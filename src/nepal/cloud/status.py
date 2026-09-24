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

from nepal import db
from nepal.cloud import spend

# The query is db's (S06 reports the same table); the name stays here for
# the status document's readers.
per_act_sources = db.per_act_sources


def _spend(ledger: spend.Ledger, state_path: Path, now: datetime) -> dict[str, Any]:
    """Booked, running, and the sum -- the number the ceiling is about.

    VM hours are booked at `remote down`; a box that is up has hours no
    entry holds. `remote up` stamps its start and its price into
    remote_state.json and `down` clears them, so an `up_since` still in
    that file is a meter running. Overstating while the file is stale is
    the safe direction for a ceiling: nepal-cpu's 104 idle hours were
    invisible here for four days.
    """
    running, since = 0.0, None
    try:
        state = json.loads(state_path.read_text())
    except (OSError, ValueError):
        state = {}
    for prof in (state or {}).values():
        start = (prof or {}).get("up_since")
        if not start:
            continue
        hours = max(0.0, (now - datetime.fromisoformat(start)).total_seconds() / 3600)
        running += hours * float(prof.get("usd_per_h", 0.0))
        since = min(since, start) if since else start
    return {"booked_usd": round(ledger.total(), 2), "running_usd": round(running, 2),
            "running_since": since, "total_usd": round(ledger.total() + running, 2)}


def build_status(conn, ledger: spend.Ledger, reports_dir: Path, *,
                 now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)

    def q(sql: str):
        return conn.execute(sql).fetchone()[0]

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
                      "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc)
                      .isoformat(timespec="seconds")}
    tail: list[str] = []
    log = Path(reports_dir) / "remote_jobs.log"
    if log.exists():
        tail = [l for l in log.read_text(errors="replace").splitlines() if l.strip()][-15:]
    return {
        "generated_utc": now.isoformat(timespec="seconds"),
        "stages": stages, "latest_stage": latest, "counts": counts,
        "per_act_sources": per_act_sources(conn),
        "spend": {**_spend(ledger, Path(reports_dir) / "remote_state.json", now),
                  "entries": [e.__dict__ for e in ledger.entries[-5:]]},
        "last_report": last_report, "draft": draft_info, "remote_jobs_tail": tail,
    }


def render_html(st: dict[str, Any]) -> str:
    e = html.escape

    def rows(pairs) -> str:
        return "".join(f"<tr><th>{e(str(k))}</th><td>{e(str(v))}</td></tr>" for k, v in pairs)

    latest = st.get("latest_stage") or {}
    stages = "".join(
        f"<tr><td>{e(s['stage'])}</td><td>{e(s['unit'])}</td><td>{e(s['status'])}</td>"
        f"<td>{e(str(s['updated_at']))}</td></tr>" for s in st["stages"])
    spend_rows = "".join(
        f"<tr><td>{e(x['when'])}</td><td>{e(x['what'])}</td><td>{x['usd']:.2f}</td>"
        f"<td>{e(x.get('detail', ''))}</td></tr>" for x in st["spend"]["entries"])
    tail = e("\n".join(st["remote_jobs_tail"]))
    source_rows = []
    for act, sources in sorted(st.get("per_act_sources", {}).items(), key=lambda kv: int(kv[0])):
        total = sum(v["slots"] for v in sources.values())
        for src, v in sorted(sources.items()):
            share = f"{v['slots'] / total * 100:.0f}%" if total else "-"
            source_rows.append(f"<tr><td>{e(act)}</td><td>{e(src)}</td><td>{v['slots']}</td>"
                               f"<td>{v['available']}</td><td>{share}</td></tr>")
    draft = st.get("draft")
    draft_txt = f"{draft['bytes'] / 1e6:.0f} MB, {draft['mtime']}" if draft else "none yet"
    rep = st.get("last_report") or {}
    rep_txt = f"{rep.get('name')} {rep.get('finished_utc')}"
    sp = st["spend"]
    spend_txt = f"total {sp['total_usd']:.2f} USD"
    if sp.get("running_usd"):
        spend_txt = (f"booked {sp['booked_usd']:.2f} + running {sp['running_usd']:.2f} "
                     f"since {sp['running_since']} = {sp['total_usd']:.2f} USD")
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8"><title>nepal pipeline</title>\n'
        "<style>body{font:14px/1.4 system-ui,sans-serif;background:#111;color:#ddd;"
        "margin:2rem;max-width:60rem}h1,h2{font-weight:600}table{border-collapse:collapse;"
        "margin:.5rem 0 1.5rem}th,td{text-align:left;padding:.2rem .8rem .2rem 0;"
        "vertical-align:top}th{color:#999;font-weight:500}pre{background:#1a1a1a;"
        "padding:1rem;overflow:auto}.big{font-size:1.4rem}</style></head>\n"
        "<body><h1>nepal pipeline</h1>\n"
        f'<p class="big">latest: <b>{e(str(latest.get("stage", "-")))} '
        f'{e(str(latest.get("unit", "-")))}</b> {e(str(latest.get("status", "")))} '
        f'at {e(str(latest.get("updated_at", "")))}</p>\n'
        f"<p>generated {e(st['generated_utc'])} · draft: {e(draft_txt)} · last report: "
        f"{e(rep_txt)}</p>\n"
        f"<h2>the funnel</h2><table>{rows(st['counts'].items())}</table>\n"
        f"<h2>spend</h2><p>{e(spend_txt)}</p><table>{spend_rows}</table>\n"
        f"<h2>sources per act</h2><table><tr><th>act</th><th>source</th><th>slots</th>"
        f"<th>available</th><th>share</th></tr>{''.join(source_rows)}</table>\n"
        f"<h2>stages</h2><table><tr><th>stage</th><th>unit</th><th>status</th><th>updated</th></tr>"
        f"{stages}</table>\n"
        f"<h2>last remote run</h2><pre>{tail}</pre>\n"
        "</body></html>\n")


def write_status(cfg) -> Path:
    conn = db.init(cfg.db_path)
    st = build_status(conn, spend.ledger(cfg), cfg.work_root / "reports")
    conn.close()
    out = cfg.workdir("status")
    (out / "status.json").write_text(json.dumps(st, indent=2, default=str))
    (out / "index.html").write_text(render_html(st))
    return out / "index.html"
