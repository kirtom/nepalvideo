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


SOURCE_JOIN = ("LEFT JOIN recordings rc ON rc.recording_id = s.recording_id "
               "LEFT JOIN assets a ON a.asset_id = s.asset_id")


def per_act_sources(conn) -> dict[str, dict[str, dict[str, int]]]:
    """Per act, per source: how many slots the cut gives it against how many
    surviving shots it had -- the number the operator asked to see, so a
    phone that filmed half the day and got a tenth of the screen is visible
    rather than felt. The source is the recording's for video and the
    asset's for a photo, the same way S06 reads it."""
    out: dict[str, dict[str, dict[str, int]]] = {}
    for r in conn.execute(
            "SELECT s.act, COALESCE(rc.source, a.source) AS source, COUNT(*) AS n FROM shots s "
            f"{SOURCE_JOIN} WHERE s.status <> 'rejected' AND s.act IS NOT NULL GROUP BY 1, 2"):
        out.setdefault(str(r["act"]), {})[str(r["source"])] = {"slots": 0, "available": int(r["n"])}
    for r in conn.execute(
            "SELECT t.act, COALESCE(rc.source, a.source) AS source, COUNT(*) AS n FROM timeline t "
            f"JOIN shots s ON s.shot_id = t.shot_id {SOURCE_JOIN} GROUP BY 1, 2"):
        cell = out.setdefault(str(r["act"]), {}).setdefault(str(r["source"]), {"slots": 0, "available": 0})
        cell["slots"] = int(r["n"])
    return out


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
        "spend": {"total_usd": round(ledger.total(), 2),
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
        f"{e(str(rep.get('name')))} {e(str(rep.get('finished_utc')))}</p>\n"
        f"<h2>the funnel</h2><table>{rows(st['counts'].items())}</table>\n"
        f"<h2>spend</h2><p>total {st['spend']['total_usd']:.2f} USD</p><table>{spend_rows}</table>\n"
        f"<h2>sources per act</h2><table><tr><th>act</th><th>source</th><th>slots</th>"
        f"<th>available</th><th>share</th></tr>{''.join(source_rows)}</table>\n"
        f"<h2>stages</h2><table><tr><th>stage</th><th>unit</th><th>status</th><th>updated</th></tr>"
        f"{stages}</table>\n"
        f"<h2>last remote run</h2><pre>{tail}</pre>\n"
        "</body></html>\n")


def write_status(cfg) -> Path:
    from nepal import db
    conn = db.init(cfg.db_path)
    st = build_status(conn, spend.ledger(cfg), cfg.work_root / "reports")
    conn.close()
    out = cfg.workdir("status")
    (out / "status.json").write_text(json.dumps(st, indent=2, default=str))
    (out / "index.html").write_text(render_html(st))
    return out / "index.html"
