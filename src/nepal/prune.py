"""Remove what the pipeline no longer produces.

``db.upsert`` never removes. A recording produced by an older grouping rule,
a proxy for a file that has since been deleted, a shot of either: all of it
survives every re-run and stays eligible for selection. On this corpus the
first draft cut carried nine slots from ``phone_kulikov_IMG`` -- 302 phone
clips the old rule had merged into one 26-minute "recording", timestamped
from the first clip, so Larke Pass footage landed in the Planning act.

The plan is pure: given the rows and the files, decide what goes. Execution
is a handful of DELETEs and unlinks. The grouping is re-derived from the
assets already in the database, so this never re-probes anything.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from nepal import db
from nepal.probe import chapters

log = logging.getLogger(__name__)


@dataclass
class PrunePlan:
    recordings: list[str] = field(default_factory=list)
    shots: list[str] = field(default_factory=list)
    files: list[Path] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not (self.recordings or self.shots or self.files)


def plan_prune(*, recordings: Sequence[Mapping[str, Any]],
               assets: Sequence[Mapping[str, Any]],
               shots: Sequence[Mapping[str, Any]],
               work_root: Path) -> PrunePlan:
    """What the current grouping and the current files do not account for."""
    rows = []
    for a in assets:
        a = dict(a)
        a["filename"] = Path(a["s3_key"]).name
        rows.append(a)
    current = {r.recording_id for r in chapters.group_recordings(rows)}
    owned = {a.get("recording_id") for a in assets if a.get("recording_id")}

    plan = PrunePlan()
    for r in recordings:
        rid = str(r["recording_id"])
        if rid not in current:
            plan.reasons[rid] = "not produced by the current grouping rule"
        elif rid not in owned:
            plan.reasons[rid] = "owns no asset"
        else:
            continue
        plan.recordings.append(rid)
    stale = set(plan.recordings)
    plan.shots = [str(s["shot_id"]) for s in shots if s.get("recording_id") in stale]

    transcripts = work_root / "transcripts" / "shots"
    for rid in plan.recordings:
        for p in (work_root / "proxies" / f"{rid}_eq.mp4",
                  work_root / "audio" / f"{rid}.wav"):
            if p.exists():
                plan.files.append(p)
        if transcripts.is_dir():
            plan.files.extend(sorted(transcripts.glob(f"{rid}_*.json")))

    # A proxy for a recording the database does not know at all is the same
    # kind of leftover: the file was deleted from data_root, the manifest
    # dropped the asset, and the 230 MB proxy stayed.
    known = {str(r["recording_id"]) for r in recordings}
    proxies = work_root / "proxies"
    if proxies.is_dir():
        for p in sorted(proxies.glob("*_eq.mp4")):
            rid = p.name[:-len("_eq.mp4")]
            if rid not in known and rid not in current:
                plan.files.append(p)
                plan.reasons[rid] = "proxy for a recording the database does not hold"
    return plan


def execute(conn, plan: PrunePlan, *, dry_run: bool = False) -> dict[str, Any]:
    """Apply the plan. Rows first (children before parents), then files."""
    size = sum(p.stat().st_size for p in plan.files if p.exists())
    if not dry_run and not plan.is_empty:
        conn.executemany("DELETE FROM timeline WHERE shot_id=?", [(s,) for s in plan.shots])
        conn.executemany("DELETE FROM shots WHERE shot_id=?", [(s,) for s in plan.shots])
        conn.executemany("DELETE FROM recordings WHERE recording_id=?",
                         [(r,) for r in plan.recordings])
        conn.executemany("DELETE FROM stage_units WHERE unit_id=? OR unit_id=?",
                         [(r, f"proxy:{r}") for r in plan.recordings])
        conn.commit()
        for p in plan.files:
            try:
                p.unlink()
            except OSError as exc:
                log.warning("prune: could not remove %s: %s", p, exc)
    for rid in plan.recordings:
        log.info("prune: recording %s -- %s", rid, plan.reasons.get(rid, ""))
    return {"dry_run": dry_run, "n_recordings": len(plan.recordings),
            "n_shots": len(plan.shots), "n_files": len(plan.files), "bytes": size,
            "recordings": plan.recordings, "reasons": plan.reasons,
            "files": [str(p) for p in plan.files]}


def run(cfg, *, dry_run: bool = False) -> dict[str, Any]:
    conn = db.init(cfg.db_path)
    recordings = [dict(r) for r in conn.execute("SELECT * FROM recordings")]
    assets = [dict(r) for r in conn.execute(
        "SELECT asset_id, s3_key, source, kind, container, duration_s, "
        "created_at, created_at_utc, recording_id FROM assets")]
    shots = [dict(r) for r in conn.execute(
        "SELECT shot_id, recording_id, asset_id FROM shots")]
    plan = plan_prune(recordings=recordings, assets=assets, shots=shots,
                      work_root=cfg.work_root)
    report = execute(conn, plan, dry_run=dry_run)
    conn.close()
    log.info("prune%s: %d recording(s), %d shot(s), %d file(s), %.0f MB",
             " (dry run)" if dry_run else "", report["n_recordings"],
             report["n_shots"], report["n_files"], report["bytes"] / 1e6)
    return report
