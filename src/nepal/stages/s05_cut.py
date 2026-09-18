"""S05-S07 driven end to end: score, assemble, render.

One driver rather than three, because the three are one question -- what is
the film -- and the operator iterates on the whole answer. The weights move,
the cut is rebuilt, the draft is watched; splitting that across three commands
would make the loop three times as long for no gain.

The Bedrock halves are deliberately absent: S04.3's captions feed one term of
``score_sem``, and S06's ordering refinement is a polish pass over a list that
is already ordered. Neither is required to produce a cut, and S05's
missing-term rule means the ranking is correct without them rather than
merely usable.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from nepal import db, freshness
from nepal.config import Config
from nepal.process import assemble as asm, render as render_mod, score as score_mod
from nepal.process import timeline_io
from nepal.util.progress import Progress

log = logging.getLogger(__name__)
STAGE = "S05"


def _load_embeddings(cfg: Config) -> dict[str, np.ndarray]:
    path = cfg.work_root / "semantic" / "clip.npy"
    idx = cfg.work_root / "semantic" / "clip_index.json"
    if not path.exists() or not idx.exists():
        log.info("S06 no CLIP embeddings yet; MMR will fall back to score order "
                 "(diversity unenforced, which shows as repetition in the cut)")
        return {}
    E = np.load(path)
    ids = json.loads(idx.read_text())["shot_ids"]
    return {sid: E[i] for i, sid in enumerate(ids) if i < len(E)}


def _message_proximity(conn, when: str | None, *, window_s: float = 1200.0) -> float:
    if not when:
        return 0.0
    row = conn.execute(
        "SELECT MIN(ABS(strftime('%s', ts_utc) - strftime('%s', ?))) "
        "FROM messages WHERE ts_utc IS NOT NULL", (when,)).fetchone()
    if not row or row[0] is None:
        return 0.0
    return float(max(0.0, 1.0 - float(row[0]) / window_s))


def score_shots(cfg: Config, conn) -> dict[str, Any]:
    """S05 -- score every surviving shot, then shortlist."""
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM shots WHERE status IN ('candidate','shortlisted') "
        "ORDER BY COALESCE(start_utc, ''), shot_id")]
    if not rows:
        return {"n_shots": 0, "note": "nothing survived the gate"}

    w = cfg.get("score")
    peak = cfg.get("score.duration_fit_peak_s", [4.0, 8.0])
    ranks = score_mod.percentile_rank([r.get("sharpness") for r in rows])

    # "first at a place" and "a new altitude record" are facts about the
    # sequence, not the shot, so they are computed in chronological order once.
    seen_places: set[str] = set()
    max_alt = float("-inf")
    for i, r in enumerate(rows):
        place = r.get("place_name")
        first_here = bool(place) and place not in seen_places
        if place:
            seen_places.add(place)
        alt = r.get("alt_dem_m")
        new_alt = alt is not None and float(alt) > max_alt
        if alt is not None:
            max_alt = max(max_alt, float(alt))
        r["_tech"] = score_mod.score_tech(r, sharpness_rank=ranks.get(i),
                                          weights=w["tech"], peak=peak)
        r["_sem"] = score_mod.score_sem(r, clip_similarity=None, weights=w["sem"])
        r["_ctx"] = score_mod.score_ctx(
            r, new_max_alt=new_alt, first_at_place=first_here,
            msg_proximity=_message_proximity(conn, r.get("start_utc")),
            weights=w["ctx"])
        r["score_total"] = score_mod.combine(r["_tech"], r["_sem"], r["_ctx"], w["total"])

    conn.executemany(
        "UPDATE shots SET score_tech=?, score_sem=?, score_ctx=?, score_total=? "
        "WHERE shot_id=?",
        [(r["_tech"], r["_sem"], r["_ctx"], r["score_total"], r["shot_id"]) for r in rows])

    picked = set(score_mod.shortlist(rows, size=int(cfg.get("score.shortlist_size", 400)),
                                     min_per_act=int(cfg.get("score.min_per_act", 40))))
    conn.execute("UPDATE shots SET status='candidate' WHERE status='shortlisted'")
    conn.executemany("UPDATE shots SET status='shortlisted' WHERE shot_id=?",
                     [(s,) for s in picked])
    conn.commit()

    by_act: dict[Any, int] = {}
    for r in rows:
        if r["shot_id"] in picked:
            by_act[r.get("act")] = by_act.get(r.get("act"), 0) + 1
    totals = [r["score_total"] for r in rows if r["score_total"] is not None]
    log.info("S05 scored %d shot(s); shortlisted %d; per act %s", len(rows), len(picked),
             {k: by_act[k] for k in sorted(by_act, key=lambda x: (x is None, x))})
    if totals:
        log.info("S05 score_total %s", {k: round(float(v), 3) for k, v in
                 zip(("p5", "p50", "p95"), np.percentile(totals, [5, 50, 95]))})
    n_sem = sum(1 for r in rows if r["_sem"] is not None)
    log.info("S05 %d of %d shot(s) have a semantic score (captions are a "
             "Bedrock stage; the rest are ranked on what is known)", n_sem, len(rows))
    return {"n_shots": len(rows), "n_shortlisted": len(picked),
            "per_act": {str(k): v for k, v in by_act.items()}, "n_with_sem": n_sem}


def build_timeline(cfg: Config, conn) -> dict[str, Any]:
    """S06 -- choose and order the shots, then write the timeline."""
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM shots WHERE status='shortlisted'")]
    if not rows:
        return {"n_slots": 0, "note": "nothing shortlisted"}

    bounds = json.loads(db.get_decision(conn, "act_boundaries") or "[]")
    music = json.loads((cfg.work_root / "music" / "music_map.json").read_text()) \
        if (cfg.work_root / "music" / "music_map.json").exists() else {"acts": []}
    act_len = {int(a["act"]): float(a["t_end"]) - float(a["t_start"])
               for a in music.get("acts", [])}
    embeddings = _load_embeddings(cfg)
    lam = float(cfg.get("assemble.mmr_lambda", 0.3))
    place_cap = int(cfg.get("assemble.max_shots_per_place_per_act", 3))
    durations = cfg.get("assemble.shot_duration_s")

    beats_by_track: dict[str, list[float]] = {}
    downbeats_by_track: dict[str, list[float]] = {}
    for r in conn.execute("SELECT track_id, t_s, is_downbeat FROM beats ORDER BY t_s"):
        beats_by_track.setdefault(r["track_id"], []).append(float(r["t_s"]))
        if r["is_downbeat"]:
            downbeats_by_track.setdefault(r["track_id"], []).append(float(r["t_s"]))

    timeline: list[dict[str, Any]] = []
    t = 0.0
    per_act: dict[int, int] = {}
    for act in sorted({int(r["act"]) for r in rows if r["act"] is not None}):
        act_rows = [r for r in rows if r.get("act") == act]
        seconds = act_len.get(act, 240.0)
        rng = durations.get(act) or durations.get(str(act)) or [4.0, 6.0]
        budget = asm.slot_budget(seconds, rng)
        track = next((a.get("track_id") for a in music.get("acts", [])
                      if int(a.get("act", -1)) == act), None)
        beats = beats_by_track.get(track or "", [])
        downs = downbeats_by_track.get(track or "", [])

        def admissible(c, chosen, _cap=place_cap):
            return asm.place_count_ok(c, chosen, limit=_cap)

        # Speech is no longer a claim on a slot here: the beat sheet (S04.5)
        # chooses the voice by reading, and assembly v2 builds around it.
        # Until that lands, the fill is by score alone, ties by id.
        by_score = sorted(act_rows, key=lambda r: (-float(r.get("score_total") or 0.0),
                                                   str(r.get("shot_id"))))
        picked = asm.mmr_select(by_score, budget=budget,
                                embeddings=embeddings, lam=lam, admissible=admissible)
        ordered = asm.chronological(picked)
        laid = asm.lay_out(ordered, start_s=t, duration_range=rng,
                           beats=beats, downbeats=downs)
        timeline.extend(laid)
        per_act[act] = len(laid)
        t = laid[-1]["t_out"] if laid else t

    conn.execute("DELETE FROM timeline")
    conn.executemany(
        "INSERT INTO timeline(slot_index, act, shot_id, t_in, t_out, src_in, src_out, yaw) "
        "VALUES (?,?,?,?,?,?,?,?)",
        [(i, r["act"], r["shot_id"], r["t_in"], r["t_out"], r["src_in"], r["src_out"],
          r.get("yaw")) for i, r in enumerate(timeline)])
    conn.commit()

    paths = timeline_io.write(timeline, cfg.workdir("."),
                              media_dir=str(cfg.work_root / "proxies"))
    log.info("S06 %d slot(s), %.1f min; per act %s", len(timeline), t / 60, per_act)
    log.info("S06 wrote %s and %s", paths["otio"].name, paths["fcpxml"].name)
    if not embeddings:
        log.warning("S06 built without CLIP embeddings, so MMR could not "
                    "penalise repetition -- expect similar shots near each other")
    return {"n_slots": len(timeline), "duration_s": round(t, 1),
            "per_act": {str(k): v for k, v in per_act.items()},
            "had_embeddings": bool(embeddings)}


def photo_source(cfg: Config, row: Mapping[str, Any]) -> Path | None:
    """What the renderer reads for a photograph slot.

    The JPEG S03.0 wrote while decoding, when it exists; the original only
    as a fallback. An iPhone still is HEIC and whether ffmpeg opens HEIC is a
    property of its build, not of the file -- the box's Ubuntu ffmpeg 4.4
    refused every one of them, and the draft died on the first.
    """
    still = cfg.work_root / "stills" / f"{row['shot_id']}.jpg"
    if still.exists():
        return still
    key = row.get("s3_key")
    return cfg.data_root / str(key).replace("raw/", "") if key else None


def render_draft(cfg: Config, conn) -> dict[str, Any]:
    """S07 -- render the cut to a watchable file."""
    rows = [dict(r) for r in conn.execute(
        "SELECT t.*, s.recording_id, s.media_kind, a.s3_key FROM timeline t "
        "JOIN shots s ON s.shot_id = t.shot_id "
        "LEFT JOIN assets a ON a.asset_id = s.asset_id ORDER BY t.slot_index")]
    if not rows:
        return {"skipped": "no timeline"}
    proxies = cfg.work_root / "proxies"
    sources: dict[str, Any] = {}
    usable: list[dict[str, Any]] = []
    missing: dict[str, int] = {}
    for r in rows:
        if r["media_kind"] == "photo":
            p = photo_source(cfg, r)
        else:
            p = proxies / f"{r['recording_id']}_eq.mp4"
        if p is not None and p.exists():
            sources[r["shot_id"]] = p
            usable.append(r)
        else:
            missing[r["media_kind"] or "?"] = missing.get(r["media_kind"] or "?", 0) + 1
    if not usable:
        return {"skipped": "no source media for any slot"}
    if missing:
        # Loudly: a slot that silently vanishes shortens the film, and the
        # first draft lost 94 photographs -- half its running time -- exactly
        # this way.
        log.warning("S07 %d slot(s) have no source media and are omitted, "
                    "which shortens the cut: %s", sum(missing.values()), missing)

    out = cfg.workdir("gates", "gate3") / "draft.mp4"
    d = cfg.get("render.draft")
    cmd = render_mod.build_command(
        usable, sources=sources, out_path=out,
        width=int(d["width"]), height=int(d["height"]), crf=int(d["crf"]),
        fps=int(cfg.get("render.fps", render_mod.DRAFT_FPS)))
    log.info("S07 rendering %d of %d slot(s) to %s", len(usable), len(rows), out)
    log.debug("S07 %s", render_mod.describe(cmd))
    import subprocess
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        log.error("S07 render failed: %s", (proc.stderr or "")[-600:])
        return {"error": (proc.stderr or "")[-600:], "n_slots": len(usable)}
    size = out.stat().st_size if out.exists() else 0
    log.info("S07 draft written: %s (%.1f MB)", out, size / 1e6)
    return {"path": str(out), "bytes": size, "n_slots": len(usable),
            "n_skipped": len(rows) - len(usable)}


def run(cfg: Config, *, force: bool = False,
        redo: set[str] | None = None) -> dict[str, Any]:
    conn = db.init(cfg.db_path)
    report: dict[str, Any] = {"stage": STAGE, "started_utc": db.utcnow()}
    report["skipped_stale"] = freshness.warn_if_stale(
        log, conn, STAGE, force=force, rerun_hint="nepal cut --force")
    # All three steps are cheap relative to S03 and they depend on weights the
    # operator is actively tuning, so they always re-run.
    for name, fn in (("score", lambda: score_shots(cfg, conn)),
                     ("timeline", lambda: build_timeline(cfg, conn)),
                     ("draft", lambda: render_draft(cfg, conn))):
        if redo and name not in redo:
            continue
        report[name] = fn()
        db.mark_unit(conn, STAGE, name, detail=json.dumps(report[name], default=str)[:2000])
    report["finished_utc"] = db.utcnow()
    cfg.work("reports", "s05_cut.json").write_text(json.dumps(report, indent=2, default=str))
    conn.close()
    return report
