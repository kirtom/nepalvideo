"""S03 -- per-clip processing.

S03.0 (this file, for now) turns photographs into shots so they can reach the
timeline at all. The rest of S03 -- proxy reprojection, scene detection,
technical metrics, the quality gate -- follows.

Photo shots are cheap: no ffmpeg, no reprojection, one decode per file. They
are built here rather than in S01 because they are shots, and S01's job is to
say what was delivered, not to decide what might end up in the film.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from nepal import db, freshness
from nepal.config import Config
from nepal.process import reproject, shots as shots_mod, stills
from nepal.spine import acts as acts_mod, gps as gps_mod
from nepal.util import proc
from nepal.util.progress import Progress, heartbeat

log = logging.getLogger(__name__)
STAGE = "S03"


def _dt(value: Any):
    from datetime import datetime, timezone
    if not value:
        return None
    d = datetime.fromisoformat(str(value))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def build_proxies(cfg: Config, conn, *, force: bool = False,
                  yaw_videos: bool | None = None) -> dict[str, Any]:
    """S03.1 -- one ffmpeg pass per recording: equirect proxy plus 16 kHz audio.

    Resumable per recording, not per stage. This is the longest step in the
    pipeline -- 4.9 hours of 5.7K footage -- and losing an hour of it to an
    interruption near the end is the difference between a pipeline you re-run
    and one you avoid re-running.

    Phase A by default: proxy and audio only, no yaw view videos. Measured at
    3.46x realtime against 0.73x for the full single-pass graph, because the
    four rectilinear branches each re-resample the whole frame. S05 needs one
    yaw per surviving shot, not four per recording, so those are rendered as
    stills later against far fewer frames.
    """
    fov_deg = db.get_decision_float(conn, "fov_deg") or float(
        cfg.get("probe.fov.fallback_deg"))
    if yaw_videos is None:
        yaw_videos = bool(cfg.get("process.yaw_videos", False))

    recs = [dict(r) for r in conn.execute(
        "SELECT recording_id, source, is_360, duration_s FROM recordings "
        "ORDER BY start_utc")]
    if not recs:
        return {"n_recordings": 0, "error": "no recordings -- run S01 first"}

    assets_by_rec: dict[str, list[dict]] = {}
    for a in conn.execute(
            "SELECT recording_id, s3_key, container, kind, chapter_index, "
            "width, height, frame_shape FROM assets WHERE recording_id IS NOT NULL"):
        assets_by_rec.setdefault(a["recording_id"], []).append(dict(a))

    work = cfg.work_root
    done = {u for u in db.done_units(conn, STAGE) if u.startswith("proxy:")}
    hwaccel = reproject.detect_hwaccel()
    encoder = reproject.detect_encoder()
    log.info("S03.1 %d recording(s), hwaccel=%s encoder=%s, yaw videos=%s",
             len(recs), hwaccel or "none", encoder, yaw_videos)

    built = skipped = failed = 0
    modes: dict[str, int] = {}
    proxy_fps = float(cfg.get("process.proxy_fps", 0)) or None
    preset = str(cfg.get("process.encoder_preset", "veryfast")) or None
    eq = cfg.get("process.proxy_size", [1024, 512])
    eq_size = (int(eq[0]), int(eq[1]))
    flat = cfg.get("process.flat_proxy_size", [960, 540])
    proxy_w, proxy_h = int(flat[0]), int(flat[1])
    total_s = sum(float(r["duration_s"] or 0) for r in recs)
    bar = Progress("S03.1 reprojecting", len(recs))
    for r in recs:
        rid = r["recording_id"]
        bar.step(note=rid[-24:])
        unit = f"proxy:{rid}"
        picked = reproject.pick_sources(assets_by_rec.get(rid, []), cfg.data_root)
        if picked is None:
            log.warning("S03.1 %s: no usable source file", rid)
            failed += 1
            continue
        paths, mode = picked
        modes[mode] = modes.get(mode, 0) + 1

        extra: list[str] = []
        inputs = paths
        if mode == "lens_pair":
            pass                       # two -i, stacked by the graph
        elif len(paths) > 1:
            # True chapters of one take: concat so the joins produce no cut.
            src = reproject.concat_list(paths, work / "concat" / f"{rid}.ffconcat")
            inputs = [src]
            extra = ["-f", "concat", "-safe", "0"]
        # A source already no larger than the proxy is remuxed, not re-encoded:
        # 55 minutes of this corpus are .lrv files the camera wrote as proxies.
        chosen = {str(a.get("s3_key") or "").rsplit("/", 1)[-1]: a
                  for a in assets_by_rec.get(rid, [])}
        first = chosen.get(paths[0].name) or {}
        w, h = first.get("width") or 0, first.get("height") or 0
        passthrough = (mode == "flat" and 0 < w <= proxy_w and 0 < h <= proxy_h)
        p = reproject.plan_for_mode(paths, rid, work, mode=mode, fov_deg=fov_deg,
                                    proxy_size=eq_size,
                                    view_size=(proxy_w, proxy_h),
                                    passthrough=passthrough)
        if not force and unit in done and p.proxy_path.exists():
            skipped += 1
            continue

        # Say what was chosen before the work starts. Three rounds of "it is
        # stuck on this file" were spent guessing which source and which filter
        # graph a recording had picked; the log should simply say.
        dur = float(r["duration_s"] or 0)
        log.info("S03.1 %s: %s%s, %s, %.0fs", rid, mode,
                 " (remux)" if passthrough else "",
                 ", ".join(x.name for x in inputs), dur)
        started = time.monotonic()
        # A pass that runs many times longer than its own footage is wedged, not
        # slow. Bounded so one bad file reports itself instead of holding the
        # whole run; the rest of the corpus still gets processed.
        budget = max(float(cfg.get("process.ffmpeg_min_timeout_s", 300)),
                     dur * float(cfg.get("process.ffmpeg_timeout_factor", 20)))
        cmd = reproject.build_command(p, hwaccel=hwaccel, encoder=encoder,
                                      has_audio=True, extra_input=extra,
                                      inputs=inputs, fps=proxy_fps,
                                      preset=preset)
        try:
            proc.run(cmd, check=True, timeout=budget)
        except subprocess.TimeoutExpired:
            log.error("S03.1 %s: gave up after %.0fs on %.0fs of footage. "
                      "The command was: %s", rid, budget, dur, " ".join(cmd))
            db.mark_unit(conn, STAGE, unit, status="failed", detail="timeout")
            failed += 1
            continue
        except (proc.ToolFailed, proc.ToolMissing) as exc:
            # A missing audio stream is the common case, not a real failure.
            log.debug("S03.1 %s with audio failed (%s); retrying video only", rid, exc)
            try:
                proc.run(reproject.build_command(p, hwaccel=hwaccel, encoder=encoder,
                                                 has_audio=False, extra_input=extra,
                                                 inputs=inputs, fps=proxy_fps,
                                                 preset=preset),
                         check=True, timeout=budget)
            except subprocess.TimeoutExpired:
                log.error("S03.1 %s: gave up after %.0fs on %.0fs of footage",
                          rid, budget, dur)
                db.mark_unit(conn, STAGE, unit, status="failed", detail="timeout")
                failed += 1
                continue
            except (proc.ToolFailed, proc.ToolMissing) as exc2:
                log.warning("S03.1 %s failed: %s", rid, exc2)
                db.mark_unit(conn, STAGE, unit, status="failed", detail=str(exc2)[:500])
                failed += 1
                continue
        took = time.monotonic() - started
        log.info("S03.1 %s: done in %.0fs (%.2fx realtime)", rid, took,
                 (dur / took) if took > 0 else 0.0)
        db.mark_unit(conn, STAGE, unit, detail=json.dumps(
            {"proxy": str(p.proxy_path), "audio": str(p.audio_path),
             "sources": [str(x) for x in inputs], "mode": mode,
             "seconds": round(took, 1)}))
        built += 1
    bar.close(f"{built} built, {skipped} already done, {failed} failed")
    if modes:
        log.info("S03.1 by frame shape: %s", ", ".join(
            f"{k}={v}" for k, v in sorted(modes.items(), key=lambda kv: -kv[1])))

    return {"n_recordings": len(recs), "n_built": built, "n_skipped": skipped,
            "n_failed": failed, "modes": modes, "proxy_fps": proxy_fps,
            "hwaccel": hwaccel, "encoder": encoder,
            "yaw_videos": yaw_videos, "source_hours": round(total_s / 3600, 2)}


def detect_shots(cfg: Config, conn) -> dict[str, Any]:
    """S03.2 -- scene boundaries on each proxy become shot rows.

    Per recording rather than per file: the proxy already spans the whole take
    through the concat demuxer, so a chapter join is invisible here and cannot
    manufacture a cut that the camera never made.

    Each shot inherits its moment from the recording's start plus its own offset,
    and from that its act and its position on the route -- interpolated from the
    GPS track rather than copied from one asset, because a shot is a span and the
    walker moved during it.
    """
    from datetime import timedelta

    bounds = _bounds(conn)
    track = [gps_mod.GpsPoint(_dt(r["ts_utc"]), r["lat"], r["lon"], r["alt_dem_m"])
             for r in conn.execute("SELECT ts_utc, lat, lon, alt_dem_m FROM gps_points "
                                   "ORDER BY ts_utc")]
    max_gap = float(cfg.get("spine.max_interp_gap_s"))
    threshold = float(cfg.get("process.scene_threshold"))
    min_len = float(cfg.get("process.min_shot_s"))

    recs = [dict(r) for r in conn.execute(
        "SELECT recording_id, start_utc, duration_s FROM recordings ORDER BY start_utc")]
    proxies = cfg.work_root / "proxies"
    pending = [r for r in recs if (proxies / f"{r['recording_id']}_eq.mp4").exists()]
    if not pending:
        return {"n_recordings": len(recs), "n_shots": 0,
                "error": "no proxies on disk -- run S03.1 first"}

    out: list[dict[str, Any]] = []
    no_cuts = 0
    bar = Progress("S03.2 detecting shots", len(pending))
    for r in pending:
        rid = r["recording_id"]
        bar.step(note=rid[-24:])
        proxy = proxies / f"{rid}_eq.mp4"
        try:
            scenes = shots_mod.detect_scenes(proxy, threshold=threshold,
                                             min_len_s=min_len)
        except Exception as exc:                       # a bad proxy, not a bug
            log.warning("S03.2 %s: detection failed (%s)", rid, exc)
            continue
        if len(scenes) <= 1:
            no_cuts += 1
        rows = shots_mod.shots_for_recording(scenes, rid, min_len_s=min_len)
        rec_start = _dt(r["start_utc"])
        for row in rows:
            ts = rec_start + timedelta(seconds=row["start_s"]) if rec_start else None
            row["start_utc"] = ts.isoformat() if ts else None
            row["act"] = acts_mod.act_for(ts, bounds) if (ts and bounds) else None
            pos = gps_mod.interpolate_at(track, ts, max_gap_s=max_gap) \
                if (ts and track) else None
            if pos:
                row["lat"], row["lon"] = pos
        out += rows
    bar.close(f"{len(out)} shots from {len(pending)} recording(s)")

    db.upsert(conn, "shots", ["shot_id"], out)
    by_act: dict[Any, int] = {}
    for row in out:
        by_act[row.get("act")] = by_act.get(row.get("act"), 0) + 1
    placed = sum(1 for row in out if row.get("lat") is not None)
    log.info("S03.2 %d shot(s) from %d recording(s); per act %s; %d positioned",
             len(out), len(pending), {k: by_act[k] for k in sorted(
                 by_act, key=lambda x: (x is None, x))}, placed)
    if no_cuts:
        log.info("S03.2 %d recording(s) had no detected cut and became one shot each",
                 no_cuts)
    return {"n_recordings": len(recs), "n_with_proxy": len(pending),
            "n_shots": len(out), "per_act": {str(k): v for k, v in by_act.items()},
            "n_positioned": placed, "n_single_shot": no_cuts}


def _bounds(conn) -> list:
    raw = db.get_decision(conn, "act_boundaries")
    if not raw:
        return []
    return [acts_mod.ActBoundary(b["act"], _dt(b["start_utc"]), _dt(b["end_utc"]),
                                 b.get("method", "")) for b in json.loads(raw)]


def _measure(item: dict[str, Any], *, max_px: int, slot_base: float,
             curve: dict[str, float]) -> dict[str, Any]:
    """Decode one photograph and score it. Pure: no database, no shared state.

    Called from a thread pool, so it must touch nothing but its argument.
    """
    from PIL import Image
    import numpy as np

    with Image.open(item["src"]) as im:
        # A no-op for HEIF -- libheif decodes at full size regardless -- but it
        # still lets the JPEG decoder scale during the DCT, which is free.
        im.draft("RGB", (max_px, max_px))
        im = im.convert("RGB")
        im.thumbnail((max_px, max_px))
        arr = np.asarray(im)
        w, h = im.size

    sharp = stills.sharpness(arr)
    pen = stills.exposure_penalty(arr)
    if not stills.passes_gate(sharp, pen, curve):
        return {"rejected": "below the quality gate"}

    dur = stills.slot_duration_s((w / h) if h else None, base_s=slot_base)
    return {
        "shot_id": f"photo_{item['asset_id'][:16]}",
        "recording_id": None,
        "asset_id": item["asset_id"],
        "media_kind": "photo",
        "start_s": 0.0,
        "end_s": round(dur, 3),
        "start_utc": item["created_at_utc"],
        "act": item["act"],
        "lat": item["lat"], "lon": item["lon"], "alt_dem_m": item["alt_dem_m"],
        "place_name": item["place_name"],
        "sharpness": round(sharp, 4),
        "exposure_pen": round(pen, 4),
        "stability": None,          # a still does not shake; that is not merit
        "motion_mag": None,
        "audio_lufs": None,
        "view_kind": "still",
        "score_tech": round(stills.technical_score(sharp, pen), 4),
        "status": "candidate",
    }


def build_photo_shots(cfg: Config, conn) -> dict[str, Any]:
    """S03.0 -- one shot per photograph that clears its source's quality gate.

    A photo shot has no recording: it is a single asset held on screen, so
    ``shots.asset_id`` is set and ``recording_id`` is NULL. Stability and motion
    stay NULL rather than taking a flattering default -- a still would beat
    every clip on stability by virtue of not moving, which is not a fact about
    its quality.

    Decoding runs in a thread pool: see the note by the pool below.
    """
    bounds = _bounds(conn)

    rows = [dict(r) for r in conn.execute(
        "SELECT asset_id, s3_key, source, quality_curve, created_at_utc, "
        "lat, lon, alt_dem_m, place_name, width, height FROM assets "
        "WHERE kind='photo' AND created_at_utc IS NOT NULL")]
    if not rows:
        return {"n_photos": 0, "error": "no dated photos"}

    root = cfg.data_root
    max_px = int(cfg.get("process.photo_analysis_px", 1024))
    slot_base = float(cfg.get("process.photo_slot_s"))
    workers = int(cfg.get("process.photo_workers", 0)) or (os.cpu_count() or 4)

    # Decide what is worth measuring before measuring anything: a photo outside
    # every act cannot become a slot, and a 12 MP HEIC costs over a second to
    # decode.
    work: list[dict[str, Any]] = []
    rejected: dict[str, int] = {}
    unplaced = 0
    for r in rows:
        act = acts_mod.act_for(_dt(r["created_at_utc"]), bounds) if bounds else None
        if act is None:
            unplaced += 1
            continue
        src = root / Path(r["s3_key"]).relative_to("raw")
        if not src.exists():
            rejected["missing file"] = rejected.get("missing file", 0) + 1
            continue
        work.append({**r, "act": act, "src": src})

    # libheif releases the GIL and decoding dominates the cost, so threads give
    # very nearly linear speedup: measured 3.7x on four cores, with byte-identical
    # results. Serially, 706 photographs took 14 minutes.
    scan = Progress("S03.0 measuring photographs", len(work))
    measured: list[dict[str, Any] | None] = [None] * len(work)

    def measure(i: int) -> None:
        item = work[i]
        try:
            measured[i] = _measure(item, max_px=max_px, slot_base=slot_base,
                                   curve=cfg.quality_curve(item["quality_curve"]
                                                           or "phone"))
        except (OSError, ValueError) as exc:
            ext = item["src"].suffix.lower()
            log.debug("could not read %s: %s", item["src"].name, exc)
            if ext in stills.HEIF_EXT and not stills.heif_available():
                key = f"{ext} needs a decoder (pip install pillow-heif)"
            else:
                key = f"unreadable {ext or 'file'} ({type(exc).__name__})"
            measured[i] = {"rejected": key}
        finally:
            scan.step()

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(measure, range(len(work))))

    out: list[dict[str, Any]] = []
    for m in measured:
        if m is None:
            continue
        if m.get("rejected"):
            rejected[m["rejected"]] = rejected.get(m["rejected"], 0) + 1
        else:
            out.append(m)

    scan.close(f"{len(out)} became shots")
    db.upsert(conn, "shots", ["shot_id"], out)
    by_act: dict[int, int] = {}
    for s in out:
        by_act[s["act"]] = by_act.get(s["act"], 0) + 1
    log.info("S03.0 %d photo shot(s) from %d dated photo(s); per act %s",
             len(out), len(rows), {k: by_act[k] for k in sorted(by_act)})
    if unplaced:
        log.info("S03.0 %d photo(s) fall outside every act and were skipped", unplaced)
    for reason, n in sorted(rejected.items(), key=lambda kv: -kv[1]):
        log.info("S03.0 %d photo(s) rejected: %s", n, reason)
    missing_heif = sum(n for k, n in rejected.items() if "pillow-heif" in k)
    if missing_heif:
        log.warning(
            "S03.0 %d HEIC photograph(s) could not be decoded -- that is %.0f%% "
            "of the dated photographs, and they are iPhone stills, not junk. "
            "Install the decoder and re-run: pip install pillow-heif",
            missing_heif, missing_heif / max(len(rows), 1) * 100)
    return {"n_photos": len(rows), "n_shots": len(out), "per_act": by_act,
            "n_unplaced": unplaced, "rejected": rejected,
            "heif_decoder": stills.heif_available(),
            "n_needs_heif": missing_heif}


def run(cfg: Config, *, force: bool = False) -> dict[str, Any]:
    conn = db.init(cfg.db_path)
    report: dict[str, Any] = {"stage": STAGE, "started_utc": db.utcnow()}
    done = db.done_units(conn, STAGE)
    report["skipped_stale"] = freshness.warn_if_stale(
        log, conn, STAGE, force=force, rerun_hint="nepal s03 --force")

    steps = [("proxies", lambda: build_proxies(cfg, conn, force=force)),
             ("shots", lambda: detect_shots(cfg, conn)),
             ("photos", lambda: build_photo_shots(cfg, conn))]
    for name, fn in steps:
        if not force and name in done:
            report[name] = {"skipped": "already done"}
            continue
        report[name] = fn()
        db.mark_unit(conn, STAGE, name,
                     detail=json.dumps(report[name], default=str)[:2000])

    report["finished_utc"] = db.utcnow()
    cfg.work("reports", "s03_process.json").write_text(
        json.dumps(report, indent=2, default=str))
    log.info("S03 report written to %s", cfg.work_root / "reports" / "s03_process.json")
    conn.close()
    return report
