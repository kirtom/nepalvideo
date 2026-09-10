"""S01 -- Probe.

Know exactly what was delivered, and auto-solve the two parameters everything
downstream depends on: the fisheye FOV and the per-phone clock offsets.

Resumable at the sub-step level: the manifest, the chapter grouping, the GPS
check, the FOV solve and each phone's clock solve are separate units in
``stage_units``, so a re-run after an interruption redoes only what is missing.
Pass force=True to recompute regardless.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nepal import db
from nepal.config import Config
from nepal.probe import chapters, clock, fov, manifest
from nepal.util import proc
from nepal.util.hashing import sha256_file

log = logging.getLogger(__name__)
STAGE = "S01"


# ---------------------------------------------------------------- manifest

def build_manifest(cfg: Config, conn, *, force: bool = False) -> dict[str, Any]:
    """S01.1 -- populate ``assets``."""
    root = cfg.data_root
    if not root.exists():
        raise FileNotFoundError(
            f"data_root {root} does not exist. Set project.data_root in "
            f"{cfg.path} to the folder containing chat_export/, "
            f"media_from_camera/ and media_from_phones/."
        )

    files = manifest.walk_media(root)
    log.info("S01.1 walking %s: %d files", root, len(files))

    exif_rows: dict[str, dict[str, Any]] = {}
    if proc.have("exiftool"):
        for row in proc.exiftool_recursive(root):
            src = row.get("SourceFile")
            if src:
                exif_rows[str(Path(src).resolve())] = row
        log.info("S01.1 exiftool returned %d rows", len(exif_rows))
    else:
        log.warning("exiftool not on PATH -- falling back to ffprobe/stat only. "
                    "GPS and device timestamps will be far less complete.")

    have_ffprobe = proc.have("ffprobe")
    rows: list[dict[str, Any]] = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        cls = manifest.classify(rel)
        exif = exif_rows.get(str(path.resolve()), {})

        created = manifest.asset_datetime(exif)
        lat, lon, alt = manifest.parse_gps(exif)
        duration = manifest.parse_duration(manifest.exif_get(exif, "Duration"))
        width = height = fps = None

        size = manifest.exif_get(exif, "ImageSize")
        if size and "x" in str(size):
            try:
                width, height = (int(v) for v in str(size).split("x"))
            except ValueError:
                pass
        fps = _f(manifest.exif_get(exif, "VideoFrameRate"))

        probe_json = None
        # ffprobe only where exiftool left a gap that matters downstream
        needs_probe = cls["kind"] in ("video360", "video_flat", "audio") and (
            duration is None or width is None or fps is None)
        if needs_probe and have_ffprobe:
            try:
                summary = proc.probe_summary(path)
                duration = duration or summary["duration_s"]
                width = width or summary["width"]
                height = height or summary["height"]
                fps = fps or summary["fps"]
                probe_json = summary["probe_json"]
            except (proc.ToolFailed, proc.ToolMissing, ValueError) as exc:
                log.warning("ffprobe failed on %s: %s", rel, exc)

        rows.append({
            "asset_id": sha256_file(path),
            "s3_key": f"raw/{rel}",
            "source": cls["source"],
            "kind": cls["kind"],
            "container": cls["container"],
            "bytes": path.stat().st_size,
            "width": width,
            "height": height,
            "fps": fps,
            "duration_s": duration,
            "created_at": created.isoformat() if created else None,
            "created_at_utc": None,          # filled once offsets are known
            "recording_id": None,            # filled by S01.2
            "chapter_index": None,
            "has_gps": 1 if (lat is not None and lon is not None) else 0,
            "lat": lat,
            "lon": lon,
            "alt_dem_m": None,               # S02.3
            "place_name": None,              # S02.4
            "quality_curve": cls["quality_curve"],
            "probe_json": probe_json,
        })

    # asset_id is a content hash, so byte-identical files collapse to one row.
    # That is the right behaviour -- processing the same content twice buys
    # nothing -- but it must not happen quietly: a duplicate usually means a
    # stray backup copy, and the operator should know which path was kept.
    seen: dict[str, str] = {}
    duplicates: list[dict[str, str]] = []
    unique_rows: list[dict[str, Any]] = []
    for r in rows:
        first = seen.get(r["asset_id"])
        if first is None:
            seen[r["asset_id"]] = r["s3_key"]
            unique_rows.append(r)
        else:
            duplicates.append({"kept": first, "dropped": r["s3_key"]})
    if duplicates:
        log.warning("S01.1 %d duplicate file(s) share content with another and were "
                    "collapsed; first path by sort order wins", len(duplicates))
        for d in duplicates[:10]:
            log.warning("    kept %s  <-- dropped %s", d["kept"], d["dropped"])

    db.upsert(conn, "assets", ["asset_id"], unique_rows)
    by_source: dict[str, int] = {}
    for r in unique_rows:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    log.info("S01.1 wrote %d assets (%d files, %d duplicate): %s",
             len(unique_rows), len(rows), len(duplicates), by_source)
    return {"n_assets": len(unique_rows), "n_files_seen": len(rows),
            "by_source": by_source, "duplicates": duplicates,
            "exiftool": proc.have("exiftool"), "ffprobe": have_ffprobe}


def _f(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- chapters

def group_chapters(cfg: Config, conn) -> dict[str, Any]:
    """S01.2 -- collapse chapter-split files into ``recordings``."""
    assets = [dict(r) for r in conn.execute(
        "SELECT asset_id, s3_key, source, kind, container, duration_s, "
        "created_at, created_at_utc FROM assets")]
    for a in assets:
        a["filename"] = Path(a["s3_key"]).name

    recs = chapters.group_recordings(assets)
    db.upsert(conn, "recordings", ["recording_id"], [{
        "recording_id": r.recording_id, "source": r.source,
        "is_360": int(r.is_360), "start_utc": r.start_utc,
        "duration_s": r.duration_s, "asset_count": r.asset_count,
    } for r in recs])

    # back-link assets to their recording and chapter index
    for r in recs:
        for aid in r.asset_ids:
            a = next(x for x in assets if x["asset_id"] == aid)
            ck = chapters.parse_chapter(a["filename"])
            conn.execute("UPDATE assets SET recording_id=?, chapter_index=? WHERE asset_id=?",
                         (r.recording_id, ck.chapter_index if ck else None, aid))
    conn.commit()

    by_id = {a["asset_id"]: a for a in assets}
    problems = [p for r in recs for p in
                chapters.check_continuity(r, by_id, cfg.get("probe.max_chapter_gap_s"))]
    multi = [r for r in recs if r.asset_count > 1]
    log.info("S01.2 %d recordings (%d multi-chapter), %d continuity notes",
             len(recs), len(multi), len(problems))
    return {"n_recordings": len(recs), "n_multi_chapter": len(multi),
            "continuity_problems": problems}


# ---------------------------------------------------------------- gps check

def check_camera_gps(cfg: Config, conn) -> dict[str, Any]:
    """S01.3 -- does the camera carry GPS? Expect no."""
    row = conn.execute(
        "SELECT COUNT(*) n, SUM(has_gps) g FROM assets WHERE source='camera'").fetchone()
    n, g = row["n"] or 0, row["g"] or 0
    has = bool(g)
    db.set_decision(conn, "camera_has_gps", int(has), 1.0 if n else 0.0,
                    f"{g}/{n} camera assets carry a fix")
    if has:
        log.warning("S01.3 camera GPS present on %d/%d assets -- S02 can geotag "
                    "directly instead of interpolating", g, n)
    else:
        log.info("S01.3 no camera GPS (expected). Location comes from phone EXIF via S02.")
    return {"camera_has_gps": has, "camera_assets": n, "with_fix": g}


# ---------------------------------------------------------------- fov

def solve_fov(cfg: Config, conn, *, work: Path | None = None) -> fov.FovResult:
    """S01.4 -- seam-discontinuity minimisation."""
    fallback = float(cfg.get("probe.fov.fallback_deg"))
    rows = [dict(r) for r in conn.execute(
        "SELECT s3_key, container, duration_s FROM assets "
        "WHERE source='camera' AND kind='video360' AND duration_s > 0")]
    if not rows:
        log.warning("S01.4 no 360 camera clips found -- FOV left at fallback %.0f", fallback)
        return fov.FovResult(fallback, 0.0, "fallback:no-360-material", {}, 0, True)

    # .lrv is already a proxy: decoding it instead of a 5.7K H.265 stream is
    # most of the runtime saving in this step.
    lrv = [r for r in rows if (r["container"] or "").lower() == "lrv"]
    pool = lrv or rows
    work = work or cfg.workdir("fov")
    root = cfg.data_root

    n_frames = int(cfg.get("probe.fov.sample_frames"))
    min_files = int(cfg.get("probe.fov.min_distinct_files"))
    per_file = max(1, -(-n_frames // max(1, min(len(pool), min_files * 2))) + 1)

    # 1. measure texture on a cheap raw frame from each candidate point
    candidates: list[tuple[Path, float, float]] = []
    for r in pool[: min_files * 3]:
        src = root / Path(r["s3_key"]).relative_to("raw")
        if not src.exists():
            continue
        for t in fov.sample_timestamps(float(r["duration_s"] or 0), per_file):
            try:
                png = proc.extract_frame(src, t, work / f"tex_{src.stem}_{t:.1f}.png")
                candidates.append((src, t, fov.texture_score(fov.load_image(png))))
            except (proc.ToolFailed, proc.ToolMissing, OSError) as exc:
                log.debug("texture probe failed %s@%.1f: %s", src.name, t, exc)

    picked = fov.pick_textured_frames(candidates, n_frames, min_files)
    if not picked:
        log.warning("S01.4 could not sample any frames -- FOV left at fallback")
        return fov.FovResult(fallback, 0.0, "fallback:no-frames-sampled", {}, 0, True)

    # 2. sweep the candidate FOVs on those frames
    fovs = fov.candidate_fovs(int(cfg.get("probe.fov.candidates_start")),
                              int(cfg.get("probe.fov.candidates_stop")),
                              int(cfg.get("probe.fov.candidates_step")))
    band = int(cfg.get("probe.fov.seam_band_px"))
    local = int(cfg.get("probe.fov.local_band_px"))

    per_frame: list[dict[int, float]] = []
    for src, t in picked:
        scores: dict[int, float] = {}
        for f in fovs:
            try:
                png = fov.render_candidate(src, t, f, work / f"{src.stem}_{t:.1f}_{f}.png")
                scores[f] = fov.seam_discontinuity(fov.load_image(png), band, local)
            except (proc.ToolFailed, proc.ToolMissing, ValueError, OSError) as exc:
                log.debug("fov %d failed on %s@%.1f: %s", f, src.name, t, exc)
        if scores:
            per_frame.append(scores)

    result = fov.solve_from_scores(
        per_frame,
        min_confidence=float(cfg.get("probe.fov.min_confidence")),
        fallback_deg=fallback,
    )
    log.info("S01.4 fov_deg=%.0f confidence=%.3f (%s) from %d frames",
             result.fov_deg, result.confidence, result.method, result.n_frames)
    return result


# ---------------------------------------------------------------- clock

def solve_clocks(cfg: Config, conn, *, work: Path | None = None) -> dict[str, clock.ClockResult]:
    """S01.5 -- one offset per phone, measured against the camera."""
    work = work or cfg.workdir("clock")
    root = cfg.data_root
    fs = int(cfg.get("probe.clock.sample_rate"))
    max_audio = float(cfg.get("probe.clock.max_audio_s"))

    def clips(where: str) -> list[clock.Clip]:
        out = []
        for r in conn.execute(
            "SELECT asset_id, s3_key, source, created_at, duration_s FROM assets "
            f"WHERE {where} AND kind IN ('video360','video_flat') "
            "AND created_at IS NOT NULL AND duration_s > 0"
        ):
            dt = manifest.parse_exif_datetime(r["created_at"])
            if dt is None:
                continue
            out.append(clock.Clip(r["asset_id"], r["source"], dt,
                                  float(r["duration_s"]), r["s3_key"]))
        return out

    camera = clips("source='camera'")
    results: dict[str, clock.ClockResult] = {}

    for device in ("phone_keller", "phone_kulikov"):
        phone = clips(f"source='{device}'")
        label = device.replace("phone_", "")

        if not camera or not phone:
            naive = clock.naive_offset(camera, phone)
            results[label] = clock.reduce_measurements(
                label, [], naive_offset_s=naive,
                min_confident_pairs=int(cfg.get("probe.clock.min_confident_pairs")))
            log.warning("S01.5 %s: no overlapping material (%d camera, %d phone clips)",
                        label, len(camera), len(phone))
            continue

        pairs = clock.find_candidate_pairs(
            camera, phone, window_s=float(cfg.get("probe.clock.pair_window_s")))
        naive = clock.naive_offset(camera, phone, pairs)
        log.info("S01.5 %s: %d candidate pairs (naive delta %.1fs)", label, len(pairs), naive)

        measurements: list[clock.PairMeasurement] = []
        for cam_clip, ph_clip, cam_off, ph_off in pairs[:60]:
            try:
                cam_path = root / Path(cam_clip.path).relative_to("raw")
                ph_path = root / Path(ph_clip.path).relative_to("raw")
                if not (cam_path.exists() and ph_path.exists()):
                    continue
                cam_wav = proc.extract_audio(cam_path, work / f"{cam_clip.clip_id[:12]}_c.wav",
                                             sample_rate=fs, start_s=cam_off, duration_s=max_audio)
                ph_wav = proc.extract_audio(ph_path, work / f"{ph_clip.clip_id[:12]}_p.wav",
                                            sample_rate=fs, start_s=ph_off, duration_s=max_audio)
                a, b = _read_wav(ph_wav), _read_wav(cam_wav)
                lag, conf = clock.gcc_phat(a, b, fs,
                                           max_lag_s=float(cfg.get("probe.clock.max_lag_s")))
                from datetime import timedelta
                offset = clock.offset_from_pair(
                    cam_clip.start + timedelta(seconds=cam_off),
                    ph_clip.start + timedelta(seconds=ph_off),
                    lag)
                measurements.append(clock.PairMeasurement(
                    cam_clip.clip_id, ph_clip.clip_id, lag, conf, offset))
            except (proc.ToolFailed, proc.ToolMissing, OSError, ValueError) as exc:
                log.debug("clock pair failed: %s", exc)

        results[label] = clock.reduce_measurements(
            label, measurements,
            min_pair_confidence=float(cfg.get("probe.clock.min_pair_confidence")),
            min_confident_pairs=int(cfg.get("probe.clock.min_confident_pairs")),
            naive_offset_s=naive)
        r = results[label]
        log.info("S01.5 %s offset=%.2fs confidence=%.3f (%s)",
                 label, r.offset_s, r.confidence, r.method)
    return results


def _read_wav(path: Path):
    """16-bit PCM WAV -> float array, without pulling in soundfile."""
    import wave
    import numpy as np
    with wave.open(str(path), "rb") as w:
        frames = w.readframes(w.getnframes())
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float64)
        if w.getnchannels() > 1:
            data = data.reshape(-1, w.getnchannels()).mean(axis=1)
    return data / 32768.0


def apply_offsets(conn, offsets: dict[str, float]) -> int:
    """Write ``created_at_utc`` = created_at + offset for every asset.

    Camera is the reference clock and takes offset 0. Telegram timestamps come
    from the server, not a device, so they are already correct.
    """
    from datetime import timedelta
    source_offset = {
        "camera": 0.0,
        "phone_keller": offsets.get("keller", 0.0),
        "phone_kulikov": offsets.get("kulikov", 0.0),
        "telegram": 0.0,
        "music": 0.0,
    }
    n = 0
    for r in conn.execute("SELECT asset_id, source, created_at FROM assets "
                          "WHERE created_at IS NOT NULL"):
        dt = manifest.parse_exif_datetime(r["created_at"])
        if dt is None:
            continue
        shifted = dt + timedelta(seconds=source_offset.get(r["source"], 0.0))
        conn.execute("UPDATE assets SET created_at_utc=? WHERE asset_id=?",
                     (shifted.astimezone(timezone.utc).isoformat(), r["asset_id"]))
        n += 1
    conn.commit()

    # recordings inherit the corrected start of their earliest asset
    conn.execute("""
        UPDATE recordings SET start_utc = (
          SELECT MIN(created_at_utc) FROM assets
          WHERE assets.recording_id = recordings.recording_id
            AND created_at_utc IS NOT NULL)
        WHERE EXISTS (SELECT 1 FROM assets
                      WHERE assets.recording_id = recordings.recording_id
                        AND created_at_utc IS NOT NULL)""")
    conn.commit()
    return n


# ---------------------------------------------------------------- driver

def run(cfg: Config, *, force: bool = False, skip_fov: bool = False,
        skip_clock: bool = False) -> dict[str, Any]:
    conn = db.init(cfg.db_path)
    report: dict[str, Any] = {"stage": STAGE, "started_utc": db.utcnow()}
    done = db.done_units(conn, STAGE)

    if force or "manifest" not in done:
        report["manifest"] = build_manifest(cfg, conn, force=force)
        db.mark_unit(conn, STAGE, "manifest", detail=json.dumps(report["manifest"]))
    else:
        report["manifest"] = {"skipped": "already done"}

    if force or "chapters" not in done:
        report["chapters"] = group_chapters(cfg, conn)
        db.mark_unit(conn, STAGE, "chapters", detail=json.dumps(report["chapters"]))
    else:
        report["chapters"] = {"skipped": "already done"}

    report["gps_check"] = check_camera_gps(cfg, conn)
    db.mark_unit(conn, STAGE, "gps_check")

    if skip_fov:
        report["fov"] = {"skipped": "--skip-fov"}
    elif force or "fov" not in done:
        fr = solve_fov(cfg, conn)
        db.set_decision(conn, "fov_deg", fr.fov_deg, fr.confidence, fr.method)
        report["fov"] = {k: v for k, v in asdict(fr).items() if k != "scores"}
        report["fov"]["scores"] = {str(k): round(v, 4) for k, v in fr.scores.items()}
        db.mark_unit(conn, STAGE, "fov")
    else:
        report["fov"] = {"skipped": "already done",
                         "fov_deg": db.get_decision_float(conn, "fov_deg")}

    if skip_clock:
        report["clock"] = {"skipped": "--skip-clock"}
    elif force or "clock" not in done:
        results = solve_clocks(cfg, conn)
        offsets = {}
        report["clock"] = {}
        for label, r in results.items():
            db.set_decision(conn, f"clock_offset_{label}_s", r.offset_s,
                            r.confidence, r.method)
            offsets[label] = r.offset_s
            report["clock"][label] = {
                "offset_s": r.offset_s, "confidence": r.confidence,
                "method": r.method, "pairs_total": r.n_pairs_total,
                "pairs_accepted": r.n_pairs_accepted, "spread_s": r.spread_s,
                "needs_manual": r.needs_manual,
            }
        report["applied_utc_to_assets"] = apply_offsets(conn, offsets)
        db.mark_unit(conn, STAGE, "clock")
    else:
        report["clock"] = {"skipped": "already done"}

    report["decisions"] = db.all_decisions(conn)
    report["finished_utc"] = db.utcnow()

    out = cfg.work("reports", "s01_probe.json")
    out.write_text(json.dumps(report, indent=2, default=str))
    log.info("S01 report written to %s", out)
    conn.close()
    return report
