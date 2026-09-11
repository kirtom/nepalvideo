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
import statistics
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

# Satellite time cannot be an hour wrong, so a device stamp further than this
# from its own GPSDateTime is the stamp that is wrong, not the satellite.
GPS_OVERRIDE_S = 3600.0


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
    clock_evidence: dict[str, list[float]] = {}
    regstamped: list[dict[str, Any]] = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        cls = manifest.classify(rel)
        exif = exif_rows.get(str(path.resolve()), {})

        created = manifest.asset_datetime(exif)
        # GPSDateTime is satellite time: where a photo carries both, the
        # agreement between them is direct evidence of whether that device's
        # clock can be trusted as the pipeline's reference.
        gps_dt = manifest.parse_exif_datetime(manifest.exif_get(exif, "GPSDateTime"))
        if created is not None and gps_dt is not None:
            delta = (created - gps_dt).total_seconds()
            # A device-wide offset cannot fix an individual asset whose stamp is
            # simply wrong -- a batch file-transfer date, say. Where satellite
            # time contradicts the device clock by more than an hour, the
            # satellite is right: it cannot be off by an hour.
            if abs(delta) > GPS_OVERRIDE_S:
                regstamped.append({"path": rel, "device_said": created.isoformat(),
                                   "gps_said": gps_dt.isoformat(),
                                   "delta_days": round(delta / 86400.0, 2)})
                created = gps_dt
            else:
                clock_evidence.setdefault(cls["source"], []).append(delta)
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
    if regstamped:
        log.warning("S01.1 %d asset(s) carried a timestamp more than %.0fh from the "
                    "satellite time in their own GPS data and were re-stamped from it. "
                    "A whole group on one date is usually a batch file-transfer date "
                    "rather than a capture date: %s",
                    len(regstamped), GPS_OVERRIDE_S / 3600,
                    ", ".join(f"{r['path'].split('/')[-1]} ({r['delta_days']:+.0f}d)"
                              for r in regstamped[:4]))

    gps_agreement: dict[str, dict[str, float]] = {}
    for source, deltas in clock_evidence.items():
        med = float(statistics.median(deltas))
        spread = float(statistics.median([abs(d - med) for d in deltas]))
        gps_agreement[source] = {"n": len(deltas), "median_delta_s": round(med, 2),
                                 "mad_s": round(spread, 2)}
        db.set_decision(conn, f"clock_vs_gps_{source}_s", round(med, 2),
                        1.0 / (1.0 + spread),
                        f"median over {len(deltas)} photos carrying both "
                        f"DateTimeOriginal and GPSDateTime")
        log.info("S01.1 %s clock vs GPS time: %+.1fs (MAD %.1fs) over %d photos",
                 source, med, spread, len(deltas))

    return {"n_assets": len(unique_rows), "n_files_seen": len(rows),
            "by_source": by_source, "duplicates": duplicates,
            "gps_agreement": gps_agreement,
            "n_restamped_from_gps": len(regstamped),
            "restamped_from_gps": regstamped[:50],
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

def _clips(conn, source: str) -> list[clock.Clip]:
    """Dated video clips for one device, with their reported start times."""
    out = []
    for r in conn.execute(
        "SELECT asset_id, s3_key, source, created_at, duration_s FROM assets "
        "WHERE source=? AND kind IN ('video360','video_flat') "
        "AND created_at IS NOT NULL AND duration_s > 0", (source,)
    ):
        dt = manifest.parse_exif_datetime(r["created_at"])
        if dt is not None:
            out.append(clock.Clip(r["asset_id"], r["source"], dt,
                                  float(r["duration_s"]), r["s3_key"]))
    return out


def _capture_times(conn, source: str) -> list:
    """Every dated capture for a device -- photos included.

    The coarse aligner wants volume: phone photos outnumber phone videos by an
    order of magnitude here, and it is their hour-by-hour distribution that
    makes the activity histogram legible.
    """
    times = []
    for r in conn.execute(
        "SELECT created_at FROM assets WHERE source=? AND created_at IS NOT NULL", (source,)
    ):
        dt = manifest.parse_exif_datetime(r["created_at"])
        if dt is not None:
            times.append(dt)
    return sorted(times)


def pick_reference(conn) -> tuple[str, str]:
    """Which device's clock the pipeline treats as truth.

    The spec nominates the camera. On this corpus that is the wrong choice: the
    camera's dates sit two weeks off the phones', which is the signature of a
    flat battery resetting an action camera's clock, while the phone photos
    carry GPS fixes and therefore satellite time. Whichever device agrees most
    closely with GPS wins; with no GPS evidence at all the order falls back to
    keller, kulikov, camera.
    """
    best, best_err, evidence = None, None, {}
    for source in ("phone_keller", "phone_kulikov"):
        med = db.get_decision_float(conn, f"clock_vs_gps_{source}_s")
        if med is None:
            continue
        evidence[source] = abs(med)
        if best_err is None or abs(med) < best_err:
            best, best_err = source, abs(med)

    if best is not None:
        return best, (f"GPS-validated: {best} sits {best_err:.1f}s from satellite "
                      f"time (candidates {evidence})")

    for source in ("phone_keller", "phone_kulikov", "camera"):
        if conn.execute("SELECT 1 FROM assets WHERE source=? AND created_at IS NOT NULL "
                        "LIMIT 1", (source,)).fetchone():
            return source, f"fallback: no GPS timestamps available, defaulting to {source}"
    return "camera", "fallback: no dated assets at all"


def _measure_pairs(cfg, root: Path, work: Path, ref_clips, pairs, coarse: float,
                   fs: int, max_audio: float) -> list:
    """GCC-PHAT over each candidate pair, returning measurements in absolute
    offset terms (coarse shift folded back in)."""
    from datetime import timedelta
    out = []
    for ref_clip, tgt_clip, ref_off, tgt_off in pairs[:60]:
        try:
            ref_path = root / Path(ref_clip.path).relative_to("raw")
            tgt_path = root / Path(tgt_clip.path).relative_to("raw")
            if not (ref_path.exists() and tgt_path.exists()):
                continue
            ref_wav = proc.extract_audio(ref_path, work / f"{ref_clip.clip_id[:12]}_r.wav",
                                         sample_rate=fs, start_s=ref_off, duration_s=max_audio)
            tgt_wav = proc.extract_audio(tgt_path, work / f"{tgt_clip.clip_id[:12]}_t.wav",
                                         sample_rate=fs, start_s=tgt_off, duration_s=max_audio)
            lag, conf = clock.gcc_phat(
                _read_wav(tgt_wav), _read_wav(ref_wav), fs,
                max_lag_s=float(cfg.get("probe.clock.max_lag_s")))
            fine = clock.offset_from_pair(
                ref_clip.start + timedelta(seconds=ref_off),
                tgt_clip.start + timedelta(seconds=tgt_off), lag)
            out.append(clock.PairMeasurement(ref_clip.clip_id, tgt_clip.clip_id,
                                             lag, conf, coarse + fine))
        except (proc.ToolFailed, proc.ToolMissing, OSError, ValueError) as exc:
            log.debug("clock pair failed: %s", exc)
    return out


def solve_clocks(cfg, conn, *, work: Path | None = None) -> dict[str, clock.ClockResult]:
    """S01.5 -- one offset per device, measured against the reference clock.

    Two stages, because a single one cannot span the errors that occur in
    practice. Capture-activity histograms recover a bulk offset of days first;
    GCC-PHAT over shared audio then refines it to sub-second. Skipping the
    coarse stage leaves a fourteen-day camera error entirely invisible to a
    +/-600 s audio search.
    """
    work = work or cfg.workdir("clock")
    root = cfg.data_root
    fs = int(cfg.get("probe.clock.sample_rate"))
    max_audio = float(cfg.get("probe.clock.max_audio_s"))
    coarse_bin = float(cfg.get("probe.clock.coarse_bin_s", 3600.0))
    coarse_max = float(cfg.get("probe.clock.coarse_max_offset_s", 45 * 86400.0))

    reference, ref_why = pick_reference(conn)
    db.set_decision(conn, "clock_reference", reference, 1.0, ref_why)
    log.info("S01.5 reference clock: %s (%s)", reference, ref_why)

    ref_times = _capture_times(conn, reference)
    ref_clips = _clips(conn, reference)
    results: dict[str, clock.ClockResult] = {}

    for device in ("camera", "phone_keller", "phone_kulikov"):
        label = device.replace("phone_", "")
        if device == reference:
            results[label] = clock.ClockResult(
                device=label, offset_s=0.0, confidence=1.0,
                method=f"reference clock ({ref_why})")
            continue

        # A device carrying GPS already tells us its clock error directly:
        # GPSDateTime is satellite time. Where both the reference and this
        # device have that evidence, the offset between them is the difference
        # of their measured errors, and no audio solve can beat it.
        #
        # Letting audio run anyway was actively harmful on real material.
        # kulikov's phone agreed with satellite time to 1.0 s over 292 photos,
        # and keller's to 1.0 s over 338, so the true offset between them is
        # zero -- but GCC-PHAT returned 6781 s at confidence 0.04 with 23 s of
        # scatter across five pairs, and that was applied. Nearly two hours of
        # spurious correction, from the weakest evidence available, overriding
        # the strongest.
        ref_gps = db.get_decision_float(conn, f"clock_vs_gps_{reference}_s")
        dev_gps = db.get_decision_float(conn, f"clock_vs_gps_{device}_s")
        if ref_gps is not None and dev_gps is not None:
            gps_offset = ref_gps - dev_gps
            results[label] = clock.ClockResult(
                device=label, offset_s=round(gps_offset, 3), confidence=1.0,
                method=(f"GPS-derived: this device sits {dev_gps:+.1f}s from satellite "
                        f"time and the reference {ref_gps:+.1f}s, so the offset between "
                        f"them is {gps_offset:+.1f}s. Satellite time outranks audio "
                        f"cross-correlation, which is reserved for devices carrying "
                        f"no GPS."))
            log.info("S01.5 %s offset=%+.2fs from GPS evidence (no audio solve needed)",
                     label, gps_offset)
            continue

        target_times = _capture_times(conn, device)
        target_clips = _clips(conn, device)
        if not target_times:
            results[label] = clock.ClockResult(
                label, 0.0, 0.0, f"no dated {device} assets", needs_manual=True)
            continue

        # -- stage 1: coarse candidates --------------------------------
        # The coarse aligner is treated as a candidate generator, not an
        # answer. Its blind spot is a whole-day shift, and on sparse material
        # it can also return a spurious single bin when the true offset is
        # near zero -- which, if applied, moves clips out of overlap and
        # starves the audio stage of the very pairs that would have corrected
        # it. So zero is always a candidate too, and audio picks the winner.
        coarse, coarse_conf = clock.coarse_offset_by_activity(
            ref_times, target_times, bin_s=coarse_bin, max_offset_s=coarse_max)
        # Coincidence votes first -- they survive sparse material, which the
        # histogram does not -- then histogram peaks as a second opinion.
        # Tolerance is tied to the audio stage's reach, NOT to the histogram
        # bin: a window as wide as coarse_bin/2 lets background coincidences
        # outvote real ones on a dense reference.
        votes = clock.offset_candidates_by_coincidence(
            ref_times, target_times,
            tolerance_s=float(cfg.get("probe.clock.max_lag_s")) / 2,
            max_offset_s=coarse_max, top_n=12)
        candidates = [(c, float(v)) for c, v in votes]
        candidates += clock.coarse_candidates(
            ref_times, target_times, bin_s=coarse_bin, max_offset_s=coarse_max, top_n=4)
        if votes:
            coarse = votes[0][0]
            top_votes = votes[0][1]
            runner = votes[1][1] if len(votes) > 1 else 0
            coarse_conf = top_votes / max(runner, 1)
        if coarse:
            log.info("S01.5 %s coarse candidates: %s (best confidence %.2f)", label,
                     ", ".join(f"{c/86400:+.2f}d" for c, _ in candidates), coarse_conf)
            db.set_decision(conn, f"clock_coarse_{label}_s", round(coarse, 1),
                            round(coarse_conf, 3),
                            "activity-histogram cross-correlation; candidates "
                            + ", ".join(f"{c/86400:+.2f}d" for c, _ in candidates))

        # Candidates that put no clips in overlap cost nothing to reject --
        # find_candidate_pairs returns empty and no audio is decoded -- so the
        # list can afford to be generous.
        offsets_to_try = [0.0]
        for c, _ in candidates:
            if all(abs(c - existing) > 2 * float(cfg.get("probe.clock.max_lag_s"))
                   for existing in offsets_to_try):
                offsets_to_try.append(c)

        # -- stage 2: audio refines, and arbitrates between candidates --
        best: clock.ClockResult | None = None
        best_coarse = 0.0
        for cand in offsets_to_try:
            shifted = clock.apply_coarse(target_clips, cand)
            pairs = clock.find_candidate_pairs(
                ref_clips, shifted, window_s=float(cfg.get("probe.clock.pair_window_s")))
            if not pairs:
                continue
            measurements = _measure_pairs(cfg, root, work, ref_clips, pairs, cand, fs, max_audio)
            r = clock.reduce_measurements(
                label, measurements,
                min_pair_confidence=float(cfg.get("probe.clock.min_pair_confidence")),
                min_confident_pairs=int(cfg.get("probe.clock.min_confident_pairs")),
                naive_offset_s=cand + clock.naive_offset(ref_clips, shifted, pairs))
            if best is None or (r.n_pairs_accepted, r.confidence) > \
                    (best.n_pairs_accepted, best.confidence):
                best, best_coarse = r, cand
            if r.n_pairs_accepted >= int(cfg.get("probe.clock.min_confident_pairs")):
                log.info("S01.5 %s: audio confirms candidate %+.2fd with %d pairs",
                         label, cand / 86400.0, r.n_pairs_accepted)
                break

        if best is None:
            best = clock.reduce_measurements(
                label, [], min_confident_pairs=int(cfg.get("probe.clock.min_confident_pairs")),
                naive_offset_s=coarse)
        result = best

        # A coarse-only answer beats nothing, but it is good only to the day.
        if result.needs_manual and coarse:
            result.offset_s = round(coarse, 1)
            result.confidence = round(min(coarse_conf / 4.0, 0.5), 3)
            result.method = (f"coarse-only ({coarse/86400:+.2f} days from activity "
                             f"histogram, confidence {coarse_conf:.2f}); no confident "
                             f"audio pair to arbitrate -- accurate to about a day, "
                             f"confirm at Gate 1")
            if coarse_conf < clock.DIURNAL_CONFIDENCE:
                result.method += (" | WARNING not decisive against a whole-day shift; "
                                  "candidates "
                                  + ", ".join(f"{c/86400:+.2f}d" for c, _ in candidates[:3]))
        elif best_coarse:
            result.method += f" | coarse stage supplied {best_coarse/86400:+.2f}d, audio confirmed"

        results[label] = result
        log.info("S01.5 %s offset=%+.2fs confidence=%.3f (%s)",
                 label, result.offset_s, result.confidence, result.method)
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

    The reference device takes offset 0; every other device takes the offset
    S01.5 measured for it. Telegram timestamps come from the server rather than
    a device, so they are already correct and are never shifted.
    """
    from datetime import timedelta
    source_offset = {
        "camera": offsets.get("camera", 0.0),
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
