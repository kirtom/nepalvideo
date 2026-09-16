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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from nepal import db, freshness
from nepal.config import Config
from nepal.process import (asr as asr_mod, audio as audio_mod,
                           embed as embed_mod, faces as faces_mod,
                           gate as gate_mod, metrics as metrics_mod,
                           reproject, shots as shots_mod, stills)
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
    max_len = float(cfg.get("process.max_shot_s", 20.0))

    recs = [dict(r) for r in conn.execute(
        "SELECT recording_id, start_utc, duration_s FROM recordings ORDER BY start_utc")]
    proxies = cfg.work_root / "proxies"
    pending = [r for r in recs if (proxies / f"{r['recording_id']}_eq.mp4").exists()]
    if not pending:
        return {"n_recordings": len(recs), "n_shots": 0,
                "error": "no proxies on disk -- run S03.1 first"}

    out: list[dict[str, Any]] = []
    no_cuts = 0
    single_scene: set[str] = set()
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
            single_scene.add(rid)
        rows = shots_mod.shots_for_recording(scenes, rid, min_len_s=min_len,
                                             max_len_s=max_len)
        rec_start = _dt(r["start_utc"])
        for row in rows:
            ts = rec_start + timedelta(seconds=row["start_s"]) if rec_start else None
            row["start_utc"] = ts.isoformat() if ts else None
            row["act"] = acts_mod.act_for(ts, bounds) if (ts and bounds) else None
            pos = gps_mod.interpolate_at(track, ts, max_gap_s=max_gap) \
                if (ts and track) else None
            # Every row carries every column: db.upsert takes its column list
            # from the first row and refuses rows that differ.
            row["lat"], row["lon"] = pos if pos else (None, None)
        out += rows
    bar.close(f"{len(out)} shots from {len(pending)} recording(s)")

    # Re-detection replaces a recording's shots rather than adding to them. The
    # boundaries move whenever the threshold or the maximum length moves, and an
    # upsert alone would leave the old numbering behind as extra shots that no
    # longer describe anything. Photo shots have no recording_id and are
    # untouched.
    stale = [(rid,) for rid in {row["recording_id"] for row in out}]
    conn.executemany("DELETE FROM shots WHERE recording_id = ?", stale)
    db.upsert(conn, "shots", ["shot_id"], out)
    by_act: dict[Any, int] = {}
    for row in out:
        by_act[row.get("act")] = by_act.get(row.get("act"), 0) + 1
    placed = sum(1 for row in out if row.get("lat") is not None)
    log.info("S03.2 %d shot(s) from %d recording(s); per act %s; %d positioned",
             len(out), len(pending), {k: by_act[k] for k in sorted(
                 by_act, key=lambda x: (x is None, x))}, placed)
    if no_cuts:
        # One scene is not one shot any more: process.max_shot_s divides it.
        from_single = sum(1 for row in out if row["recording_id"] in single_scene)
        log.info("S03.2 %d recording(s) had no detected cut; the %.0f s cap "
                 "divided those into %d shot(s)", no_cuts, max_len, from_single)
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
    # A photograph that fails the gate is still written, as a rejected shot.
    # S03.7 owns status for every shot in the film and re-runs whenever the
    # thresholds move; a photo deleted here instead would need the 9-minute
    # decode pass again to come back.
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
        "status": "candidate" if stills.passes_gate(sharp, pen, curve) else "rejected",
    }


def measure_shots(cfg: Config, conn, *, force: bool = False) -> dict[str, Any]:
    """S03.3 -- sharpness, exposure, motion and stability for every video shot.

    Sampled from the proxy, five points per shot, three consecutive frames at
    each. One VideoCapture per recording walked forward in shot order: seeking
    a long proxy is the cost here, and a recording that became a single shot
    still only pays for five seeks.

    Resumable through the data rather than through ``stage_units``: a shot with
    a sharpness has been measured. That survives a kill mid-run without a unit
    per shot, and ``--force`` re-measures everything.

    The distribution of each metric is logged at the end. The gate's thresholds
    are the spec's, written before anyone had seen this corpus, and the only way
    to know whether they reject the intended 70% is to look at the numbers they
    are about to be applied to.
    """
    proxies = cfg.work_root / "proxies"
    where = "" if force else " AND s.sharpness IS NULL"
    rows = [dict(r) for r in conn.execute(
        "SELECT s.shot_id, s.recording_id, s.start_s, s.end_s, r.is_360 "
        "FROM shots s JOIN recordings r ON r.recording_id = s.recording_id "
        f"WHERE s.media_kind = 'video'{where} "
        "ORDER BY s.recording_id, s.start_s")]
    if not rows:
        return {"n_shots": 0, "note": "every video shot already measured"}

    by_rec: dict[str, list[dict]] = {}
    for r in rows:
        by_rec.setdefault(r["recording_id"], []).append(r)
    missing = [rid for rid in by_rec if not (proxies / f"{rid}_eq.mp4").exists()]
    for rid in missing:
        by_rec.pop(rid, None)

    n_samples = int(cfg.get("process.metric_samples_per_shot", 5))
    n_frames = int(cfg.get("process.metric_frames_per_sample", 3))
    flow_px = int(cfg.get("process.metric_flow_px", metrics_mod.FLOW_WIDTH))
    jerk_ref = float(cfg.get("process.metric_jerk_ref_px", metrics_mod.JERK_REF_PX))
    workers = int(cfg.get("process.metric_workers", 0)) or (os.cpu_count() or 4)

    bar = Progress("S03.3 measuring shots", sum(len(v) for v in by_rec.values()))
    results: list[dict[str, Any]] = []
    lock = threading.Lock()
    failed: dict[str, int] = {}

    def measure_recording(rid: str) -> None:
        import cv2
        shots = by_rec[rid]
        cap = cv2.VideoCapture(str(proxies / f"{rid}_eq.mp4"))
        try:
            if not cap.isOpened():
                with lock:
                    failed["unreadable proxy"] = failed.get("unreadable proxy", 0) + len(shots)
                bar.step(len(shots))
                return
            for sh in shots:
                try:
                    times = metrics_mod.sample_times(sh["start_s"], sh["end_s"], n_samples)
                    groups = metrics_mod.read_samples(
                        proxies / f"{rid}_eq.mp4", times,
                        frames_per_sample=n_frames, cap=cap)
                    m = metrics_mod.measure_samples(
                        groups, equirect=bool(sh["is_360"]),
                        width=flow_px, jerk_ref=jerk_ref)
                except Exception as exc:               # a bad proxy, not a bug
                    log.debug("S03.3 %s: %s", sh["shot_id"], exc)
                    m = None
                    with lock:
                        key = type(exc).__name__
                        failed[key] = failed.get(key, 0) + 1
                if m is None:
                    with lock:
                        failed["no frames"] = failed.get("no frames", 0) + 1
                else:
                    with lock:
                        results.append({"shot_id": sh["shot_id"], **m})
                bar.step()
        finally:
            cap.release()

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(measure_recording, list(by_rec)))
    bar.close(f"{len(results)} measured")

    conn.executemany(
        "UPDATE shots SET sharpness=?, exposure_pen=?, motion_mag=?, stability=?, "
        "jerk_px=? WHERE shot_id=?",
        [(r["sharpness"], r["exposure_pen"], r["motion_mag"], r["stability"],
          r.get("jerk_px"), r["shot_id"]) for r in results])
    conn.commit()

    dist = {k: metrics_mod.percentiles([r[k] for r in results])
            for k in ("sharpness", "exposure_pen", "motion_mag", "jerk_px",
                      "stability")}
    for k, v in dist.items():
        log.info("S03.3 %-13s %s", k, v)
    if missing:
        log.warning("S03.3 %d recording(s) have shots but no proxy on disk", len(missing))
    if failed:
        log.warning("S03.3 could not measure: %s", failed)
    return {"n_shots": len(rows), "n_measured": len(results),
            "n_missing_proxies": len(missing), "failed": failed,
            "samples_per_shot": n_samples, "distribution": dist}


def refresh_audio_flags(cfg: Config, conn) -> int:
    """Derive ``has_speech`` and ``wind`` from what S03.4 stored.

    Seconds of speech and the low-frequency share are measurements; whether
    they amount to "speech" or "wind" is a threshold, so the flags are
    re-derived whenever the thresholds could have moved -- after measuring,
    and again at the gate. Returns how many shots were flagged either way.
    """
    speech_min = float(cfg.get("process.speech_min_s", audio_mod.SPEECH_MIN_S))
    wind_thr = float(cfg.get("process.wind_lf_ratio", audio_mod.WIND_SHARE))
    flags: list[tuple[int, int, str]] = []
    for r in conn.execute("SELECT shot_id, speech_s, wind_lf_share FROM shots "
                          "WHERE speech_s IS NOT NULL OR wind_lf_share IS NOT NULL"):
        flags.append((int(audio_mod.has_speech(r["speech_s"], min_s=speech_min)),
                      int(audio_mod.is_wind(r["wind_lf_share"], threshold=wind_thr)),
                      r["shot_id"]))
    if flags:
        conn.executemany("UPDATE shots SET has_speech=?, wind=? WHERE shot_id=?", flags)
        conn.commit()
    return len(flags)


def measure_audio(cfg: Config, conn, *, force: bool = False) -> dict[str, Any]:
    """S03.4 -- loudness, wind and speech for every video shot.

    From the 16 kHz track S03.1 wrote, so no media is decoded here. The VAD
    runs once per recording and its spans are cut against each shot's window;
    loudness and the low-frequency share are per shot. Resumable through the
    data: a shot with ``speech_s`` has been measured (it is 0.0 when nothing
    was heard, unlike ``audio_lufs``, which is null for digital silence).

    The total minutes of speech is logged because it is the number that
    decides where S03.5 runs: transcription cost is proportional to it and
    nothing else.
    """
    audio_dir = cfg.work_root / "audio"
    where = "" if force else " AND s.speech_s IS NULL"
    rows = [dict(r) for r in conn.execute(
        "SELECT s.shot_id, s.recording_id, s.start_s, s.end_s FROM shots s "
        f"WHERE s.media_kind = 'video'{where} ORDER BY s.recording_id, s.start_s")]
    if not rows:
        return {"n_shots": 0, "note": "every video shot already measured"}

    by_rec: dict[str, list[dict]] = {}
    for r in rows:
        by_rec.setdefault(r["recording_id"], []).append(r)
    missing = [rid for rid in by_rec if not (audio_dir / f"{rid}.wav").exists()]
    for rid in missing:
        by_rec.pop(rid, None)
    if missing:
        log.warning("S03.4 %d recording(s) have no audio track and are skipped: %s",
                    len(missing), ", ".join(missing[:6]))

    cutoff = float(cfg.get("process.wind_lf_cutoff_hz", audio_mod.WIND_CUTOFF_HZ))
    min_sil = int(cfg.get("process.vad_min_silence_ms", 500))
    pad = int(cfg.get("process.vad_pad_ms", 200))
    workers = int(cfg.get("process.audio_workers", 0)) or (os.cpu_count() or 1)

    results: list[dict[str, Any]] = []
    failed: dict[str, int] = {}
    lock = threading.Lock()
    bar = Progress("S03.4 measuring audio", len(rows))

    def measure_recording(rid: str) -> None:
        wav = audio_dir / f"{rid}.wav"
        try:
            segments = audio_mod.speech_segments_of(wav, min_silence_ms=min_sil, pad_ms=pad)
        except Exception as exc:                       # a bad track, not a bug
            log.warning("S03.4 %s: VAD failed (%s); speech unknown", rid, exc)
            segments = None
        for sh in by_rec[rid]:
            try:
                x, sr = audio_mod.read_window(wav, sh["start_s"], sh["end_s"])
                share = audio_mod.low_frequency_share(x, sr, cutoff_hz=cutoff)
                lufs = audio_mod.lufs_of(wav, sh["start_s"], sh["end_s"])
            except Exception as exc:
                log.debug("S03.4 %s: %s", sh["shot_id"], exc)
                with lock:
                    key = type(exc).__name__
                    failed[key] = failed.get(key, 0) + 1
                bar.step()
                continue
            speech = (audio_mod.speech_seconds(segments, sh["start_s"], sh["end_s"])
                      if segments is not None else None)
            with lock:
                results.append({"shot_id": sh["shot_id"], "audio_lufs": lufs,
                                "wind_lf_share": round(share, 4),
                                "speech_s": None if speech is None else round(speech, 3)})
            bar.step()

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(measure_recording, list(by_rec)))
    bar.close(f"{len(results)} measured")

    conn.executemany(
        "UPDATE shots SET audio_lufs=?, wind_lf_share=?, speech_s=? WHERE shot_id=?",
        [(r["audio_lufs"], r["wind_lf_share"], r["speech_s"], r["shot_id"])
         for r in results])
    conn.commit()
    refresh_audio_flags(cfg, conn)

    lufs = [r["audio_lufs"] for r in results if r["audio_lufs"] is not None]
    shares = [r["wind_lf_share"] for r in results]
    speech = [r["speech_s"] for r in results if r["speech_s"] is not None]
    dist = {"audio_lufs": metrics_mod.percentiles(lufs),
            "wind_lf_share": metrics_mod.percentiles(shares),
            "speech_s": metrics_mod.percentiles(speech)}
    for k, v in dist.items():
        log.info("S03.4 %-13s %s", k, v)
    total_speech = sum(speech)
    with_speech = sum(1 for v in speech if audio_mod.has_speech(
        v, min_s=float(cfg.get("process.speech_min_s", audio_mod.SPEECH_MIN_S))))
    shot_s = sum(float(r["end_s"]) - float(r["start_s"]) for r in rows)
    log.info("S03.4 speech: %.1f min in %d of %d shot(s) (%.0f%% of %.1f min of "
             "footage) -- this is what S03.5 has to transcribe",
             total_speech / 60, with_speech, len(results),
             100.0 * total_speech / max(1.0, shot_s), shot_s / 60)
    if failed:
        log.warning("S03.4 %d shot(s) failed: %s", sum(failed.values()), failed)
    return {"n_shots": len(rows), "n_measured": len(results),
            "n_missing_audio": len(missing), "failed": failed,
            "speech_total_s": round(total_speech, 1), "n_with_speech": with_speech,
            "footage_s": round(shot_s, 1), "distribution": dist}


def transcribe_shots(cfg: Config, conn, *, force: bool = False) -> dict[str, Any]:
    """S03.5 -- transcribe every candidate shot that carries speech.

    Resumable through the data: a shot with a transcript (even an empty one)
    has been done. Only candidates: a rejected shot is not worth a model's
    time, and if a threshold change brings it back it is picked up then.

    Logs the realtime factor as it goes. On a CPU this stage is the slowest
    thing in the pipeline by an order of magnitude and the number that
    decides whether it should be running here at all is only known once it
    has run for a few minutes.
    """
    audio_dir = cfg.work_root / "audio"
    where = "" if force else " AND s.transcript IS NULL"
    rows = [dict(r) for r in conn.execute(
        "SELECT s.shot_id, s.recording_id, s.start_s, s.end_s, s.speech_s "
        "FROM shots s WHERE s.media_kind = 'video' AND s.has_speech = 1 "
        f"AND s.status = 'candidate'{where} ORDER BY s.recording_id, s.start_s")]
    if not rows:
        return {"n_shots": 0, "note": "every speech shot already transcribed"}

    model_name = cfg.get("spine.whisper_model")
    language = cfg.get("spine.whisper_language")
    beam = int(cfg.get("process.asr_beam_size", 5))
    threads = int(cfg.get("process.asr_cpu_threads", 0))
    speech_total = sum(float(r["speech_s"] or 0) for r in rows)
    log.info("S03.5 transcribing %d shot(s), %.1f min of speech, with %s (%s), beam %d",
             len(rows), speech_total / 60, model_name, language or "auto", beam)
    try:
        model = asr_mod.load_model(model_name, cpu_threads=threads)
    except ImportError:
        log.warning("S03.5 faster-whisper not installed -- skipping (pip install '.[asr]')")
        return {"skipped": "faster-whisper not installed"}

    out_dir = cfg.workdir("transcripts", "shots")
    bar = Progress("S03.5 transcribing", len(rows))
    done_n = 0
    failed: dict[str, int] = {}
    t0 = time.monotonic()
    audio_done = 0.0
    for r in rows:
        bar.step(note=r["shot_id"][-24:])
        wav = audio_dir / f"{r['recording_id']}.wav"
        try:
            x, _sr = audio_mod.read_window(wav, r["start_s"], r["end_s"])
            res = asr_mod.transcribe_window(model, x, language=language, beam_size=beam,
                                            offset_s=float(r["start_s"]))
        except Exception as exc:                       # one shot must not end the stage
            log.warning("S03.5 %s failed: %s", r["shot_id"], exc)
            key = type(exc).__name__
            failed[key] = failed.get(key, 0) + 1
            continue
        (out_dir / f"{r['shot_id'].replace('#', '_')}.json").write_text(json.dumps(
            {"shot_id": r["shot_id"], **res}, ensure_ascii=False, indent=1))
        conn.execute("UPDATE shots SET transcript=? WHERE shot_id=?",
                     (res["text"], r["shot_id"]))
        conn.commit()                                  # each shot survives a kill
        done_n += 1
        audio_done += float(r["end_s"]) - float(r["start_s"])
        if done_n % 25 == 0:
            el = time.monotonic() - t0
            log.info("S03.5 %d done: %.1fx realtime on the shot audio; %.0f min left at this rate",
                     done_n, el / max(audio_done, 1e-6),
                     (sum(float(q["end_s"]) - float(q["start_s"]) for q in rows) - audio_done)
                     * (el / max(audio_done, 1e-6)) / 60)
    el = time.monotonic() - t0
    bar.close(f"{done_n} transcribed in {el/60:.1f} min")
    chars = conn.execute("SELECT COALESCE(SUM(LENGTH(transcript)),0) FROM shots "
                         "WHERE transcript IS NOT NULL").fetchone()[0]
    empty = conn.execute("SELECT COUNT(*) FROM shots WHERE transcript = ''").fetchone()[0]
    log.info("S03.5 %d transcribed, %.1fx realtime; %d character(s) of transcript in "
             "the table, %d speech shot(s) came back empty",
             done_n, el / max(audio_done, 1e-6), chars, empty)
    if failed:
        log.warning("S03.5 %d shot(s) failed: %s", sum(failed.values()), failed)
    return {"n_shots": len(rows), "n_transcribed": done_n, "failed": failed,
            "seconds": round(el, 1), "realtime_factor": round(el / max(audio_done, 1e-6), 2),
            "n_empty": empty, "chars": chars}


def detect_faces(cfg: Config, conn, *, force: bool = False) -> dict[str, Any]:
    """S03.6 -- who is in each shot.

    Two passes. The first reads every candidate shot and collects face
    embeddings; the second clusters them across the whole corpus at once,
    because "the same person" cannot be decided one shot at a time. Only then
    are ``has_face`` and ``face_cluster`` written.

    360 shots are read through four rectilinear yaw views; flat shots are
    already rectilinear and are read directly. Embeddings are kept in memory
    between the passes -- a few thousand 512-float vectors is a few megabytes
    -- and the shot keeps the strongest face it showed, with the yaw it was
    facing, which is what S04.2 needs to pick a framing.

    Resumable through the data: a shot with ``has_face`` set has been looked
    at. Note that the clustering is global, so a partial re-run relabels
    against only what it re-reads; ``--redo faces`` re-reads everything.
    """
    proxies = cfg.work_root / "proxies"
    # Resumable on face_score, not has_face: has_face carries a DEFAULT 0 from
    # the schema, so every never-looked-at shot already has a value and a
    # resumable run would select nothing. face_score is null until measured.
    where = "" if force else " AND s.face_score IS NULL"
    rows = [dict(r) for r in conn.execute(
        "SELECT s.shot_id, s.recording_id, s.start_s, s.end_s, r.is_360 "
        "FROM shots s JOIN recordings r ON r.recording_id = s.recording_id "
        f"WHERE s.media_kind = 'video' AND s.status = 'candidate'{where} "
        "ORDER BY s.recording_id, s.start_s")]
    if not rows:
        return {"n_shots": 0, "note": "every candidate shot already looked at"}

    yaws = [int(y) for y in cfg.get("process.face_yaws", list(faces_mod.YAWS))]
    fov = float(cfg.get("process.face_view_fov_deg", faces_mod.VIEW_FOV_DEG))
    px = int(cfg.get("process.face_view_px", faces_mod.VIEW_SIZE[0]))
    n_samples = int(cfg.get("process.face_samples_per_shot", 2))
    min_score = float(cfg.get("process.face_min_det_score", faces_mod.MIN_DET_SCORE))
    thresh = float(cfg.get("process.face_same_person_cos", faces_mod.SAME_PERSON_COS))
    model = str(cfg.get("process.face_model", "buffalo_l"))
    gpu = bool(cfg.get("process.face_gpu", False))

    log.info("S03.6 %d candidate shot(s), %d sample(s) each, %s on %s",
             len(rows), n_samples, model, "GPU" if gpu else "CPU")
    try:
        app = faces_mod.load_model(model, gpu=gpu, det_size=(px, px))
    except ImportError:
        log.warning("S03.6 insightface not installed -- skipping "
                    "(pip install '.[faces]')")
        return {"skipped": "insightface not installed"}

    by_rec: dict[str, list[dict]] = {}
    for r in rows:
        by_rec.setdefault(r["recording_id"], []).append(r)
    missing = [rid for rid in by_rec if not (proxies / f"{rid}_eq.mp4").exists()]
    for rid in missing:
        by_rec.pop(rid, None)
    if missing:
        log.warning("S03.6 %d recording(s) have no proxy and are skipped", len(missing))

    # pass one: look at every shot
    per_shot: list[dict[str, Any]] = []
    embeddings: list[np.ndarray] = []
    owners: list[int] = []                 # index into per_shot, per embedding
    views_for: dict[tuple[int, int], faces_mod.YawViews] = {}
    failed: dict[str, int] = {}
    t0 = time.monotonic()
    bar = Progress("S03.6 reading faces", len(rows))
    for rid, shots in by_rec.items():
        cap = None
        try:
            import cv2
            cap = cv2.VideoCapture(str(proxies / f"{rid}_eq.mp4"))
            for sh in shots:
                bar.step(note=sh["shot_id"][-24:])
                try:
                    times = metrics_mod.sample_times(sh["start_s"], sh["end_s"], n_samples)
                    groups = metrics_mod.read_samples(
                        proxies / f"{rid}_eq.mp4", times,
                        frames_per_sample=1, cap=cap)
                except Exception as exc:               # a bad proxy, not a bug
                    log.debug("S03.6 %s: %s", sh["shot_id"], exc)
                    failed[type(exc).__name__] = failed.get(type(exc).__name__, 0) + 1
                    continue
                found: list[dict[str, Any]] = []
                for frames in groups:
                    for frame in frames:
                        if frame is None or not getattr(frame, "size", 0):
                            continue
                        if sh["is_360"]:
                            key = frame.shape[:2]
                            views = views_for.get(key)
                            if views is None:
                                views = faces_mod.YawViews(key, yaws, fov_deg=fov,
                                                           out=(px, px))
                                views_for[key] = views
                            for yaw in yaws:
                                for f in faces_mod.detect(app, views.render(frame, yaw),
                                                          min_score=min_score):
                                    found.append({**f, "yaw": float(yaw)})
                        else:
                            for f in faces_mod.detect(app, frame, min_score=min_score):
                                found.append({**f, "yaw": None})
                idx = len(per_shot)
                best = max(found, key=lambda f: f["score"]) if found else None
                per_shot.append({"shot_id": sh["shot_id"], "n": len(found),
                                 "score": best["score"] if best else None,
                                 "yaw": best["yaw"] if best else None,
                                 "cluster": None})
                for f in found:
                    embeddings.append(f["embedding"])
                    owners.append(idx)
        finally:
            if cap is not None:
                cap.release()
    bar.close(f"{len(per_shot)} shot(s), {len(embeddings)} face(s)")
    took = time.monotonic() - t0

    # The embeddings are kept on disk next to the database, indexed by the
    # shot they came from. Clustering is a threshold decision over them, and
    # this project has twice paid to re-measure something because only the
    # verdict was stored (jerk_ref, then the wind and speech flags). Moving
    # face_same_person_cos is now a re-cluster of a 2 MB array, not a
    # sixteen-minute second look at every frame.
    if embeddings:
        emb_dir = cfg.workdir("faces")
        np.save(emb_dir / "embeddings.npy", np.vstack(embeddings))
        (emb_dir / "embeddings_index.json").write_text(json.dumps(
            {"shot_ids": [per_shot[o]["shot_id"] for o in owners]}))
        log.info("S03.6 %d embedding(s) saved to %s", len(embeddings),
                 emb_dir / "embeddings.npy")

    # pass two: one clustering over the whole corpus, then a merge
    labels = faces_mod.merge_clusters(
        embeddings, faces_mod.cluster(embeddings, threshold=thresh),
        merge_cos=float(cfg.get("process.face_merge_cos", faces_mod.MERGE_COS)))
    named = faces_mod.name_clusters(labels)
    # a shot is attributed to the person it shows most often
    votes: dict[int, dict[int, int]] = {}
    for label, owner in zip(labels, owners):
        votes.setdefault(owner, {})[label] = votes.setdefault(owner, {}).get(label, 0) + 1
    for owner, tally in votes.items():
        per_shot[owner]["cluster"] = max(tally, key=lambda c: (tally[c], -c))

    conn.executemany(
        "UPDATE shots SET has_face=?, face_cluster=?, face_yaw=?, face_score=? "
        "WHERE shot_id=?",
        [(1 if r["n"] else 0,
          named.get(r["cluster"]) if r["cluster"] is not None else None,
          r["yaw"],
          # a shot that was looked at and showed nothing scores 0, not null:
          # null means "not yet looked at", which is what resumes the stage
          r["score"] if r["score"] is not None else 0.0,
          r["shot_id"]) for r in per_shot])
    conn.commit()

    sizes: dict[int, int] = {}
    for c in labels:
        sizes[c] = sizes.get(c, 0) + 1
    with_face = sum(1 for r in per_shot if r["n"])
    log.info("S03.6 %d of %d shot(s) show a face; %d face(s) in %d cluster(s), "
             "%.1f min (%.1f s/shot)",
             with_face, len(per_shot), len(embeddings), len(sizes),
             took / 60, took / max(len(per_shot), 1))
    for c, name in named.items():
        log.info("S03.6 cluster %d -> %s: %d face(s) -- confirm at Gate 2",
                 c, name, sizes.get(c, 0))
    biggest = sorted(sizes.values(), reverse=True)[:6]
    log.info("S03.6 cluster sizes, largest first: %s%s", biggest,
             " (a long tail of ones is normal: strangers, and profiles that "
             "did not match)" if len(sizes) > len(named) else "")
    if failed:
        log.warning("S03.6 %d shot(s) failed: %s", sum(failed.values()), failed)
    return {"n_shots": len(per_shot), "n_faces": len(embeddings),
            "n_with_face": with_face, "n_clusters": len(sizes),
            "named": {str(k): v for k, v in named.items()},
            "cluster_sizes": biggest, "seconds": round(took, 1), "failed": failed}


def recluster_faces(cfg: Config, conn) -> dict[str, Any]:
    """Redo S03.6's clustering from the stored embeddings.

    Seconds against sixteen minutes, because the thresholds it depends on --
    ``face_same_person_cos`` and ``face_merge_cos`` -- are exactly the kind of
    value this corpus keeps moving. Nothing is detected again.
    """
    emb_path = cfg.work_root / "faces" / "embeddings.npy"
    idx_path = cfg.work_root / "faces" / "embeddings_index.json"
    if not emb_path.exists() or not idx_path.exists():
        return {"skipped": "no stored embeddings; run `nepal s03 --redo faces`"}
    # Re-labelling from stored embeddings is seconds; detection is hours. The
    # two must never be confused, and once were: `--redo recluster` on a
    # machine where detection had run *elsewhere* found no `faces` stage unit,
    # decided the expensive step was outstanding, and started a four-hour job
    # to answer a question the .npy already answered.
    if "faces" not in db.done_units(conn, STAGE):
        db.mark_unit(conn, STAGE, "faces",
                     detail="embeddings present; detection ran on another machine")
    E = np.load(emb_path)
    shot_ids = json.loads(idx_path.read_text())["shot_ids"]
    if len(shot_ids) != len(E):
        return {"error": f"index has {len(shot_ids)} ids for {len(E)} embeddings"}

    thresh = float(cfg.get("process.face_same_person_cos", faces_mod.SAME_PERSON_COS))
    merge = float(cfg.get("process.face_merge_cos", faces_mod.MERGE_COS))
    labels = faces_mod.merge_clusters(
        E, faces_mod.cluster(E, threshold=thresh), merge_cos=merge)
    named = faces_mod.name_clusters(labels)

    votes: dict[str, dict[int, int]] = {}
    for label, sid in zip(labels, shot_ids):
        votes.setdefault(sid, {})[label] = votes.setdefault(sid, {}).get(label, 0) + 1
    conn.executemany(
        "UPDATE shots SET face_cluster=? WHERE shot_id=?",
        [(named.get(max(t, key=lambda c: (t[c], -c))), sid) for sid, t in votes.items()])
    conn.commit()

    sizes: dict[int, int] = {}
    for c in labels:
        sizes[c] = sizes.get(c, 0) + 1
    log.info("S03.6 re-clustered %d face(s) at same=%.2f merge=%.2f: %d cluster(s), "
             "largest %s", len(E), thresh, merge, len(sizes),
             sorted(sizes.values(), reverse=True)[:6])
    for c, name in named.items():
        log.info("S03.6 cluster %d -> %s: %d face(s) -- confirm at Gate 2",
                 c, name, sizes.get(c, 0))
    return {"n_faces": len(E), "n_clusters": len(sizes),
            "named": {str(k): v for k, v in named.items()},
            "cluster_sizes": sorted(sizes.values(), reverse=True)[:6]}


def apply_gate(cfg: Config, conn) -> dict[str, Any]:
    """S03.7 -- reject what is not worth a caption, a transcription or a human.

    Runs over every shot each time rather than only the new ones: the gate is
    a pure function of metrics and thresholds, both of which change while the
    film is being tuned, and a stale rejection is invisible -- the shot simply
    never appears again. Cheap enough to redo: one SQL read and no decoding.
    """
    rows = [dict(r) for r in conn.execute(
        "SELECT s.shot_id, s.media_kind, s.start_s, s.end_s, s.sharpness, "
        "s.exposure_pen, s.stability, s.jerk_px, s.has_speech, s.speech_s, "
        "s.wind_lf_share, s.act, "
        "COALESCE(a.quality_curve, ra.quality_curve, 'camera') AS curve "
        "FROM shots s "
        "LEFT JOIN assets a ON a.asset_id = s.asset_id "
        "LEFT JOIN assets ra ON ra.asset_id = ("
        "  SELECT asset_id FROM assets WHERE recording_id = s.recording_id "
        "  ORDER BY chapter_index LIMIT 1)")]
    if not rows:
        return {"n_shots": 0}

    # Stability is a function of the measured jerk and a reference that is
    # tuned against the corpus, so it is re-derived here from the stored jerk
    # rather than trusted from measure time: moving the reference is then a
    # gate re-run (seconds), not a re-measure (the better part of an hour).
    jerk_ref = float(cfg.get("process.metric_jerk_ref_px", metrics_mod.JERK_REF_PX))
    restab: list[tuple[float, str]] = []
    for r in rows:
        if r["jerk_px"] is not None:
            r["stability"] = round(metrics_mod.stability_from_jerk(
                r["jerk_px"], ref=jerk_ref), 4)
            restab.append((r["stability"], r["shot_id"]))
    if restab:
        conn.executemany("UPDATE shots SET stability=? WHERE shot_id=?", restab)

    # Same for the two audio flags, which the speech reprieve reads.
    speech_min = float(cfg.get("process.speech_min_s", audio_mod.SPEECH_MIN_S))
    for r in rows:
        if r["speech_s"] is not None:
            r["has_speech"] = int(audio_mod.has_speech(r["speech_s"], min_s=speech_min))
    refresh_audio_flags(cfg, conn)

    speech_factor = float(cfg.get("gate.speech_sharpness_factor",
                                  gate_mod.SPEECH_SHARPNESS_FACTOR))
    speech_min_s = float(cfg.get("gate.speech_min_duration_s",
                                 gate_mod.SPEECH_MIN_DURATION_S))
    reasons: dict[str, int] = {}
    by_curve: dict[str, list[int]] = {}
    by_act: dict[Any, list[int]] = {}
    updates: list[tuple[str, str]] = []
    unmeasured = 0
    for r in rows:
        if r["media_kind"] == "video" and r["sharpness"] is None:
            unmeasured += 1
        why = gate_mod.verdict(r, cfg.quality_curve(r["curve"]),
                               speech_sharpness_factor=speech_factor,
                               speech_min_duration_s=speech_min_s)
        updates.append(("candidate" if why is None else "rejected", r["shot_id"]))
        if why:
            reasons[why] = reasons.get(why, 0) + 1
        for tally, key in ((by_curve, r["curve"]), (by_act, r["act"])):
            seen = tally.setdefault(key, [0, 0])
            seen[0] += 1
            seen[1] += 1 if why is None else 0

    conn.executemany("UPDATE shots SET status=? WHERE shot_id=?", updates)
    conn.commit()

    kept = sum(1 for st, _ in updates if st == "candidate")
    surviving_by_act = {k: v[1] for k, v in by_act.items()}
    log.info("S03.7 %d of %d shot(s) survive (%.0f%% rejected); reasons %s",
             kept, len(rows), 100.0 * (1 - kept / max(1, len(rows))), reasons)
    for curve, (n, ok) in sorted(by_curve.items()):
        log.info("S03.7 %-9s %d of %d survive (%.0f%% rejected)",
                 curve, ok, n, 100.0 * (1 - ok / max(1, n)))
    log.info("S03.7 survivors per act %s",
             {k: v for k, v in sorted(surviving_by_act.items(), key=lambda kv: (kv[0] is None, kv[0]))})
    if unmeasured:
        log.warning("S03.7 %d video shot(s) have no metrics -- run S03.3 first; "
                    "they were judged on duration alone", unmeasured)
    return {"n_shots": len(rows), "n_kept": kept, "reasons": reasons,
            "by_curve": {k: {"n": n, "kept": ok} for k, (n, ok) in by_curve.items()},
            "survivors_by_act": {str(k): v for k, v in surviving_by_act.items()},
            "n_unmeasured": unmeasured}


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
    gated = 0
    for m in measured:
        if m is None:
            continue
        if m.get("rejected"):
            rejected[m["rejected"]] = rejected.get(m["rejected"], 0) + 1
        else:
            out.append(m)
            gated += m["status"] == "rejected"

    scan.close(f"{len(out)} became shots, {gated} of them below the gate")
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


def run(cfg: Config, *, force: bool = False,
        redo: set[str] | None = None) -> dict[str, Any]:
    conn = db.init(cfg.db_path)
    report: dict[str, Any] = {"stage": STAGE, "started_utc": db.utcnow()}
    done = db.done_units(conn, STAGE)
    report["skipped_stale"] = freshness.warn_if_stale(
        log, conn, STAGE, force=force, rerun_hint="nepal s03 --force")

    steps = [("proxies", lambda: build_proxies(cfg, conn, force=force)),
             ("shots", lambda: detect_shots(cfg, conn)),
             ("photos", lambda: build_photo_shots(cfg, conn)),
             ("metrics", lambda: measure_shots(
                 cfg, conn, force=force or "metrics" in (redo or ()))),
             ("audio", lambda: measure_audio(
                 cfg, conn, force=force or "audio" in (redo or ()))),
             ("asr", lambda: transcribe_shots(
                 cfg, conn, force=force or "asr" in (redo or ()))),
             ("faces", lambda: detect_faces(
                 cfg, conn, force=force or "faces" in (redo or ()))),
             # Cheap, and depends on two thresholds that move, so it is re-run
             # whenever asked rather than remembered as done.
             ("recluster", lambda: recluster_faces(cfg, conn)),
             # The gate is re-run every time: it is pure, it is cheap, and a
             # rejection left over from an older threshold is invisible.
             ("gate", lambda: apply_gate(cfg, conn))]
    # The gate is a pure function of metrics and thresholds, both of which move
    # while the film is being tuned, so it is never skipped as already done.
    always = {"gate"} | set(redo or ())
    if "faces" in always:
        always.add("recluster")          # a fresh detection pass re-clusters itself
    unknown = (redo or set()) - {name for name, _ in steps}
    if unknown:
        raise SystemExit(f"unknown --redo step(s): {', '.join(sorted(unknown))}")
    for name, fn in steps:
        if not force and name in done and name not in always:
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
