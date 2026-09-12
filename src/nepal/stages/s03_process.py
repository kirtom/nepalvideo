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
from pathlib import Path
from typing import Any

from nepal import db, freshness
from nepal.config import Config
from nepal.process import stills
from nepal.spine import acts as acts_mod
from nepal.util.progress import Progress

log = logging.getLogger(__name__)
STAGE = "S03"


def _dt(value: Any):
    from datetime import datetime, timezone
    if not value:
        return None
    d = datetime.fromisoformat(str(value))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def build_photo_shots(cfg: Config, conn) -> dict[str, Any]:
    """S03.0 -- one shot per photograph that clears its source's quality gate.

    A photo shot has no recording: it is a single asset held on screen, so
    ``shots.asset_id`` is set and ``recording_id`` is NULL. Stability and motion
    stay NULL rather than taking a flattering default -- a still would beat
    every clip on stability by virtue of not moving, which is not a fact about
    its quality.
    """
    from PIL import Image

    bounds_raw = db.get_decision(conn, "act_boundaries")
    bounds = []
    if bounds_raw:
        bounds = [acts_mod.ActBoundary(b["act"], _dt(b["start_utc"]),
                                       _dt(b["end_utc"]), b.get("method", ""))
                  for b in json.loads(bounds_raw)]

    rows = [dict(r) for r in conn.execute(
        "SELECT asset_id, s3_key, source, quality_curve, created_at_utc, "
        "lat, lon, alt_dem_m, place_name, width, height FROM assets "
        "WHERE kind='photo' AND created_at_utc IS NOT NULL")]
    if not rows:
        return {"n_photos": 0, "error": "no dated photos"}

    root = cfg.data_root
    max_px = int(cfg.get("process.photo_analysis_px", 1024))
    out: list[dict[str, Any]] = []
    rejected: dict[str, int] = {}
    unplaced = 0

    scan = Progress("S03.0 measuring photographs", len(rows))
    for r in rows:
        scan.step()
        # A photo outside every act cannot be a slot, so measuring it is waste.
        act = acts_mod.act_for(_dt(r["created_at_utc"]), bounds) if bounds else None
        if act is None:
            unplaced += 1
            continue
        src = root / Path(r["s3_key"]).relative_to("raw")
        ext = src.suffix.lower()
        if not src.exists():
            rejected["missing file"] = rejected.get("missing file", 0) + 1
            continue
        try:
            with Image.open(src) as im:
                im.draft("RGB", (max_px, max_px))   # let the JPEG decoder downscale
                im = im.convert("RGB")
                im.thumbnail((max_px, max_px))
                import numpy as np
                arr = np.asarray(im)
                w, h = im.size
        except (OSError, ValueError) as exc:
            # Name the format. "unreadable" as a single bucket hid that 339 of
            # 706 photographs were HEIC and simply had no decoder installed --
            # a fixable one-line problem reported as an unexplained loss.
            log.debug("could not read %s: %s", src.name, exc)
            if ext in stills.HEIF_EXT and not stills.heif_available():
                key = f"{ext} needs a decoder (pip install pillow-heif)"
            else:
                key = f"unreadable {ext or 'file'} ({type(exc).__name__})"
            rejected[key] = rejected.get(key, 0) + 1
            continue

        sharp = stills.sharpness(arr)
        pen = stills.exposure_penalty(arr)
        curve = cfg.quality_curve(r["quality_curve"] or "phone")
        if not stills.passes_gate(sharp, pen, curve):
            rejected["below the quality gate"] = \
                rejected.get("below the quality gate", 0) + 1
            continue

        dur = stills.slot_duration_s((w / h) if h else None,
                                     base_s=float(cfg.get("process.photo_slot_s")))
        out.append({
            "shot_id": f"photo_{r['asset_id'][:16]}",
            "recording_id": None,
            "asset_id": r["asset_id"],
            "media_kind": "photo",
            "start_s": 0.0,
            "end_s": round(dur, 3),
            "start_utc": r["created_at_utc"],
            "act": act,
            "lat": r["lat"], "lon": r["lon"], "alt_dem_m": r["alt_dem_m"],
            "place_name": r["place_name"],
            "sharpness": round(sharp, 4),
            "exposure_pen": round(pen, 4),
            "stability": None,          # a still does not shake; that is not merit
            "motion_mag": None,
            "audio_lufs": None,
            "view_kind": "still",
            "score_tech": round(stills.technical_score(sharp, pen), 4),
            "status": "candidate",
        })

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

    for name, fn in [("photos", lambda: build_photo_shots(cfg, conn))]:
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
