"""Command line entry point.

    nepal s01 [--force] [--skip-fov] [--skip-clock]
    nepal s02
    nepal report
    nepal decisions
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="nepal", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default=None, help="path to pipeline.yaml")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("s01", help="probe: manifest, chapters, FOV, clock offsets")
    p1.add_argument("--force", action="store_true", help="recompute completed sub-steps")
    p1.add_argument("--skip-fov", action="store_true", help="leave fov_deg at its fallback")
    p1.add_argument("--skip-clock", action="store_true", help="skip audio cross-correlation")

    p2 = sub.add_parser("s02", help="spine: gps, altitude, places, telegram, music")
    p2.add_argument("--force", action="store_true")
    p2.add_argument("--skip-asr", action="store_true", help="skip round-video transcription")

    sub.add_parser("decisions", help="print the auto-solved decisions table")
    sub.add_parser("report", help="print the chronological table (the Milestone 1 checkpoint)")
    sub.add_parser("doctor", help="check external binaries and optional packages")

    args = ap.parse_args(argv)
    _setup_logging(args.verbose)

    from nepal.config import Config
    cfg = Config.load(args.config)

    if args.cmd == "s01":
        from nepal.stages import s01_probe
        rep = s01_probe.run(cfg, force=args.force, skip_fov=args.skip_fov,
                            skip_clock=args.skip_clock)
        _print_s01(rep)
        return 0

    if args.cmd == "s02":
        from nepal.stages import s02_spine
        rep = s02_spine.run(cfg, force=args.force, skip_asr=args.skip_asr)
        print(json.dumps(rep, indent=2, default=str)[:4000])
        return 0

    if args.cmd == "decisions":
        from nepal import db
        conn = db.init(cfg.db_path)
        rows = db.all_decisions(conn)
        if not rows:
            print("no decisions recorded yet -- run `nepal s01` first")
            return 1
        w = max(len(r["key"]) for r in rows)
        print(f"{'key'.ljust(w)}  {'value':>12}  {'conf':>6}  method")
        print("-" * (w + 40))
        for r in rows:
            conf = f"{r['confidence']:.3f}" if r["confidence"] is not None else "  -  "
            print(f"{r['key'].ljust(w)}  {str(r['value'])[:12]:>12}  {conf:>6}  {r['method'] or ''}")
        conn.close()
        return 0

    if args.cmd == "report":
        from nepal.stages import s02_spine
        return s02_spine.print_chronology(cfg)

    if args.cmd == "doctor":
        return _doctor(cfg)

    return 1


def _print_s01(rep: dict) -> None:
    m, c = rep.get("manifest", {}), rep.get("chapters", {})
    print("\n=== S01 Probe ===")
    if "n_assets" in m:
        print(f"assets      : {m['n_assets']}")
        for src, n in sorted(m.get("by_source", {}).items()):
            print(f"  {src:<16} {n}")
    if "n_recordings" in c:
        print(f"recordings  : {c['n_recordings']} ({c['n_multi_chapter']} multi-chapter)")
        for p in c.get("continuity_problems", [])[:10]:
            print(f"  ! {p}")
    g = rep.get("gps_check", {})
    if g:
        print(f"camera GPS  : {g.get('camera_has_gps')} ({g.get('with_fix')}/{g.get('camera_assets')})")
    f = rep.get("fov", {})
    if "fov_deg" in f:
        flag = "  <-- FALLBACK, confirm at Gate 1" if f.get("used_fallback") else ""
        print(f"fov_deg     : {f['fov_deg']}  conf={f.get('confidence')}  {f.get('method','')}{flag}")
    for label, r in (rep.get("clock") or {}).items():
        if not isinstance(r, dict) or "offset_s" not in r:
            continue
        flag = "  <-- NEEDS MANUAL ENTRY AT GATE 1" if r.get("needs_manual") else ""
        print(f"clock {label:<7}: {r['offset_s']:+.2f}s  conf={r['confidence']}  "
              f"({r['pairs_accepted']}/{r['pairs_total']} pairs){flag}")
    print()


def _doctor(cfg) -> int:
    from nepal.util import proc

    # Where the pipeline will actually read and write. Relative paths in the
    # config resolve against the config file's project root, not the working
    # directory, so this is the same wherever `nepal` is invoked from.
    print(f"config      : {cfg.path}")
    print(f"project root: {cfg.base_dir}")
    print("paths:")
    for label, path, need in [
        ("data_root", cfg.data_root, "your nepal_data/ -- REQUIRED"),
        ("work_root", cfg.work_root, "derived output, created as needed"),
        ("db", cfg.db_path, "pipeline state"),
        ("srtm_dir", cfg.srtm_dir, "elevation tiles (tools/fetch_reference.py)"),
        ("geonames", cfg.geonames_path, "gazetteer (tools/fetch_reference.py)"),
    ]:
        mark = "ok" if path.exists() else "--"
        print(f"  [{mark}] {label:<10} {path}")
        if not path.exists():
            print(f"       {' ' * 10} ^ missing: {need}")
    if cfg.data_root.exists():
        for sub in ("media_from_camera", "media_from_phones", "chat_export", "music"):
            d = cfg.data_root / sub
            n = sum(1 for _ in d.rglob("*")) if d.exists() else 0
            print(f"       {'':<10}   {sub:<20} "
                  f"{'%d files' % n if d.exists() else 'ABSENT'}")
    print()
    print("external binaries:")
    ok = True
    for tool, why in [("ffmpeg", "S01.4 frames, S03 reprojection, S07/S08 render"),
                      ("ffprobe", "S01.1 stream metadata"),
                      ("exiftool", "S01.1 manifest, S02.1 photo GPS")]:
        present = proc.have(tool)
        ok = ok and present
        print(f"  [{'ok' if present else '--'}] {tool:<10} {why}")
    print("python packages:")
    for mod, extra in [("numpy", "core"), ("scipy", "core"), ("PIL", "core"),
                       ("yaml", "core"), ("librosa", "music"), ("soundfile", "music"),
                       ("faster_whisper", "asr"), ("cv2", "vision"),
                       ("open_clip", "semantic"), ("boto3", "cloud")]:
        try:
            __import__(mod)
            print(f"  [ok] {mod:<16} ({extra})")
        except ImportError:
            print(f"  [--] {mod:<16} ({extra})  pip install '.[{extra}]'")
    if not ok:
        print("\ninstall the missing binaries:\n"
              "  apt-get install -y ffmpeg libimage-exiftool-perl")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
