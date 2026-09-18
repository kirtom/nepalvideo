"""Command line entry point.

    nepal doctor                 check binaries, packages and resolved paths
    nepal s01 [--force] [--skip-fov] [--skip-clock] [--redo UNITS]
    nepal fetch-reference        SRTM tiles + GeoNames gazetteer
    nepal s02 [--force] [--skip-asr] [--redo UNITS]
    nepal s03 [--force] [--redo STEPS]   per-clip processing
    nepal s04 [--force] [--redo STEPS]   semantic layer (CLIP embeddings)
    nepal cut [--redo score,timeline,draft]  score, assemble, render the draft
    nepal fov-check              Gate 1 seam comparison sheets
    nepal report                 the chronological checkpoint table
    nepal decisions              auto-solved values with confidence
    nepal diagnose               when the checkpoint table looks wrong
    nepal prune [--dry-run]      remove what the pipeline no longer produces
    nepal remote up|down|status|push|pull|ssh [--gpu]
    nepal remote run <nepal args> / exec -- <shell command>   on the GCP box

Everything runs through this one entry point on purpose: a console script uses
the interpreter the package was installed into, so it cannot pick up a
different environment the way `python some_script.py` can.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _setup_logging(verbose: bool) -> None:
    # A progress line is redrawn in place, so a log record written straight to
    # stderr lands in the middle of it. This handler wipes the line first and
    # puts it back afterwards.
    from nepal.util.progress import ProgressAwareHandler
    handler = ProgressAwareHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s | %(message)s", datefmt="%H:%M:%S"))
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        handlers=[handler])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="nepal", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default=None, help="path to pipeline.yaml")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--no-progress", action="store_true",
                    help="never draw progress lines (also: NEPAL_NO_PROGRESS=1)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("s01", help="probe: manifest, chapters, FOV, clock offsets")
    p1.add_argument("--force", action="store_true", help="recompute completed sub-steps")
    p1.add_argument("--skip-fov", action="store_true", help="leave fov_deg at its fallback")
    p1.add_argument("--skip-clock", action="store_true", help="skip audio cross-correlation")
    p1.add_argument("--redo", metavar="UNITS", default="",
                    help="comma-separated sub-steps to recompute: manifest,chapters,fov,clock")

    p2 = sub.add_parser("s02", help="spine: gps, altitude, places, telegram, music")
    p2.add_argument("--force", action="store_true")
    p2.add_argument("--skip-asr", action="store_true", help="skip round-video transcription")
    p2.add_argument("--redo", metavar="UNITS", default="",
                    help="comma-separated sub-steps to recompute: "
                         "gps_track,telegram,geotag,asr,music,acts")

    pf = sub.add_parser("fetch-reference",
                        help="download SRTM elevation tiles and the GeoNames gazetteer")
    from nepal import reference as _reference
    _reference.add_arguments(pf)

    pd = sub.add_parser("diagnose",
                        help="report what S02 built, when the checkpoint looks wrong")
    from nepal import diagnose as _diagnose
    _diagnose.add_arguments(pd)

    p3 = sub.add_parser("s03", help="S03 -- per-clip processing")
    p3.add_argument("--force", action="store_true", help="recompute completed sub-steps")
    # --force on S03 means rebuilding every proxy: three and a half hours. The
    # cheap steps after it are the ones that get re-run while the film is being
    # tuned, so they are reachable without paying for the expensive one.
    p3.add_argument("--redo", metavar="STEPS", default="",
                    help="comma-separated sub-steps to recompute: "
                         "proxies,shots,photos,place,metrics,audio,asr,hallucination,"
                         "faces,recluster,gate")

    p4 = sub.add_parser("s04", help="S04 -- semantic layer (CLIP embeddings)")
    p4.add_argument("--force", action="store_true", help="recompute completed sub-steps")
    p4.add_argument("--redo", metavar="STEPS", default="",
                    help="comma-separated sub-steps to recompute: embeddings")

    pcut = sub.add_parser("cut", help="S05-S07 -- score, assemble and render the draft")
    pcut.add_argument("--force", action="store_true")
    pcut.add_argument("--redo", metavar="STEPS", default="score,timeline,draft",
                      help="comma-separated: score,timeline,draft (default: all)")

    pfc = sub.add_parser("fov-check",
                         help="render the Gate 1 FOV comparison sheets")
    pfc.add_argument("--frames", type=int, default=2,
                     help="how many source frames to compare (default 2)")
    pfc.add_argument("--width", type=int, default=2048,
                     help="equirect width to render at (default 2048)")
    pfc.add_argument("--proxy", action="store_true",
                     help="use .lrv proxies instead of full-resolution source")

    sub.add_parser("decisions", help="print the auto-solved decisions table")
    sub.add_parser("report", help="print the chronological table (the Milestone 1 checkpoint)")
    sub.add_parser("doctor", help="check external binaries and optional packages")

    pprune = sub.add_parser("prune", help="remove recordings, shots and work files "
                                          "the pipeline no longer produces")
    pprune.add_argument("--dry-run", action="store_true",
                        help="report what would go without touching anything")

    sub.add_parser("status-page", help="write work/status/{status.json,index.html}: "
                                       "where the pipeline is, from the database and the reports")

    prem = sub.add_parser("remote", help="run things on the GCP box (Film v2 section 2.2)")
    prem.add_argument("action", choices=["up", "down", "status", "push", "pull", "run",
                                         "exec", "ssh"])
    prem.add_argument("--gpu", action="store_true", help="the GPU profile instead of cpu")
    prem.add_argument("--delete", action="store_true", help="down: delete rather than stop")
    prem.add_argument("--no-wait", action="store_true", help="up: do not wait for READY")
    prem.add_argument("rest", nargs=argparse.REMAINDER,
                      help="run: nepal arguments; exec: a shell command (after --)")

    args = ap.parse_args(argv)
    if getattr(args, "no_progress", False):
        import os
        os.environ["NEPAL_NO_PROGRESS"] = "1"
    _setup_logging(args.verbose)

    from nepal.config import Config, ConfigNotFound
    try:
        cfg = Config.load(args.config)
    except ConfigNotFound as exc:
        raise SystemExit(str(exc))
    # Which file the run is reading is worth one line: the config can now be
    # found in several places, and "that tunable had no effect" is a much
    # harder thing to diagnose than "it read a different file than you edited".
    logging.getLogger("nepal").info("config %s", cfg.path)

    if args.cmd == "s01":
        from nepal.stages import s01_probe
        rep = s01_probe.run(cfg, force=args.force, skip_fov=args.skip_fov,
                            skip_clock=args.skip_clock,
                            redo={x.strip() for x in args.redo.split(",") if x.strip()})
        _print_s01(rep)
        return 0

    if args.cmd == "s02":
        from nepal.stages import s02_spine
        rep = s02_spine.run(cfg, force=args.force, skip_asr=args.skip_asr,
                            redo={x.strip() for x in args.redo.split(",") if x.strip()})
        print(json.dumps(rep, indent=2, default=str)[:4000])
        return 0

    if args.cmd == "fetch-reference":
        from nepal import reference
        return reference.run(cfg, args)

    if args.cmd == "diagnose":
        from nepal import diagnose
        return diagnose.run(cfg, args)

    if args.cmd == "s03":
        from nepal.stages import s03_process
        redo = {x.strip() for x in args.redo.split(",") if x.strip()}
        rep = s03_process.run(cfg, force=args.force, redo=redo)
        ph = rep.get("photos") or {}
        if ph.get("error"):
            print(f"S03.0 {ph['error']}")
            return 1
        if not ph.get("skipped"):
            print(f"\n=== S03.0 photo shots ===")
            print(f"dated photos : {ph.get('n_photos', 0)}")
            print(f"became shots : {ph.get('n_shots', 0)}")
            for act in sorted(ph.get("per_act") or {}):
                print(f"      act {act} : {ph['per_act'][act]}")
            for reason, n in sorted((ph.get("rejected") or {}).items(),
                                    key=lambda kv: -kv[1]):
                print(f"  {n:>5} rejected: {reason}")
        return 0

    if args.cmd == "s04":
        from nepal.stages import s04_semantic
        redo = {x.strip() for x in args.redo.split(",") if x.strip()}
        rep = s04_semantic.run(cfg, force=args.force, redo=redo)
        print(json.dumps(rep, indent=2, default=str)[:2000])
        return 0

    if args.cmd == "cut":
        from nepal.stages import s05_cut
        redo = {x.strip() for x in args.redo.split(",") if x.strip()}
        rep = s05_cut.run(cfg, force=args.force, redo=redo)
        print(json.dumps(rep, indent=2, default=str)[:2500])
        return 0

    if args.cmd == "fov-check":
        import json as _json
        from nepal import db
        from nepal.probe import fov as _fov
        from nepal.stages import s01_probe
        conn = db.init(cfg.db_path)
        # the curve the solve already measured, so the numbers and the pictures
        # are read side by side rather than one standing in for the other
        rep = cfg.work_root / "reports" / "s01_probe.json"
        if rep.exists():
            try:
                scores = _json.loads(rep.read_text()).get("fov", {}).get("scores") or {}
            except (OSError, ValueError):
                scores = {}
            if scores:
                print("seam discontinuity by candidate "
                      "(1.0 = the join is invisible):")
                print("\n".join(_fov.score_curve({int(k): float(v)
                                                  for k, v in scores.items()})))
        res = s01_probe.fov_thumbnails(cfg, conn, n_frames=args.frames,
                                       width=args.width, prefer_proxy=args.proxy)
        conn.close()
        if res.get("error"):
            print(f"\ncould not render: {res['error']}")
            return 1
        print(f"\n{len(res['sheets'])} sheet(s) from {res['n_frames']} frame(s), "
              f"{res['source']} at {res['width']}px:")
        for f in res["sheets"]:
            print(f"  {f}")
        print("\nEach sheet stacks the candidates over one seam meridian. Pick the\n"
              "row where the vertical join disappears -- look for doubled or\n"
              "sliced detail, not for sharpness. Then set probe.fov.fallback_deg\n"
              "in the config, or record it with:\n"
              "  nepal decisions   (to see what is stored now)")
        return 0

    if args.cmd == "status-page":
        from nepal.cloud import status as status_mod
        print(status_mod.write_status(cfg))
        return 0

    if args.cmd == "remote":
        from nepal.cloud import remote as remote_mod, spend
        r = remote_mod.Remote(cfg, "gpu" if args.gpu else "cpu")
        rest = [a for a in args.rest if a != "--"]
        if args.action == "status":
            st = r.status()
            led = spend.ledger(cfg)
            print(f"{r.profile.name}: {st.state}{' at ' + st.ip if st.ip else ''}; "
                  f"ledger {led.total():.2f} of {cfg.get('cloud.spend_ceiling_usd')} USD")
            return 0
        if args.action == "up":
            st = r.up(wait=not args.no_wait)
            print(f"{r.profile.name}: {st.state} at {st.ip}")
            return 0
        if args.action == "down":
            r.down(delete=args.delete)
            return 0
        if args.action == "push":
            r.push()
            return 0
        if args.action == "pull":
            r.pull()
            return 0
        if args.action == "run":
            return r.run_nepal(rest)
        if args.action == "exec":
            return r.exec_cmd(" ".join(rest))
        if args.action == "ssh":
            return r.ssh()

    if args.cmd == "prune":
        from nepal import prune
        rep = prune.run(cfg, dry_run=args.dry_run)
        tag = "would remove" if rep["dry_run"] else "removed"
        print(f"prune: {tag} {rep['n_recordings']} recording(s), {rep['n_shots']} shot(s), "
              f"{rep['n_files']} file(s), {rep['bytes'] / 1e6:.0f} MB")
        # reasons cover both pruned recordings and orphan proxies on disk
        for rid, why in rep["reasons"].items():
            print(f"  {rid}: {why}")
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


def _git_head(path) -> str:
    """`<short sha> <subject>` for the tree containing path, plus a dirty mark."""
    import subprocess
    def _git(*args: str) -> str:
        try:
            r = subprocess.run(["git", "-C", str(path), *args],
                               capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            return ""
        return r.stdout.strip() if r.returncode == 0 else ""

    sha = _git("rev-parse", "--short", "HEAD")
    if not sha:
        return ""
    subject = _git("log", "-1", "--format=%s")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    dirty = " +local changes" if _git("status", "--porcelain") else ""
    return f"{sha} ({branch}){dirty}  {subject[:60]}"


def _doctor(cfg) -> int:
    import pathlib
    import shutil
    import sys as _sys
    from nepal.util import proc

    # An environment split is the likeliest cause of a confusing ImportError:
    # `nepal` is a console script bound to the interpreter it was installed
    # into, while a bare `python` picks up whatever is active. Showing both
    # makes the mismatch obvious instead of surfacing as "No module named yaml".
    print(f"interpreter : {_sys.executable}")
    print(f"python      : {_sys.version.split()[0]}")
    which_python = shutil.which("python") or shutil.which("python3")
    if which_python:
        import subprocess
        try:
            other = subprocess.run([which_python, "-c", "import sys; print(sys.executable)"],
                                   capture_output=True, text=True, timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            other = ""
        # Compare resolved targets: python and python3 are commonly symlinks to
        # the same binary, and warning about that is a false alarm.
        def _real(path: str) -> str:
            try:
                return str(pathlib.Path(path).resolve())
            except OSError:
                return path

        if other and _real(other) != _real(_sys.executable):
            print(f"  NOTE  `python` on your PATH is a DIFFERENT interpreter:\n"
                  f"        {other}\n"
                  f"        Run everything as `nepal ...` rather than "
                  f"`python tools/...`, or the two environments will disagree "
                  f"about which packages exist.")

    # Which copy of the source is actually running. A non-editable install
    # copies the package into site-packages, so `git pull` changes the checkout
    # and nothing else -- the fix is pulled but not running, and every report
    # looks unchanged for reasons no amount of re-running will reveal.
    import nepal as _pkg
    pkg_dir = pathlib.Path(_pkg.__file__).resolve().parent
    print(f"package     : {pkg_dir}")
    repo_src = (cfg.base_dir / "src" / "nepal").resolve()
    if pkg_dir == repo_src:
        print(f"              (editable -- this checkout is what runs)")
    else:
        print(f"  WARN  the running code is a COPY, not this checkout:\n"
              f"        checkout    {repo_src}\n"
              f"        `git pull` will not change what `nepal` runs. Either\n"
              f"        reinstall after every pull, or install once as editable:\n"
              f"          pip install -e .")
    head = _git_head(repo_src)
    if head:
        print(f"checkout    : {head}")
    print()

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
                       ("yaml", "core"), ("pillow_heif", "core -- .heic photos"),
                       ("librosa", "music"), ("soundfile", "music"),
                       ("faster_whisper", "asr"), ("cv2", "vision"),
                       ("scenedetect", "vision -- S03.2 shot detection"),
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
