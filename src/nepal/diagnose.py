#!/usr/bin/env python3
"""Diagnose what S02 actually built, when the checkpoint table looks wrong.

Read-only. Reports the things the chronological table aggregates away: where
each source's timestamps actually land, whether the camera clock is coherent
across the corpus, how much of the GPS track is inside Nepal, and what the
playlist's columns really are.

    python tools/diagnose_spine.py
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# Nepal's bounding box, generously padded.
NEPAL = (26.0, 80.0, 31.0, 89.0)


def hr(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 66 - len(title)))


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--top", type=int, default=25,
                    help="how many music tracks to list (default 25)")
    ap.add_argument("--timestamps", action="store_true",
                    help="why unreachable assets are dated the way they are: "
                         "re-reads a sample of the files and prints every time and "
                         "GPS tag exiftool can see in them")
    ap.add_argument("--sample", type=int, default=4,
                    help="files per outlier day to re-read with --timestamps")


def timestamps_report(cfg, conn, sample: int) -> None:
    """Where the dates on unreachable assets actually came from.

    A clip dated eighteen months after the trek is either real material from
    that date or a camera whose clock reset, and the two are indistinguishable
    from the database alone. The file itself distinguishes them: a camera with a
    GPS fix writes satellite time alongside its own, so if GPSDateTime is there
    the stamp is recoverable, and if it is not, only the clip's position can
    date it.
    """
    from nepal.spine import acts as acts_mod
    from nepal.util import proc

    hr("dates on assets that fall outside every act")
    raw = db_decision(conn, "act_boundaries")
    if not raw:
        print("  no act boundaries recorded -- run `nepal s02` first")
        return
    bounds = [acts_mod.ActBoundary(b["act"], _iso(b["start_utc"]), _iso(b["end_utc"]),
                                  b.get("method", "")) for b in json.loads(raw)]

    rows = [r for r in conn.execute(
        "SELECT s3_key, source, kind, created_at, created_at_utc, lat, lon, "
        "COALESCE(duration_s,0) dur FROM assets WHERE created_at_utc IS NOT NULL "
        "AND kind IN ('video360','video_flat','photo') ORDER BY created_at_utc")]
    lost = [r for r in rows
            if acts_mod.act_for(_iso(r["created_at_utc"]), bounds) is None]
    if not lost:
        print("  none -- every dated asset lands in an act")
        return

    by_day: dict[str, list] = {}
    for r in lost:
        by_day.setdefault(str(r["created_at_utc"])[:10], []).append(r)
    print(f"  {len(lost)} asset(s) across {len(by_day)} day(s):")
    for day, rs in sorted(by_day.items()):
        secs = sum(r["dur"] for r in rs if r["kind"] != "photo")
        withgps = sum(1 for r in rs if r["lat"] is not None)
        srcs = ",".join(sorted({r["source"] for r in rs}))
        print(f"    {day}  {len(rs):>4} asset(s)  {secs/60:>6.1f} min video  "
              f"{withgps:>4} with GPS  source={srcs}")

    if not proc.have("exiftool"):
        print("\n  exiftool is not on PATH -- cannot re-read the files")
        return

    print(f"\n  re-reading up to {sample} file(s) per day. Every time and GPS tag "
          f"exiftool\n  can see is listed; what matters is whether GPSDateTime is "
          f"among them.")
    want = re.compile(r"(date|time|gps)", re.I)
    for day, rs in sorted(by_day.items()):
        for r in rs[:sample]:
            path = cfg.data_root / str(r["s3_key"]).removeprefix("raw/")
            print(f"\n  -- {path.name}  (db: created_at={r['created_at']} "
                  f"-> utc={r['created_at_utc']})")
            if not path.exists():
                print(f"     file not found at {path}")
                continue
            try:
                tags = proc.exiftool_one(path)
            except (proc.ToolFailed, proc.ToolMissing) as exc:
                print(f"     exiftool failed: {exc}")
                continue
            hits = {k: v for k, v in tags.items()
                    if want.search(k) and k not in ("SourceFile",)}
            if not hits:
                print("     no time or GPS tag at all -- the date came from elsewhere")
            for k, v in sorted(hits.items()):
                print(f"     {k:<44} {str(v)[:52]}")
            print("     GPSDateTime present: "
                  f"{'YES -- the stamp is recoverable from satellite time' if any('gpsdatetime' in k.lower() for k in hits) else 'no -- only its position can date it'}")


def db_decision(conn, key: str) -> str | None:
    row = conn.execute("SELECT value FROM decisions WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def _iso(value) -> datetime | None:
    if not value:
        return None
    dt = datetime.fromisoformat(str(value))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def run(cfg, args) -> int:
    if not cfg.db_path.exists():
        print(f"no database at {cfg.db_path}")
        return 1
    conn = sqlite3.connect(f"file:{cfg.db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    if getattr(args, "timestamps", False):
        timestamps_report(cfg, conn, int(getattr(args, "sample", 4)))
        conn.close()
        return 0

    hr("asset time span per source (corrected UTC)")
    for r in conn.execute(
        "SELECT source, COUNT(*) n, MIN(created_at_utc) lo, MAX(created_at_utc) hi "
        "FROM assets WHERE created_at_utc IS NOT NULL GROUP BY source ORDER BY source"
    ):
        print(f"  {r['source']:<15} {r['n']:>5}  {str(r['lo'])[:16]} .. {str(r['hi'])[:16]}")
    n_undated = conn.execute("SELECT COUNT(*) n FROM assets "
                             "WHERE created_at_utc IS NULL").fetchone()["n"]
    print(f"  {'(undated)':<15} {n_undated:>5}")

    hr("camera recordings per corrected day -- is the clock coherent?")
    rows = list(conn.execute(
        "SELECT substr(created_at_utc,1,10) d, COUNT(*) n, MIN(created_at) raw_lo, "
        "MAX(created_at) raw_hi FROM assets WHERE source='camera' "
        "AND created_at_utc IS NOT NULL GROUP BY d ORDER BY d"))
    for r in rows:
        print(f"  {r['d']}  {r['n']:>4} assets   raw device time "
              f"{str(r['raw_lo'])[:16]} .. {str(r['raw_hi'])[:16]}")
    if len(rows) > 1:
        span_days = (rows[-1]["d"], rows[0]["d"])
        print(f"  -> corrected camera dates span {rows[0]['d']} .. {rows[-1]['d']}")
        print("     A camera whose battery died mid-trek resets its clock, so ONE global")
        print("     offset cannot correct both halves. Groups far apart mean exactly that.")

    hr("clock decisions")
    for r in conn.execute("SELECT key, value, confidence, method FROM decisions "
                          "WHERE key LIKE 'clock%' ORDER BY key"):
        print(f"  {r['key']:<30} {str(r['value'])[:20]:<20} conf={r['confidence']}")
        if r["method"]:
            print(f"      {str(r['method'])[:110]}")

    hr("shots placed, per kind -- the context score reads these")
    for r in conn.execute(
        "SELECT media_kind, COUNT(*) n, SUM(lat IS NOT NULL) pos, "
        "SUM(alt_dem_m IS NOT NULL) alt, SUM(place_name IS NOT NULL) named, "
        "SUM(day_index IS NOT NULL) dayed FROM shots WHERE status <> 'rejected' "
        "GROUP BY media_kind ORDER BY media_kind"
    ):
        print(f"  {r['media_kind']:<6} {r['n']:>5} surviving: {r['pos']:>5} positioned, "
              f"{r['alt']:>5} with altitude, {r['named']:>5} named, {r['dayed']:>5} on a trek day")
        if r["n"] and not r["pos"]:
            print("     ^ none positioned: run `nepal s03 --redo place` after `nepal s02`")

    hr("GPS track: how much is actually in Nepal?")
    la0, lo0, la1, lo1 = NEPAL
    tot = conn.execute("SELECT COUNT(*) n FROM gps_points").fetchone()["n"]
    inside = conn.execute(
        "SELECT COUNT(*) n, MIN(ts_utc) lo, MAX(ts_utc) hi FROM gps_points "
        "WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?", (la0, la1, lo0, lo1)).fetchone()
    outside = conn.execute(
        "SELECT COUNT(*) n, MIN(ts_utc) lo, MAX(ts_utc) hi FROM gps_points "
        "WHERE NOT (lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?)",
        (la0, la1, lo0, lo1)).fetchone()
    print(f"  total points     : {tot}")
    print(f"  inside Nepal     : {inside['n']:>5}  {str(inside['lo'])[:16]} .. {str(inside['hi'])[:16]}")
    print(f"  outside Nepal    : {outside['n']:>5}  {str(outside['lo'])[:16]} .. {str(outside['hi'])[:16]}")
    print("  -> the envelope used for message phase classification is the FULL span.")
    print("     Non-trek photos with GPS stretch it, which mislabels months of")
    print("     planning messages as 'trek' and collapses Act 1.")
    for r in conn.execute(
        "SELECT ROUND(lat,1) la, ROUND(lon,1) lo, COUNT(*) n FROM gps_points "
        "WHERE NOT (lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?) "
        "GROUP BY la, lo ORDER BY n DESC LIMIT 8", (la0, la1, lo0, lo1)):
        print(f"       {r['n']:>4} points near {r['la']:.1f}, {r['lo']:.1f}")

    hr("altitude: days at or above 97% of maximum")
    mx = conn.execute("SELECT MAX(alt_dem_m) m FROM assets").fetchone()["m"]
    if mx:
        print(f"  max altitude {mx:.0f} m; 97% threshold {mx*0.97:.0f} m")
        for r in conn.execute(
            "SELECT substr(created_at_utc,1,10) d, MAX(alt_dem_m) a, COUNT(*) n "
            "FROM assets WHERE alt_dem_m >= ? GROUP BY d ORDER BY d", (mx * 0.97,)):
            print(f"    {r['d']}  {r['a']:.0f} m  ({r['n']} assets)")
        print("  -> Act 4 takes the CONTIGUOUS run around the peak, within the trek")
        print("     window, so a day listed here that sits outside the window does not")
        print("     stretch it. A day inside the window but far from the others would.")

    hr("message phases")
    for r in conn.execute("SELECT phase, COUNT(*) n, MIN(ts_utc) lo, MAX(ts_utc) hi "
                          "FROM messages GROUP BY phase"):
        print(f"  {str(r['phase']):<10} {r['n']:>5}  {str(r['lo'])[:16]} .. {str(r['hi'])[:16]}")

    hr("music tracks as parsed")
    for r in conn.execute("SELECT track_id, title, assigned_act, tempo_bpm, energy_mean, "
                          "key_est FROM music_tracks ORDER BY assigned_act, track_id LIMIT ?",
                          (args.top,)):
        act = r["assigned_act"] if r["assigned_act"] is not None else "-"
        print(f"  act {str(act):<3} {str(r['title'])[:34]:<34} "
              f"bpm={r['tempo_bpm']:>6.1f} energy={r['energy_mean']:>5.2f} key={r['key_est']}")
        print(f"          track_id={r['track_id'][:70]}")

    hr("playlist CSV header")
    music_dir = cfg.data_root / "music"
    csvs = sorted(music_dir.glob("*.csv")) if music_dir.exists() else []
    for f in csvs[:2]:
        try:
            first = f.read_text(encoding="utf-8-sig", errors="replace").splitlines()[:2]
        except OSError as exc:
            print(f"  could not read {f.name}: {exc}")
            continue
        print(f"  {f.name}")
        for i, line in enumerate(first):
            print(f"    {'header' if i == 0 else 'row 1 '}: {line[:200]}")

    conn.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    from nepal.config import Config
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default=None)
    add_arguments(ap)
    args = ap.parse_args(argv)
    return run(Config.load(args.config), args)


if __name__ == "__main__":
    sys.exit(main())
