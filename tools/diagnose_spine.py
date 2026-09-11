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
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Nepal's bounding box, generously padded.
NEPAL = (26.0, 80.0, 31.0, 89.0)


def hr(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 66 - len(title)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default=None)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    from nepal.config import Config
    cfg = Config.load(args.config)
    if not cfg.db_path.exists():
        print(f"no database at {cfg.db_path}")
        return 1
    conn = sqlite3.connect(f"file:{cfg.db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

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
        print("  -> Act 4 runs from the FIRST to the LAST of these days. A stray day")
        print("     here stretches Act 4 across everything between and empties Act 5.")

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


if __name__ == "__main__":
    sys.exit(main())
