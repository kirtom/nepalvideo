#!/usr/bin/env python3
"""Fetch the two offline reference datasets S02 needs.

Both are free and need no account:

  SRTM elevation   AWS Open Data (elevation-tiles-prod), 1 arc-second .hgt.gz.
                   Roughly 7 MB per tile compressed, 25 MB on disk. Nepal's
                   Everest region is four tiles.
  GeoNames         download.geonames.org country dump. A few megabytes of
                   tab-separated text covering every populated place, peak,
                   pass and glacier in the country.

Which SRTM tiles are needed is derived from the GPS track in the database, so
run this after `nepal s01` and the S02.1 track step. Failing that, pass --bbox.

    python tools/fetch_reference.py
    python tools/fetch_reference.py --bbox 27.6 86.7 28.1 87.0 --country NP
"""
from __future__ import annotations

import argparse
import gzip
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SRTM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi/{lat_dir}/{tile}.hgt.gz"
GEONAMES_URL = "https://download.geonames.org/export/dump/{country}.zip"
UA = {"User-Agent": "nepal-film-pipeline/0.1 (+reference data fetch)"}


def download(url: str, dest: Path, *, timeout: float = 120.0) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as fh:
            shutil.copyfileobj(r, fh)
        tmp.replace(dest)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as exc:
        tmp.unlink(missing_ok=True)
        print(f"  FAILED {url}\n         {exc}")
        return False


def fetch_srtm(tiles: list[str], dest_dir: Path, *, keep_gz: bool = False) -> dict:
    dest_dir.mkdir(parents=True, exist_ok=True)
    got, skipped, failed = [], [], []
    for tile in tiles:
        hgt = dest_dir / f"{tile}.hgt"
        if hgt.exists():
            skipped.append(tile)
            continue
        url = SRTM_URL.format(lat_dir=tile[:3], tile=tile)
        gz = dest_dir / f"{tile}.hgt.gz"
        print(f"  {tile} ...", end="", flush=True)
        if not download(url, gz):
            failed.append(tile)
            continue
        try:
            with gzip.open(gz, "rb") as src, open(hgt, "wb") as out:
                shutil.copyfileobj(src, out)
        except (OSError, gzip.BadGzipFile) as exc:
            print(f" bad archive: {exc}")
            failed.append(tile)
            gz.unlink(missing_ok=True)
            continue
        if not keep_gz:
            gz.unlink(missing_ok=True)
        print(f" {hgt.stat().st_size / 1e6:.1f} MB")
        got.append(tile)
    return {"downloaded": got, "already_present": skipped, "failed": failed}


def fetch_geonames(country: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    txt = dest.parent / f"{country}.txt"
    if txt.exists():
        print(f"  {txt.name} already present ({txt.stat().st_size / 1e6:.1f} MB)")
        return True
    zip_path = dest.parent / f"{country}.zip"
    print(f"  {country}.zip ...", end="", flush=True)
    if not download(GEONAMES_URL.format(country=country), zip_path):
        return False
    import zipfile
    try:
        with zipfile.ZipFile(zip_path) as z:
            member = next((n for n in z.namelist()
                           if n.lower().endswith(".txt") and "readme" not in n.lower()), None)
            if member is None:
                print(" no data member in archive")
                return False
            with z.open(member) as src, open(txt, "wb") as out:
                shutil.copyfileobj(src, out)
    except zipfile.BadZipFile as exc:
        print(f" bad archive: {exc}")
        return False
    zip_path.unlink(missing_ok=True)
    print(f" {txt.stat().st_size / 1e6:.1f} MB")
    return True


BBox = tuple[float, float, float, float]

# A trek is at most a couple of hundred kilometres end to end. Anything much
# further from the centre of the fixes is a photo from home, an airport, or a
# stale cached location -- not part of the route.
DEFAULT_MAX_RADIUS_KM = 200.0

# Refuse to fetch more than this without an explicit override. Four tiles covers
# a Nepal trek; a hundred means the extent is wrong, and silently downloading
# 2.6 GB is worse than stopping.
DEFAULT_MAX_TILES = 12


def robust_bbox(points: Sequence[tuple[float, float]], *,
                max_radius_km: float = DEFAULT_MAX_RADIUS_KM
                ) -> tuple[BBox, int, float] | None:
    """Bounding box of the trek, ignoring fixes that are not on it.

    Taking a raw min/max over every GPS fix is maximally sensitive to a single
    outlier, and a corpus like this is full of them: planning photos taken at
    home, an airport layover, a stale cached location from the last place with
    signal. One photo from Dubai turns a four-tile Everest box into 132 tiles
    and 3.4 GB; one from Berlin gives 1,950 tiles and 49 GB.

    So the centre is taken as the median of the fixes -- which no outlier can
    move -- and points beyond ``max_radius_km`` of it are dropped before the
    extent is measured. That preserves genuine one-off extremes like the summit
    push, which a percentile clip would discard.

    Returns (bbox, n_dropped, furthest_dropped_km).
    """
    import statistics
    from nepal.spine.gps import haversine_m

    pts = [(float(la), float(lo)) for la, lo in points
           if la is not None and lo is not None and not (la == 0 and lo == 0)]
    if not pts:
        return None

    c_lat = statistics.median([p[0] for p in pts])
    c_lon = statistics.median([p[1] for p in pts])

    kept, dropped_km = [], []
    for la, lo in pts:
        d_km = haversine_m(c_lat, c_lon, la, lo) / 1000.0
        if d_km <= max_radius_km:
            kept.append((la, lo))
        else:
            dropped_km.append(d_km)

    if not kept:
        return None
    lats = [p[0] for p in kept]
    lons = [p[1] for p in kept]
    return ((min(lats), min(lons), max(lats), max(lons)),
            len(dropped_km), max(dropped_km) if dropped_km else 0.0)


def points_from_db(db_path: Path) -> tuple[list[tuple[float, float]], str] | None:
    """Every GPS fix in the database.

    gps_points first (written by S02.1), then the assets table, whose lat/lon
    come straight from photo EXIF and are therefore populated by S01. Depending
    only on gps_points created a deadlock: the tiles are needed to compute
    altitude in S02, but the track that says which tiles to fetch is only
    written by S02.
    """
    if not db_path.exists():
        return None
    import sqlite3
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        for table, label in (("gps_points", "GPS track"), ("assets", "photo EXIF")):
            try:
                rows = conn.execute(
                    f"SELECT lat, lon FROM {table} "
                    "WHERE lat IS NOT NULL AND lon IS NOT NULL").fetchall()
            except sqlite3.Error:
                continue
            if rows:
                return [(r[0], r[1]) for r in rows], label
    finally:
        conn.close()
    return None


def points_from_exif(data_root: Path) -> tuple[list[tuple[float, float]], str] | None:
    """GPS fixes read straight off the phone photos.

    Makes this tool independent of pipeline order: it works before anything has
    been run, on nothing but the delivered tree.
    """
    import shutil
    import subprocess
    if not shutil.which("exiftool") or not data_root.exists():
        return None
    target = data_root / "media_from_phones"
    if not target.exists():
        target = data_root
    print(f"  reading GPS from {target} ...", end="", flush=True)
    try:
        proc = subprocess.run(
            ["exiftool", "-q", "-r", "-n", "-if", "$GPSLatitude",
             "-p", "$GPSLatitude,$GPSLongitude", str(target)],
            capture_output=True, text=True, timeout=1800)
    except (subprocess.TimeoutExpired, OSError) as exc:
        print(f" failed: {exc}")
        return None

    pts = []
    for line in proc.stdout.splitlines():
        parts = line.strip().split(",")
        if len(parts) != 2:
            continue
        try:
            pts.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    if not pts:
        print(" no fixes found")
        return None
    print(f" {len(pts)} fixes")
    return pts, f"{len(pts)} phone photos"


def prune_tiles(keep: Sequence[str], tile_dir: Path, *, dry_run: bool) -> list[str]:
    """Remove tiles outside the needed set -- cleanup after an over-large fetch."""
    if not tile_dir.exists():
        return []
    keeping = set(keep)
    victims = sorted(f for f in tile_dir.glob("*.hgt") if f.stem not in keeping)
    for f in victims:
        if not dry_run:
            f.unlink(missing_ok=True)
    return [f.stem for f in victims]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default=None)
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("MIN_LAT", "MIN_LON",
                                                            "MAX_LAT", "MAX_LON"))
    ap.add_argument("--country", default="NP", help="GeoNames country code (default NP)")
    ap.add_argument("-n", "--dry-run", action="store_true",
                    help="report which tiles are needed and their size, download nothing")
    ap.add_argument("--skip-srtm", action="store_true")
    ap.add_argument("--skip-geonames", action="store_true")
    ap.add_argument("--pad", type=float, default=0.1,
                    help="degrees of padding around the track bbox (default 0.1)")
    ap.add_argument("--max-radius-km", type=float, default=DEFAULT_MAX_RADIUS_KM,
                    help="reject GPS fixes further than this from the route centre "
                         f"(default {DEFAULT_MAX_RADIUS_KM:.0f})")
    ap.add_argument("--max-tiles", type=int, default=DEFAULT_MAX_TILES,
                    help=f"refuse to fetch more than this many tiles "
                         f"(default {DEFAULT_MAX_TILES})")
    ap.add_argument("--prune", action="store_true",
                    help="delete tiles outside the needed set (cleans up an "
                         "over-large earlier fetch)")
    args = ap.parse_args()

    from nepal.config import Config
    from nepal.spine.dem import tiles_for_bbox
    cfg = Config.load(args.config)

    srtm_dir = cfg.srtm_dir
    geo_path = cfg.geonames_path

    rc = 0
    if not args.skip_srtm:
        found = None
        if args.bbox:
            found = (tuple(args.bbox), "--bbox")
        else:
            pts = points_from_db(cfg.db_path) or points_from_exif(cfg.data_root)
            if pts:
                reduced = robust_bbox(pts[0], max_radius_km=args.max_radius_km)
                if reduced:
                    bbox, n_dropped, furthest = reduced
                    if n_dropped:
                        print(f"  ignored {n_dropped} fix(es) more than "
                              f"{args.max_radius_km:.0f} km from the route centre "
                              f"(furthest {furthest:,.0f} km) -- photos from home or "
                              f"in transit, not part of the trek")
                    found = (bbox, f"{pts[1]}, outliers rejected")
        if found is None:
            print("could not determine the trek extent. Tried: gps_points and assets "
                  f"in {cfg.db_path}, then phone photo EXIF under {cfg.data_root}.\n"
                  "Pass --bbox MIN_LAT MIN_LON MAX_LAT MAX_LON, or check that "
                  "project.data_root in the config is correct (`nepal doctor`).")
            rc = 1
        else:
            bbox, via = found
            print(f"  extent from {via}")
            lo_la, lo_lo, hi_la, hi_lo = bbox
            tiles = tiles_for_bbox(lo_la - args.pad, lo_lo - args.pad,
                                   hi_la + args.pad, hi_lo + args.pad)
            print(f"SRTM tiles for bbox {lo_la:.3f},{lo_lo:.3f} .. "
                  f"{hi_la:.3f},{hi_lo:.3f} (pad {args.pad}): {len(tiles)}")
            if len(tiles) > args.max_tiles:
                print(f"\n  REFUSING: {len(tiles)} tiles is {len(tiles)*26/1024:.1f} GB, "
                      f"above --max-tiles {args.max_tiles}.\n"
                      f"  A Nepal trek needs 2-4. An extent this large means the GPS "
                      f"fixes span more than the route --\n"
                      f"  check with --dry-run, narrow it with --max-radius-km, or set "
                      f"it explicitly with --bbox.")
                return 1

            present = [t for t in tiles if (srtm_dir / f"{t}.hgt").exists()]
            missing = [t for t in tiles if t not in present]
            for t in tiles:
                mark = "have" if t in present else "need"
                print(f"    [{mark}] {t}")
            print(f"  {len(missing)} to download, about {len(missing) * 7} MB over the wire, "
                  f"{len(missing) * 26} MB on disk")
            if args.dry_run:
                print("  (dry run -- nothing downloaded)")
                res = {"downloaded": [], "already_present": present, "failed": []}
            else:
                res = fetch_srtm(tiles, srtm_dir)
            print(f"  -> {len(res['downloaded'])} downloaded, "
                  f"{len(res['already_present'])} already present, "
                  f"{len(res['failed'])} failed")
            if res["failed"]:
                rc = 1
            if args.prune:
                removed = prune_tiles(tiles, srtm_dir, dry_run=args.dry_run)
                if removed:
                    verb = "would remove" if args.dry_run else "removed"
                    print(f"  {verb} {len(removed)} unneeded tile(s), "
                          f"{len(removed)*26/1024:.1f} GB: "
                          f"{', '.join(removed[:8])}"
                          f"{' ...' if len(removed) > 8 else ''}")
                else:
                    print("  no unneeded tiles to remove")

    if not args.skip_geonames:
        print(f"GeoNames {args.country} -> {geo_path.parent}")
        if args.dry_run:
            txt = geo_path.parent / f"{args.country}.txt"
            print(f"  [{'have' if txt.exists() else 'need'}] {args.country}.txt"
                  f"  (about 5 MB zipped)")
        elif not fetch_geonames(args.country, geo_path):
            rc = 1

    if not args.dry_run:
        print("\nAltitude and place names will populate on the next `nepal s02 --force`.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
