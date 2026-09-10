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


def bbox_from_db(db_path: Path) -> tuple[float, float, float, float] | None:
    if not db_path.exists():
        return None
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT MIN(lat), MIN(lon), MAX(lat), MAX(lon) FROM gps_points "
                       "WHERE lat IS NOT NULL AND lon IS NOT NULL").fetchone()
    conn.close()
    return tuple(row) if row and row[0] is not None else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default=None)
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("MIN_LAT", "MIN_LON",
                                                            "MAX_LAT", "MAX_LON"))
    ap.add_argument("--country", default="NP", help="GeoNames country code (default NP)")
    ap.add_argument("--skip-srtm", action="store_true")
    ap.add_argument("--skip-geonames", action="store_true")
    ap.add_argument("--pad", type=float, default=0.1,
                    help="degrees of padding around the track bbox (default 0.1)")
    args = ap.parse_args()

    from nepal.config import Config
    from nepal.spine.dem import tiles_for_bbox
    cfg = Config.load(args.config)

    srtm_dir = Path(cfg.get("spine.srtm_dir", "./data/srtm")).expanduser()
    geo_path = Path(cfg.get("spine.geonames_path", "./data/geonames/NP.txt")).expanduser()

    rc = 0
    if not args.skip_srtm:
        bbox = tuple(args.bbox) if args.bbox else bbox_from_db(cfg.db_path)
        if bbox is None:
            print("no GPS track in the database and no --bbox given; "
                  "run `nepal s01` and `nepal s02` first, or pass --bbox")
            rc = 1
        else:
            lo_la, lo_lo, hi_la, hi_lo = bbox
            tiles = tiles_for_bbox(lo_la - args.pad, lo_lo - args.pad,
                                   hi_la + args.pad, hi_lo + args.pad)
            print(f"SRTM tiles for bbox {lo_la:.3f},{lo_lo:.3f} .. "
                  f"{hi_la:.3f},{hi_lo:.3f} (pad {args.pad}): {len(tiles)}")
            res = fetch_srtm(tiles, srtm_dir)
            print(f"  -> {len(res['downloaded'])} downloaded, "
                  f"{len(res['already_present'])} already present, "
                  f"{len(res['failed'])} failed")
            if res["failed"]:
                rc = 1

    if not args.skip_geonames:
        print(f"GeoNames {args.country} -> {geo_path.parent}")
        if not fetch_geonames(args.country, geo_path):
            rc = 1

    print("\nAltitude and place names will populate on the next `nepal s02 --force`.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
