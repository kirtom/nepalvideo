import sys, pathlib, sqlite3
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from tools.fetch_reference import (points_from_db, points_from_exif, robust_bbox,
                                   prune_tiles, DEFAULT_MAX_RADIUS_KM)
from nepal.spine.dem import tiles_for_bbox
from nepal import db

# a plausible Khumbu route
TREK = [(27.6869, 86.7314), (27.8069, 86.7133), (27.8361, 86.7644),
        (27.8917, 86.8250), (27.9500, 86.8100), (28.0026, 86.8528),
        (27.9950, 86.8280)]


def _db(tmp_path, *, gps=(), assets=()):
    p = tmp_path / "nepal.sqlite"
    conn = db.init(p)
    for i, (la, lo) in enumerate(gps):
        conn.execute("INSERT INTO gps_points(ts_utc, lat, lon) VALUES (?,?,?)",
                     (f"2024-04-27T00:{i // 60:02d}:{i % 60:02d}+00:00", la, lo))
    for i, (la, lo) in enumerate(assets):
        conn.execute("INSERT INTO assets(asset_id, s3_key, source, kind, lat, lon, has_gps) "
                     "VALUES (?,?,?,?,?,?,1)",
                     (f"a{i}", f"raw/x{i}.jpg", "phone_keller", "photo", la, lo))
    conn.commit()
    conn.close()
    return p


# -- the outlier bug ---------------------------------------------------

def test_trek_only_gives_a_small_box():
    bbox, dropped, _ = robust_bbox(TREK)
    assert dropped == 0
    assert len(tiles_for_bbox(*[bbox[0]-0.1, bbox[1]-0.1, bbox[2]+0.1, bbox[3]+0.1])) <= 4


@pytest.mark.parametrize("name,stray,raw_tiles_at_least", [
    ("Dubai layover", (25.2522, 55.3644), 100),
    ("Moscow", (55.7558, 37.6176), 1000),
    ("Berlin", (52.5200, 13.4050), 1000),
])
def test_a_single_stray_fix_is_rejected(name, stray, raw_tiles_at_least):
    """The bug this guards: one photo from an airport or from home turned a
    four-tile Everest box into 132 tiles and 3.4 GB. A raw min/max over GPS
    fixes is maximally sensitive to exactly one bad point."""
    pts = TREK + [stray]

    # what a naive min/max would have produced
    lats = [p[0] for p in pts]; lons = [p[1] for p in pts]
    naive = tiles_for_bbox(min(lats)-0.1, min(lons)-0.1, max(lats)+0.1, max(lons)+0.1)
    assert len(naive) >= raw_tiles_at_least, "fixture should be pathological"

    bbox, dropped, furthest = robust_bbox(pts)
    assert dropped == 1
    assert furthest > DEFAULT_MAX_RADIUS_KM
    robust = tiles_for_bbox(bbox[0]-0.1, bbox[1]-0.1, bbox[2]+0.1, bbox[3]+0.1)
    assert len(robust) <= 4, f"{name}: still {len(robust)} tiles"


def test_kathmandu_is_close_enough_to_keep():
    """Real trek logistics -- the flight in -- must not be treated as an outlier."""
    bbox, dropped, _ = robust_bbox(TREK + [(27.7172, 85.3240)])
    assert dropped == 0
    assert bbox[1] < 86.0, "Kathmandu should widen the box, not be discarded"


def test_genuine_extremes_survive():
    """A summit visited once is a real extreme, not an outlier. A percentile
    clip would have discarded it; a distance test keeps it."""
    summit = (28.0026, 86.8528)
    bbox, dropped, _ = robust_bbox(TREK)
    assert dropped == 0
    assert bbox[2] >= summit[0] - 1e-9


def test_many_outliers_still_leave_the_trek():
    pts = TREK + [(55.75, 37.62)] * 3 + [(25.25, 55.36)] * 2
    bbox, dropped, _ = robust_bbox(pts)
    assert dropped == 5
    assert len(tiles_for_bbox(bbox[0]-0.1, bbox[1]-0.1, bbox[2]+0.1, bbox[3]+0.1)) <= 4


def test_null_island_fixes_are_ignored():
    bbox, dropped, _ = robust_bbox(TREK + [(0.0, 0.0)])
    assert dropped == 0, "0,0 is filtered before the distance test, not counted"
    assert bbox[0] > 27.0


def test_empty_input():
    assert robust_bbox([]) is None


def test_all_points_dropped_returns_none():
    """Two fixes on opposite sides of the world: no coherent route."""
    assert robust_bbox([(55.75, 37.62), (-33.86, 151.20)], max_radius_km=1.0) is None


# -- source precedence -------------------------------------------------

def test_prefers_the_gps_track_when_present(tmp_path):
    p = _db(tmp_path, gps=[(27.5, 86.5), (28.0, 87.0)], assets=[(20.0, 80.0)])
    pts, via = points_from_db(p)
    assert via == "GPS track"
    assert (20.0, 80.0) not in pts


def test_falls_back_to_assets_after_s01_only(tmp_path):
    p = _db(tmp_path, assets=TREK)
    pts, via = points_from_db(p)
    assert via == "photo EXIF"
    assert len(pts) == len(TREK)


def test_empty_and_missing_databases(tmp_path):
    assert points_from_db(_db(tmp_path)) is None
    assert points_from_db(tmp_path / "absent.sqlite") is None


def test_reads_the_database_without_locking_it(tmp_path):
    p = _db(tmp_path, assets=[(27.0, 86.0)])
    conn = sqlite3.connect(p)
    conn.execute("BEGIN EXCLUSIVE")
    try:
        assert points_from_db(p) is not None
    finally:
        conn.rollback()
        conn.close()


def test_exif_scan_on_a_missing_root(tmp_path):
    assert points_from_exif(tmp_path / "nope") is None


# -- pruning -----------------------------------------------------------

def test_prune_removes_only_unneeded_tiles(tmp_path):
    for name in ("N27E086", "N28E086", "N27E087", "N55E037", "N25E055"):
        (tmp_path / f"{name}.hgt").write_bytes(b"x")
    removed = prune_tiles(["N27E086", "N28E086", "N27E087"], tmp_path, dry_run=False)
    assert sorted(removed) == ["N25E055", "N55E037"]
    assert (tmp_path / "N27E086.hgt").exists()
    assert not (tmp_path / "N55E037.hgt").exists()


def test_prune_dry_run_deletes_nothing(tmp_path):
    (tmp_path / "N55E037.hgt").write_bytes(b"x")
    removed = prune_tiles(["N27E086"], tmp_path, dry_run=True)
    assert removed == ["N55E037"]
    assert (tmp_path / "N55E037.hgt").exists()


def test_prune_on_a_missing_directory(tmp_path):
    assert prune_tiles(["N27E086"], tmp_path / "nope", dry_run=True) == []
