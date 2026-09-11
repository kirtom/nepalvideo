import sys, pathlib, sqlite3
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from tools.fetch_reference import bbox_from_db, bbox_from_exif
from nepal import db


def _db(tmp_path, *, gps=(), assets=()):
    p = tmp_path / "nepal.sqlite"
    conn = db.init(p)
    for ts, la, lo in gps:
        conn.execute("INSERT INTO gps_points(ts_utc, lat, lon) VALUES (?,?,?)", (ts, la, lo))
    for i, (la, lo) in enumerate(assets):
        conn.execute("INSERT INTO assets(asset_id, s3_key, source, kind, lat, lon, has_gps) "
                     "VALUES (?,?,?,?,?,?,1)",
                     (f"a{i}", f"raw/x{i}.jpg", "phone_keller", "photo", la, lo))
    conn.commit()
    conn.close()
    return p


def test_prefers_the_gps_track_when_present():
    """After S02 the merged track is the best source."""
    import tempfile
    tmp = pathlib.Path(tempfile.mkdtemp())
    p = _db(tmp, gps=[("2024-04-27T00:00:00+00:00", 27.5, 86.5),
                      ("2024-04-28T00:00:00+00:00", 28.0, 87.0)],
            assets=[(20.0, 80.0)])
    bbox, via = bbox_from_db(p)
    assert via == "GPS track"
    assert bbox == (27.5, 86.5, 28.0, 87.0)


def test_falls_back_to_assets_after_s01_only(tmp_path):
    """The deadlock this guards: SRTM tiles are needed to compute altitude in
    S02, but gps_points is only written by S02. assets.lat/lon comes from photo
    EXIF and is populated by S01, so the tiles can be fetched in between."""
    p = _db(tmp_path, assets=[(27.6869, 86.7133), (28.0026, 86.8528)])
    bbox, via = bbox_from_db(p)
    assert via == "photo EXIF"
    assert bbox == pytest.approx((27.6869, 86.7133, 28.0026, 86.8528))


def test_empty_database_yields_nothing(tmp_path):
    assert bbox_from_db(_db(tmp_path)) is None


def test_missing_database_yields_nothing(tmp_path):
    assert bbox_from_db(tmp_path / "absent.sqlite") is None


def test_reads_the_database_without_locking_it(tmp_path):
    """Opened read-only, so fetching tiles cannot interfere with a running stage."""
    p = _db(tmp_path, assets=[(27.0, 86.0)])
    conn = sqlite3.connect(p)
    conn.execute("BEGIN EXCLUSIVE")
    try:
        assert bbox_from_db(p) is not None
    finally:
        conn.rollback()
        conn.close()


def test_exif_scan_needs_no_database_at_all(tmp_path):
    """Makes the tool independent of pipeline order: it works on nothing but
    the delivered tree."""
    import shutil
    if not shutil.which("exiftool"):
        pytest.skip("needs exiftool")
    assert bbox_from_exif(tmp_path) is None      # empty tree, no fixes


def test_exif_scan_on_a_missing_root(tmp_path):
    assert bbox_from_exif(tmp_path / "nope") is None
