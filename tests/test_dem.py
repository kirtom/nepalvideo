import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from nepal.spine.dem import Srtm, tile_name, tiles_for_bbox, resolve_altitude, VOID

N = 1201  # SRTM3


@pytest.fixture(scope="module")
def tiledir(tmp_path_factory):
    """A synthetic N27E086 whose elevation is a known function of grid position."""
    d = tmp_path_factory.mktemp("srtm")
    rows = np.arange(N, dtype=np.int16).reshape(N, 1)
    cols = np.arange(N, dtype=np.int16).reshape(1, N)
    grid = (rows + cols).astype(">i2")
    grid[600, 600] = VOID                       # a void pixel to exercise the filter
    (d / "N27E086.hgt").write_bytes(grid.tobytes())
    return d


@pytest.mark.parametrize("lat,lon,expected", [
    (27.98, 86.92, "N27E086"), (28.00, 86.92, "N28E086"),
    (-1.5, -0.5, "S02W001"), (0.5, 0.5, "N00E000"),
])
def test_tile_name(lat, lon, expected):
    assert tile_name(lat, lon) == expected


def test_tiles_for_bbox_covers_the_everest_region():
    assert tiles_for_bbox(27.6, 86.7, 28.1, 87.0) == \
        ["N27E086", "N27E087", "N28E086", "N28E087"]


def test_reads_the_north_west_corner(tiledir):
    """Row 0 is the NORTHERN edge -- inverting this flips the whole DEM."""
    # just inside the tile, so this tests the row mapping and not the edge rule
    assert Srtm(tiledir).elevation(28.0 - 1e-6, 86.0 + 1e-6) == pytest.approx(0.0, abs=0.02)


def test_reads_the_south_east_corner(tiledir):
    got = Srtm(tiledir).elevation(27.0 + 1e-6, 87.0 - 1e-6)
    assert got == pytest.approx(2 * (N - 1), abs=0.02)


def test_point_exactly_on_a_shared_tile_edge(tiledir):
    """Latitude 28.0 is the top row of N27E086 and the bottom row of N28E086.
    floor() picks N28; with only N27 bundled the lookup must still resolve."""
    assert Srtm(tiledir).elevation(28.0, 86.5) is not None


def test_point_on_a_corner_shared_by_four_tiles(tiledir):
    assert Srtm(tiledir).elevation(28.0, 86.0) == pytest.approx(0.0, abs=0.02)


def test_edge_fallback_does_not_invent_data_far_away(tiledir):
    """The nudge must only cross a shared edge, never fabricate a neighbour."""
    assert Srtm(tiledir).elevation(45.0, 10.0) is None


def test_latitude_increases_northward(tiledir):
    """A sanity check that catches a flipped row index: going north must move
    toward row 0, and in this grid that means a lower value."""
    s = Srtm(tiledir)
    assert s.elevation(27.9, 86.5) < s.elevation(27.1, 86.5)


def test_bilinear_interpolation_between_samples(tiledir):
    s = Srtm(tiledir)
    a = s.elevation(27.5, 86.5)
    b = s.elevation(27.5 + 0.5 / (N - 1), 86.5)
    assert a != b, "must interpolate, not quantise to the nearest grid cell"


def test_void_pixels_are_excluded(tiledir):
    """SRTM marks voids with -32768; averaging one in yields nonsense."""
    s = Srtm(tiledir)
    v = s.elevation(28.0 - 600 / (N - 1), 86.0 + 600 / (N - 1))
    assert v is not None and v > 0, f"got {v}"


def test_missing_tile_returns_none_and_is_recorded(tiledir):
    s = Srtm(tiledir)
    assert s.elevation(45.5, 10.5) is None
    assert "N45E010" in s.missing


def test_missing_directory_is_not_fatal(tmp_path):
    s = Srtm(tmp_path / "nope")
    assert s.elevation(27.5, 86.5) is None


def test_batch_lookup(tiledir):
    got = Srtm(tiledir).elevations([(27.5, 86.5), (45.0, 10.0)])
    assert got[0] is not None and got[1] is None


# -- altitude source precedence ----------------------------------------

def test_gpx_elevation_wins():
    v, src = resolve_altitude(dem_m=5300.0, gpx_ele=5364.0, gps_exif_m=5200.0)
    assert (v, src) == (5364.0, "gpx")


def test_dem_beats_gps_exif():
    """The spec's whole point: GPS altitude is unusable at elevation."""
    v, src = resolve_altitude(dem_m=5300.0, gpx_ele=None, gps_exif_m=5200.0)
    assert (v, src) == (5300.0, "srtm")


def test_gps_exif_is_the_last_resort_and_says_so():
    v, src = resolve_altitude(None, None, 5200.0)
    assert v == 5200.0 and "unreliable" in src


def test_no_altitude_available():
    assert resolve_altitude(None, None, None) == (None, "none")
