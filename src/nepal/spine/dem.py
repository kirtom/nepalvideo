"""S02.3 -- altitude from an SRTM digital elevation model.

GPS altitude is unusable at elevation: receivers report height above the
WGS84 ellipsoid rather than sea level (about -30 m of offset in Nepal), and
vertical error runs several times the horizontal. Altitude is the dramatic
axis of this film, so it comes from a DEM keyed on lat/lon instead.

Deviation from the spec, and why: the spec says to bundle SRTM tiles, which
is right, but reading them does not need GDAL or rasterio. An ``.hgt`` file is
raw big-endian int16 in a square grid with no header at all -- numpy reads it
directly. That keeps a heavyweight geo stack out of the container for the sake
of one array lookup.

Resolution is inferred from file size: SRTM1 is 3601x3601 (1 arc-second, ~30 m)
and SRTM3 is 1201x1201 (3 arc-second, ~90 m).
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)

VOID = -32768
SIZES = {3601 * 3601 * 2: 3601, 1201 * 1201 * 2: 1201, 7201 * 7201 * 2: 7201}


def tile_name(lat: float, lon: float) -> str:
    """The tile containing a coordinate, e.g. (27.98, 86.92) -> N27E086."""
    la, lo = math.floor(lat), math.floor(lon)
    return (f"{'N' if la >= 0 else 'S'}{abs(la):02d}"
            f"{'E' if lo >= 0 else 'W'}{abs(lo):03d}")


def tiles_for_bbox(min_lat: float, min_lon: float,
                   max_lat: float, max_lon: float) -> list[str]:
    """Every tile needed to cover a bounding box -- what to bundle."""
    out = []
    for la in range(math.floor(min_lat), math.floor(max_lat) + 1):
        for lo in range(math.floor(min_lon), math.floor(max_lon) + 1):
            out.append(tile_name(la + 0.5, lo + 0.5))
    return sorted(set(out))


class Srtm:
    """Lazily memory-mapped tile set. Missing tiles yield None, not an error:
    a trek that strays off the bundled tiles should lose altitude for those
    shots, not fail the stage."""

    def __init__(self, tile_dir: str | Path):
        self.dir = Path(tile_dir)
        self._cache: dict[str, np.ndarray | None] = {}
        self.missing: set[str] = set()

    def _load(self, name: str) -> np.ndarray | None:
        if name in self._cache:
            return self._cache[name]
        arr = None
        for candidate in (self.dir / f"{name}.hgt", self.dir / f"{name}.HGT",
                          self.dir / f"{name}.hgt.gz"):
            if not candidate.exists():
                continue
            try:
                if candidate.suffix == ".gz":
                    import gzip
                    raw = gzip.decompress(candidate.read_bytes())
                    n = SIZES.get(len(raw))
                    if n:
                        arr = np.frombuffer(raw, dtype=">i2").reshape(n, n)
                else:
                    n = SIZES.get(candidate.stat().st_size)
                    if n:
                        arr = np.memmap(candidate, dtype=">i2", mode="r", shape=(n, n))
                    else:
                        log.warning("DEM tile %s has unexpected size %d bytes",
                                    candidate.name, candidate.stat().st_size)
            except (OSError, ValueError) as exc:
                log.warning("DEM tile %s unreadable: %s", candidate.name, exc)
            break
        if arr is None:
            self.missing.add(name)
        self._cache[name] = arr
        return arr

    def elevation(self, lat: float, lon: float) -> float | None:
        """Bilinearly interpolated elevation in metres, or None.

        Bilinear rather than nearest: on a 30 m grid in terrain that climbs
        600 m in a kilometre, nearest-neighbour quantises the altitude curve
        into visible steps, and that curve is what drives act boundaries.
        """
        arr = self._load(tile_name(lat, lon))
        if arr is None:
            arr, lat, lon = self._edge_fallback(lat, lon)
        if arr is None:
            return None
        n = arr.shape[0]
        # row 0 is the northern edge of the tile
        row = (math.floor(lat) + 1 - lat) * (n - 1)
        col = (lon - math.floor(lon)) * (n - 1)
        r0, c0 = int(math.floor(row)), int(math.floor(col))
        r1, c1 = min(r0 + 1, n - 1), min(c0 + 1, n - 1)
        if not (0 <= r0 < n and 0 <= c0 < n):
            return None
        fr, fc = row - r0, col - c0

        corners = [(arr[r0, c0], (1 - fr) * (1 - fc)), (arr[r0, c1], (1 - fr) * fc),
                   (arr[r1, c0], fr * (1 - fc)), (arr[r1, c1], fr * fc)]
        valid = [(float(v), w) for v, w in corners if int(v) != VOID]
        if not valid:
            return None
        total_w = sum(w for _, w in valid)
        if total_w <= 0:
            return float(valid[0][0])
        return sum(v * w for v, w in valid) / total_w

    def _edge_fallback(self, lat: float, lon: float):
        """Tiles share their edges: latitude 28.0 is both the top row of
        N27E086 and the bottom row of N28E086. floor() always picks the
        northern/eastern neighbour, so a point landing exactly on a boundary
        reads as missing whenever only the other side was bundled. Nudge into
        the neighbour that does exist rather than returning no altitude.
        """
        if lat == math.floor(lat):
            arr = self._load(tile_name(lat - 1e-9, lon))
            if arr is not None:
                return arr, lat - 1e-9, lon
        if lon == math.floor(lon):
            arr = self._load(tile_name(lat, lon - 1e-9))
            if arr is not None:
                return arr, lat, lon - 1e-9
        if lat == math.floor(lat) and lon == math.floor(lon):
            arr = self._load(tile_name(lat - 1e-9, lon - 1e-9))
            if arr is not None:
                return arr, lat - 1e-9, lon - 1e-9
        return None, lat, lon

    def elevations(self, coords: Sequence[tuple[float, float]]) -> list[float | None]:
        return [self.elevation(la, lo) for la, lo in coords]


def resolve_altitude(dem_m: float | None, gpx_ele: float | None,
                     gps_exif_m: float | None) -> tuple[float | None, str]:
    """Pick an altitude and say where it came from.

    Order: a barometric or survey-grade elevation from a shared GPX track,
    then the DEM, and only then EXIF GPS altitude -- which the spec rules out
    at elevation and which is kept purely as a last resort so a shot is not
    left with no altitude at all.
    """
    if gpx_ele is not None:
        return float(gpx_ele), "gpx"
    if dem_m is not None:
        return float(dem_m), "srtm"
    if gps_exif_m is not None:
        return float(gps_exif_m), "gps_exif(unreliable)"
    return None, "none"
