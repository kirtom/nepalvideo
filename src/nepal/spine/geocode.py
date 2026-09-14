"""S02.4 -- offline reverse geocoding to place names.

Deviation from the spec, and why: the spec calls for a bundled offline
Nominatim extract. Nominatim needs PostgreSQL, PostGIS and a multi-gigabyte
OSM import that takes hours -- a database to run inside the Batch container,
for the sake of naming roughly fifty cluster centroids.

A GeoNames country dump does the same job better here. The Nepal file is a few
megabytes of tab-separated text covering every populated place, peak, pass,
glacier and lake in the country, and a KD-tree over it answers a nearest-name
query instantly with no service to run. It is also a better fit for the
question actually being asked: on a trek you want "Namche Bazaar", "Tengboche"
and "Kala Patthar", which are GeoNames populated places and peaks, whereas
Nominatim's reverse geocoding is built to return postal addresses and roads.

Download: https://download.geonames.org/export/dump/NP.zip  (or any country code)
"""
from __future__ import annotations

import logging
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

log = logging.getLogger(__name__)

# GeoNames feature classes, in the order a trek narrative cares about.
# P populated place, T mountain/pass/ridge, H lake/glacier/stream, L park/area,
# S spot (buildings, huts), V vegetation, R road, A administrative division.
CLASS_PRIORITY = {"P": 0, "T": 1, "H": 2, "S": 3, "L": 4, "V": 5, "R": 6, "A": 7}
EARTH_R = 6371008.8


@dataclass(frozen=True)
class Place:
    name: str
    lat: float
    lon: float
    feature_class: str
    feature_code: str
    population: int = 0

    @property
    def priority(self) -> int:
        return CLASS_PRIORITY.get(self.feature_class, 9)


def load_geonames(path: str | Path, *, min_priority: int = 6) -> list[Place]:
    """Read a GeoNames country dump (``NP.txt`` or ``NP.zip``).

    Column layout is fixed and documented at
    https://download.geonames.org/export/dump/readme.txt -- fields 1 name,
    4 latitude, 5 longitude, 6 feature class, 7 feature code, 14 population.
    """
    p = Path(path)
    if not p.exists():
        log.warning("GeoNames dump not found at %s -- place names will be null", p)
        return []

    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as z:
            member = next((n for n in z.namelist() if n.lower().endswith(".txt")
                           and "readme" not in n.lower()), None)
            if member is None:
                return []
            text = z.read(member).decode("utf-8", "replace")
    else:
        text = p.read_text(encoding="utf-8", errors="replace")

    places: list[Place] = []
    for line in text.splitlines():
        f = line.split("\t")
        if len(f) < 15:
            continue
        try:
            place = Place(f[1], float(f[4]), float(f[5]), f[6], f[7],
                          int(f[14]) if f[14].isdigit() else 0)
        except (ValueError, IndexError):
            continue
        if place.priority <= min_priority:
            places.append(place)
    log.info("GeoNames: %d places loaded from %s", len(places), p.name)
    return places


class Gazetteer:
    """Nearest named place, weighted so a village outranks a spot height."""

    def __init__(self, places: Sequence[Place]):
        self.places = list(places)
        self._tree = None
        self._xyz = None
        if self.places:
            self._build()

    def _build(self) -> None:
        import numpy as np
        from scipy.spatial import cKDTree
        lat = np.radians([p.lat for p in self.places])
        lon = np.radians([p.lon for p in self.places])
        # 3-D unit sphere so the tree's Euclidean metric behaves like a great
        # circle; a flat lat/lon tree distorts badly away from the equator.
        self._xyz = np.column_stack([np.cos(lat) * np.cos(lon),
                                     np.cos(lat) * np.sin(lon),
                                     np.sin(lat)])
        self._tree = cKDTree(self._xyz)

    def nearest(self, lat: float, lon: float, *, k: int = 12,
                max_m: float = 5000.0) -> Place | None:
        """The most narratively useful place within ``max_m``.

        Not simply the closest: an unnamed spot height 200 m away is a worse
        answer than the village 900 m away whose name appears in the chat.
        Candidates are ranked by feature class first, then by distance, with a
        nudge for population so a real settlement beats a hamlet.
        """
        if self._tree is None:
            return None
        import numpy as np
        la, lo = math.radians(lat), math.radians(lon)
        q = np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])
        k = min(k, len(self.places))
        _, idx = self._tree.query(q, k=k)
        idx = np.atleast_1d(idx)

        best, best_score = None, float("inf")
        for i in idx:
            if i >= len(self.places):
                continue
            pl = self.places[int(i)]
            d = _haversine(lat, lon, pl.lat, pl.lon)
            if d > max_m:
                continue
            pop_bonus = 300.0 * math.log10(pl.population + 1)
            score = pl.priority * 1200.0 + d - pop_bonus
            if score < best_score:
                best, best_score = pl, score
        return best


def _haversine(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def cluster_coords(coords: Sequence[tuple[float, float]],
                   radius_m: float = 500.0) -> list[tuple[tuple[float, float], list[int]]]:
    """Group nearby coordinates so the gazetteer is queried once per place.

    Greedy single-pass clustering, which is enough at this scale and keeps the
    result deterministic. Returns (centroid, member indices).
    """
    clusters: list[tuple[list[tuple[float, float]], list[int]]] = []
    for i, (la, lo) in enumerate(coords):
        for members, idxs in clusters:
            c_la = sum(m[0] for m in members) / len(members)
            c_lo = sum(m[1] for m in members) / len(members)
            if _haversine(la, lo, c_la, c_lo) <= radius_m:
                members.append((la, lo))
                idxs.append(i)
                break
        else:
            clusters.append(([(la, lo)], [i]))
    return [((sum(m[0] for m in ms) / len(ms), sum(m[1] for m in ms) / len(ms)), idxs)
            for ms, idxs in clusters]
