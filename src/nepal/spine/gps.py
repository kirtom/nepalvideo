"""S02.1 / S02.2 -- the geolocation spine.

The camera has no GPS receiver. Phone photo EXIF is the only source of
location in the project, and every camera recording is placed by interpolating
against it. That makes this module load-bearing: an error here does not fail
loudly, it quietly puts shots in the wrong valley.

A real ``.gpx`` shared in the chat outranks photo EXIF where one exists -- a
route track is sampled every few seconds, photo EXIF every few minutes.
"""
from __future__ import annotations

import logging
import math
import statistics
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence

log = logging.getLogger(__name__)

EARTH_R = 6371008.8


@dataclass(frozen=True)
class GpsPoint:
    ts: datetime
    lat: float
    lon: float
    ele: float | None = None
    source: str = "unknown"
    accuracy_m: float | None = None

    def key(self) -> str:
        return self.ts.astimezone(timezone.utc).isoformat()


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


# -- gpx ---------------------------------------------------------------

def parse_gpx(text: str, *, source: str = "gpx") -> list[GpsPoint]:
    """Track points from a GPX document, tolerant of namespaces and of
    trackpoints that carry no time (which are dropped -- untimed points cannot
    anchor anything)."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        log.warning("GPX parse failed: %s", exc)
        return []

    pts: list[GpsPoint] = []
    for el in root.iter():
        if not el.tag.rsplit("}", 1)[-1] == "trkpt":
            continue
        try:
            lat, lon = float(el.attrib["lat"]), float(el.attrib["lon"])
        except (KeyError, ValueError):
            continue
        ts = ele = None
        for child in el:
            name = child.tag.rsplit("}", 1)[-1]
            if name == "time" and child.text:
                ts = _parse_iso(child.text.strip())
            elif name == "ele" and child.text:
                try:
                    ele = float(child.text)
                except ValueError:
                    pass
        if ts is not None:
            pts.append(GpsPoint(ts, lat, lon, ele, source))
    pts.sort(key=lambda p: p.ts)
    log.info("GPX: %d timed track points", len(pts))
    return pts


def _parse_iso(s: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def write_gpx(points: Sequence[GpsPoint], *, name: str = "trek") -> str:
    rows = []
    for p in points:
        ele = f"<ele>{p.ele:.1f}</ele>" if p.ele is not None else ""
        t = p.ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows.append(f'      <trkpt lat="{p.lat:.7f}" lon="{p.lon:.7f}">{ele}'
                    f"<time>{t}</time></trkpt>")
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<gpx version="1.1" creator="nepal-film-pipeline" '
            'xmlns="http://www.topografix.com/GPX/1/1">\n'
            f"  <trk><name>{name}</name><trkseg>\n"
            + "\n".join(rows) + "\n    </trkseg></trk>\n</gpx>\n")


# -- merging -----------------------------------------------------------

PREFERENCE = ("gpx", "phone_keller", "phone_kulikov")


def merge_points(groups: Iterable[Sequence[GpsPoint]]) -> list[GpsPoint]:
    """One time-sorted track from several sources.

    Section S02.2 rule: on a collision at the same second, prefer the point
    with better reported accuracy; failing that, prefer by source order --
    a real GPX track first, then keller.
    """
    best: dict[str, GpsPoint] = {}
    for group in groups:
        for p in group:
            k = p.key()
            cur = best.get(k)
            if cur is None or _better(p, cur):
                best[k] = p
    return sorted(best.values(), key=lambda p: p.ts)


def _better(candidate: GpsPoint, incumbent: GpsPoint) -> bool:
    ca, ia = candidate.accuracy_m, incumbent.accuracy_m
    if ca is not None and ia is not None and ca != ia:
        return ca < ia
    if ca is not None and ia is None:
        return True
    if ia is not None and ca is None:
        return False
    return _rank(candidate.source) < _rank(incumbent.source)


def _rank(source: str) -> int:
    return PREFERENCE.index(source) if source in PREFERENCE else len(PREFERENCE)


def drop_outliers(points: Sequence[GpsPoint], *, max_speed_ms: float = 60.0) -> list[GpsPoint]:
    """Remove points implying impossible travel.

    A single bad fix -- and phone EXIF does produce them, usually a stale
    cached location from the last city with signal -- otherwise drags every
    interpolated camera position between it and its neighbours. 60 m/s keeps
    flights and buses while rejecting a jump to another country.
    """
    if len(points) < 3:
        return list(points)
    out = [points[0]]
    dropped = 0
    for p in points[1:]:
        prev = out[-1]
        dt = (p.ts - prev.ts).total_seconds()
        if dt <= 0:
            continue
        if haversine_m(prev.lat, prev.lon, p.lat, p.lon) / dt > max_speed_ms:
            dropped += 1
            continue
        out.append(p)
    if dropped:
        log.warning("S02.1 dropped %d GPS point(s) implying travel above %.0f m/s",
                    dropped, max_speed_ms)
    return out


# -- interpolation -----------------------------------------------------

def interpolate_at(points: Sequence[GpsPoint], ts: datetime, *,
                   max_gap_s: float = 14400.0) -> tuple[float, float] | None:
    """Linear position at ``ts``, or None.

    Returns None outside the track and across any gap longer than
    ``max_gap_s`` (4 hours by default). Refusing is the point: a camera
    recording made during an untracked half-day would otherwise be smeared
    between two villages and mis-assigned to an act.
    """
    if not points:
        return None
    lo, hi = _bracket(points, ts)
    if lo is None or hi is None:
        return None
    if lo is hi:
        return (lo.lat, lo.lon)
    span = (hi.ts - lo.ts).total_seconds()
    if span > max_gap_s:
        return None
    if span <= 0:
        return (lo.lat, lo.lon)
    f = (ts - lo.ts).total_seconds() / span
    return (lo.lat + (hi.lat - lo.lat) * f, lo.lon + (hi.lon - lo.lon) * f)


def _bracket(points: Sequence[GpsPoint], ts: datetime):
    import bisect
    times = [p.ts for p in points]
    i = bisect.bisect_left(times, ts)
    if i < len(points) and points[i].ts == ts:
        return points[i], points[i]
    if i == 0 or i == len(points):
        return None, None          # outside the envelope: do not extrapolate
    return points[i - 1], points[i]


def envelope(points: Sequence[GpsPoint]) -> tuple[datetime, datetime] | None:
    return (points[0].ts, points[-1].ts) if points else None


def coverage(points: Sequence[GpsPoint], timestamps: Sequence[datetime],
             *, max_gap_s: float = 14400.0) -> float:
    """Fraction of ``timestamps`` this track can place. The S01.5 acceptance
    check (90% of camera recordings inside the GPS envelope) reads this."""
    if not timestamps:
        return 0.0
    ok = sum(1 for t in timestamps if interpolate_at(points, t, max_gap_s=max_gap_s))
    return ok / len(timestamps)


def day_index(ts: datetime, start: datetime) -> int:
    """1-based trek day. Day boundaries are taken in Nepal local time, so a
    shot at 23:40 and one at 00:20 sit on the days a person would call them."""
    tz = timezone(timedelta(hours=5, minutes=45))
    return (ts.astimezone(tz).date() - start.astimezone(tz).date()).days + 1
