"""S02.1 / S02.2 -- the geolocation spine.

The camera has no GPS receiver. Phone photo EXIF is the only source of
location in the project, and every camera recording is placed by interpolating
against it. That makes this module load-bearing: an error here does not fail
loudly, it quietly puts shots in the wrong valley.

A Strava track from a watch outranks everything: a fix a second, barometric
altitude and heart rate. A real ``.gpx`` shared in the chat comes next, then
photo EXIF -- a route track is sampled every few seconds, photo EXIF every
few minutes.
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
    # From a watch: heart rate, and which activity the fix belongs to. Both
    # None for a photo fix.
    hr: float | None = None
    activity_id: str | None = None

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

PREFERENCE = ("strava", "gpx", "phone_keller", "phone_kulikov")


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


def interpolate_point(points: Sequence[GpsPoint], ts: datetime, *,
                      max_gap_s: float = 14400.0) -> GpsPoint | None:
    """Like ``interpolate_at`` but a whole point: altitude and heart rate
    come along **only between two watch fixes**. A photo fix's ``ele`` is a
    DEM lookup at the photo, and interpolating two of those is worse than
    looking the DEM up at the interpolated position, so between anything
    else they are left None for the caller to resolve."""
    if not points:
        return None
    lo, hi = _bracket(points, ts)
    if lo is None or hi is None:
        return None
    if lo is hi:
        return GpsPoint(ts, lo.lat, lo.lon, lo.ele if lo.activity_id else None,
                        "interp", None, lo.hr if lo.activity_id else None,
                        lo.activity_id)
    span = (hi.ts - lo.ts).total_seconds()
    if span > max_gap_s:
        return None
    f = 0.0 if span <= 0 else (ts - lo.ts).total_seconds() / span
    both_watch = bool(lo.activity_id and hi.activity_id)

    def mix(a: float | None, b: float | None) -> float | None:
        if not both_watch or a is None or b is None:
            return None
        return a + (b - a) * f

    return GpsPoint(ts, lo.lat + (hi.lat - lo.lat) * f, lo.lon + (hi.lon - lo.lon) * f,
                    mix(lo.ele, hi.ele), "interp", None, mix(lo.hr, hi.hr),
                    lo.activity_id if both_watch else None)


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
    """Full temporal span of the track, outliers included."""
    return (points[0].ts, points[-1].ts) if points else None


def trek_envelope(points: Sequence[GpsPoint], *, max_radius_km: float = 200.0
                  ) -> tuple[tuple[datetime, datetime] | None, int]:
    """Temporal span of the points that are actually on the trek.

    Section S02.5 classifies a message as planning, trek or after by where it
    falls against this envelope -- so the envelope had better describe the trek.
    The full span does not: phone photos taken at home during the planning
    months carry GPS too, and on real material they stretched the envelope back
    eleven weeks. Every planning message then landed inside it, was labelled
    "trek", and Act 1 -- which is built entirely from planning-phase material --
    collapsed from ten weeks to forty-one minutes.

    The trek is where the route is, so points further than ``max_radius_km``
    from the median position are excluded before the span is measured. The
    median cannot be moved by outliers, and the radius is generous enough to
    keep the trailhead city: Kathmandu sits about 110 km from Manaslu.

    Returns (envelope, n_excluded).
    """
    pts = [p for p in points if p.lat is not None and p.lon is not None]
    if not pts:
        return None, 0
    c_lat = statistics.median([p.lat for p in pts])
    c_lon = statistics.median([p.lon for p in pts])
    on_route = [p for p in pts
                if haversine_m(c_lat, c_lon, p.lat, p.lon) / 1000.0 <= max_radius_km]
    if not on_route:
        return envelope(pts), 0
    excluded = len(pts) - len(on_route)
    if excluded:
        log.info("S02 trek envelope: %d of %d GPS points are within %.0f km of the "
                 "route; %d elsewhere (photos from home or in transit) are excluded "
                 "from the envelope that classifies message phase",
                 len(on_route), len(pts), max_radius_km, excluded)
    on_route.sort(key=lambda p: p.ts)
    return (on_route[0].ts, on_route[-1].ts), excluded


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
