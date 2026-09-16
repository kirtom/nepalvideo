"""Where and when a moment was.

Every shot needs a position, an altitude, a place name and a trek day: the
place cards, the altitude axis, the context score's "first at a place" and
"new altitude record" terms, and the six-act boundaries all read them. S03.0
filled them for photographs from the photo's own fix; video shots never had
them, so on the first draft the context score favoured stills and Act 2 was
a slideshow.

Pure except for the two lookups it is handed (the DEM and the gazetteer),
which are cheap, local and already tested.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from nepal.spine import dem as dem_mod
from nepal.spine import gps as gps_mod
from nepal.spine.gps import GpsPoint


@dataclass(frozen=True)
class Placement:
    lat: float | None
    lon: float | None
    alt_m: float | None
    alt_source: str
    place_name: str | None
    day_index: int | None


def load_track(conn) -> list[GpsPoint]:
    """The merged track from the database, with what the watch added."""
    out: list[GpsPoint] = []
    for r in conn.execute("SELECT ts_utc, lat, lon, alt_dem_m, hr_bpm, source, "
                          "activity_id FROM gps_points ORDER BY ts_utc"):
        try:
            ts = datetime.fromisoformat(str(r["ts_utc"]).replace("Z", "+00:00"))
        except ValueError:
            continue
        out.append(GpsPoint(ts, r["lat"], r["lon"], r["alt_dem_m"], r["source"],
                            None, r["hr_bpm"], r["activity_id"]))
    return out


def trek_span(bounds: Sequence) -> tuple[datetime, datetime] | None:
    """From the start of Act 2 to the end of the last act. Day 1 is the
    first day of Act 2; planning and after-trek material carry no day."""
    trek = [b for b in bounds if int(b.act) >= 2]
    if not trek:
        return None
    return (min(b.start_utc for b in trek), max(b.end_utc for b in trek))


def day_of(ts: datetime | None, span: tuple[datetime, datetime] | None) -> int | None:
    if ts is None or span is None or not (span[0] <= ts <= span[1]):
        return None
    return gps_mod.day_index(ts, span[0])


def describe(lat: float, lon: float, *, srtm: dem_mod.Srtm | None,
             gazetteer, ele: float | None = None
             ) -> tuple[float | None, str, str | None]:
    """Altitude (barometric first, then DEM), where it came from, and the
    nearest narratively useful place."""
    dem = srtm.elevation(lat, lon) if srtm is not None else None
    alt, src = dem_mod.resolve_altitude(dem, ele, None)
    pl = gazetteer.nearest(lat, lon) if gazetteer is not None else None
    return alt, src, (pl.name if pl else None)


def place_at(track: Sequence[GpsPoint], ts: datetime | None, *, srtm, gazetteer,
             max_gap_s: float, span: tuple[datetime, datetime] | None) -> Placement:
    """Everything the film wants to know about one moment."""
    day = day_of(ts, span)
    pt = gps_mod.interpolate_point(track, ts, max_gap_s=max_gap_s) \
        if (ts is not None and track) else None
    if pt is None:
        return Placement(None, None, None, "none", None, day)
    alt, src, name = describe(pt.lat, pt.lon, srtm=srtm, gazetteer=gazetteer, ele=pt.ele)
    return Placement(pt.lat, pt.lon, alt, src, name, day)
