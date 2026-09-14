"""Exertion, from the GPS track alone.

``new_max_alt`` rewards altitude novelty -- the first time the trek goes higher
than it has been. That is a fact about the route, not about the people on it.
The dramatically valuable material is struggle: the slowest hour, the steepest
gain, the long unexplained stop where something went wrong. All three are
already in the track, and none of them was being read.

Everything here is pure: a list of GpsPoint in, numbers out. The film uses it
twice -- as a term in ``score_ctx`` so hard-won shots outrank pretty ones, and
to choose where the music drops out entirely, because breathing at five thousand
metres is worth more than any cue that could be laid over it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Sequence

from nepal.spine.gps import GpsPoint, haversine_m

log = logging.getLogger(__name__)

# Walking a loaded pack on a trail sits near 1 m/s; above this the party is in a
# vehicle, and a jeep to the trailhead is not exertion however far it travels.
MAX_WALK_MS = 2.5
# Below this the party is standing still, whatever the GPS jitter says.
STOPPED_MS = 0.15
# A day's serious climbing is roughly 500 m; per hour, 300 is very steep.
STEEP_GAIN_M_PER_H = 300.0


@dataclass(frozen=True)
class Effort:
    """What the track says about one moment."""
    ts: datetime
    speed_ms: float
    gain_m_per_h: float
    alt_m: float | None
    stopped_s: float = 0.0

    @property
    def is_stopped(self) -> bool:
        return self.speed_ms < STOPPED_MS


def profile(track: Sequence[GpsPoint], *, min_dt_s: float = 60.0,
            max_dt_s: float = 3600.0) -> list[Effort]:
    """Speed and altitude-gain rate between consecutive fixes.

    Pairs closer together than ``min_dt_s`` are skipped: over a few seconds GPS
    jitter dominates and the speed is noise. Pairs further apart than
    ``max_dt_s`` are skipped too -- across a gap of hours the average says
    nothing about any moment inside it.
    """
    out: list[Effort] = []
    ordered = sorted([p for p in track if p.ts], key=lambda p: p.ts)
    stopped_run = 0.0
    for prev, cur in zip(ordered, ordered[1:]):
        dt = (cur.ts - prev.ts).total_seconds()
        if dt < min_dt_s or dt > max_dt_s:
            stopped_run = 0.0
            continue
        speed = haversine_m(prev.lat, prev.lon, cur.lat, cur.lon) / dt
        if speed > MAX_WALK_MS:
            stopped_run = 0.0
            continue                       # in a vehicle, not on foot
        gain = 0.0
        if prev.ele is not None and cur.ele is not None:
            gain = (cur.ele - prev.ele) / (dt / 3600.0)
        stopped_run = stopped_run + dt if speed < STOPPED_MS else 0.0
        out.append(Effort(cur.ts, speed, gain, cur.ele, stopped_run))
    return out


def exertion(e: Effort) -> float:
    """How hard this moment was, 0..1.

    Two things count and they are not the same. Climbing steeply is effort even
    at a reasonable pace. Moving slowly *while* climbing is effort at its
    limit -- which at altitude is what the last hour to a pass looks like, and
    is the footage the film most wants.

    A long stop scores too. Nobody stands still for twenty minutes on a cold
    trail because things are going well.
    """
    gain = max(0.0, e.gain_m_per_h) / STEEP_GAIN_M_PER_H
    climb = min(1.0, gain)
    # slowness only counts while climbing; ambling downhill is not effort
    slow = max(0.0, 1.0 - e.speed_ms / MAX_WALK_MS) if gain > 0.15 else 0.0
    stop = min(1.0, e.stopped_s / 1200.0)          # twenty minutes
    return float(min(1.0, 0.55 * climb + 0.30 * slow + 0.15 * stop))


def nepal_hour(ts: datetime, utc_offset_h: float = 5.75) -> float:
    """Local hour as a float. Nepal runs at UTC+05:45."""
    local = ts.astimezone(timezone.utc) + timedelta(hours=utc_offset_h)
    return local.hour + local.minute / 60.0 + local.second / 3600.0


def light_quality(ts: datetime, utc_offset_h: float = 5.75) -> float:
    """How good the light is, 0..1, from the local clock alone.

    Solar position would be more exact, but the trek is two weeks in May at one
    longitude, so the sun rises within a few minutes of the same local time
    every day and the bands below are as accurate as the calculation would be.
    Worth revisiting only if the pipeline is ever pointed at a corpus that
    spans seasons.

    The alpine start scores highest: a pre-dawn push to a pass is inherently
    dramatic whatever the light is doing.
    """
    h = nepal_hour(ts, utc_offset_h)
    if 3.0 <= h < 5.0:
        return 1.0                     # alpine start, head torches
    if 5.0 <= h < 7.0:
        return 0.9                     # sunrise
    if 17.0 <= h < 18.5:
        return 0.85                    # golden hour
    if 18.5 <= h < 20.0:
        return 0.6                     # dusk
    if 10.0 <= h < 15.0:
        return 0.25                    # flat midday sun, the worst of it
    return 0.45


def at(prof: Sequence[Effort], ts: datetime, *, tolerance_s: float = 900.0
       ) -> Effort | None:
    """The nearest effort sample to a moment, or None if none is near enough."""
    if not prof or ts is None:
        return None
    best = min(prof, key=lambda e: abs((e.ts - ts).total_seconds()))
    return best if abs((best.ts - ts).total_seconds()) <= tolerance_s else None


def hardest_windows(prof: Sequence[Effort], *, count: int = 4,
                    window_s: float = 15.0, min_gap_s: float = 120.0
                    ) -> list[tuple[datetime, datetime]]:
    """When to drop the music out entirely.

    Returns the ``count`` hardest moments as time windows, far enough apart that
    the film does not fall silent twice in the same minute. These are the
    passages where location sound carries alone -- breathing, wind, boots -- and
    they are chosen by effort rather than by beauty, because the point is to let
    the audience feel the cost rather than admire the view.
    """
    ranked = sorted(prof, key=lambda e: (-exertion(e), e.ts))
    chosen: list[Effort] = []
    for e in ranked:
        if len(chosen) >= count:
            break
        if all(abs((e.ts - c.ts).total_seconds()) >= min_gap_s for c in chosen):
            chosen.append(e)
    half = timedelta(seconds=window_s / 2.0)
    return [(e.ts - half, e.ts + half) for e in sorted(chosen, key=lambda e: e.ts)]
