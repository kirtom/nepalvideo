"""Act boundary segmentation.

Listed in the spec's auto-solved table as "altitude profile segmentation +
message phase classification", with an even split by day count as the
fallback. Altitude is the dramatic axis, so the boundaries come off the
altitude curve rather than off the calendar:

  Act 1 Planning      everything before the track starts
  Act 2 Approach      the low ground -- villages, tea houses, warm-up
  Act 3 The climb     from where the ascent steepens to the summit push
  Act 4 Highest point the narrow band around maximum altitude
  Act 5 Descent       from the summit to the end, plus the return home

The Act 2/3 split is the only genuinely ambiguous one. It is placed at the
change point in the altitude-vs-day curve: the day whose two-segment linear
fit leaves the least error, subject to the second segment being steeper.
That matches what the brief asks for -- Act 3 should "feel like it costs
something" -- without hard-coding an altitude threshold that would be wrong
for a different trek.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Sequence

log = logging.getLogger(__name__)

NEPAL_TZ = timezone(timedelta(hours=5, minutes=45))


@dataclass
class DayStat:
    day_index: int
    date: str
    alt_max: float | None
    alt_min: float | None
    start_utc: datetime
    end_utc: datetime


@dataclass
class ActBoundary:
    act: int
    start_utc: datetime
    end_utc: datetime
    method: str = ""

    def as_dict(self) -> dict:
        return {"act": self.act,
                "start_utc": self.start_utc.astimezone(timezone.utc).isoformat(),
                "end_utc": self.end_utc.astimezone(timezone.utc).isoformat(),
                "method": self.method}


def _fit_error(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    """Least-squares line through points; returns (sum squared error, slope)."""
    n = len(xs)
    if n < 2:
        return 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return sum((y - my) ** 2 for y in ys), 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    err = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    return err, slope


def find_climb_start(days: Sequence[DayStat], summit_idx: int) -> int:
    """Index into ``days`` where the approach becomes the climb.

    Two-segment fit over the ascent, choosing the break that minimises total
    residual error while leaving the upper segment steeper than the lower.
    """
    ascent = [d for d in days[: summit_idx + 1] if d.alt_max is not None]
    if len(ascent) < 4:
        return max(1, summit_idx // 2)

    xs = [float(d.day_index) for d in ascent]
    ys = [float(d.alt_max) for d in ascent]

    best_i, best_err = None, float("inf")
    for i in range(2, len(ascent) - 1):
        e1, s1 = _fit_error(xs[:i], ys[:i])
        e2, s2 = _fit_error(xs[i - 1:], ys[i - 1:])
        if s2 <= s1:
            continue                       # the climb must be steeper than the approach
        if e1 + e2 < best_err:
            best_i, best_err = i, e1 + e2

    if best_i is None:                     # a uniform grind with no break point
        best_i = max(2, len(ascent) // 2)
    return days.index(ascent[best_i - 1])


def segment_acts(days: Sequence[DayStat], *,
                 planning_start: datetime | None = None,
                 after_end: datetime | None = None,
                 summit_band: float = 0.97) -> list[ActBoundary]:
    """Five act boundaries from the altitude profile.

    ``summit_band`` sets how wide Act 4 is: days reaching at least this
    fraction of the maximum altitude. Act 4 is only 1.5-2 minutes of film, so
    it should be a narrow band, not a whole phase.
    """
    days = sorted(days, key=lambda d: d.day_index)
    dated = [d for d in days if d.alt_max is not None]

    if not days:
        return []
    if not dated:
        return _even_split(days, planning_start, after_end, "fallback:even-split(no altitude)")

    trek_start, trek_end = days[0].start_utc, days[-1].end_utc
    act1_start = planning_start or (trek_start - timedelta(days=30))
    act5_end = after_end or trek_end

    alt_max = max(d.alt_max for d in dated)
    summit_days = [d for d in dated if d.alt_max >= alt_max * summit_band]
    summit_first, summit_last = summit_days[0], summit_days[-1]
    summit_idx = days.index(summit_first)

    if summit_idx == 0:
        # the trek starts at its high point -- no approach or climb to find
        return _even_split(days, planning_start, after_end,
                           "fallback:even-split(summit on day 1)")

    climb_idx = find_climb_start(days, summit_idx)
    climb_idx = max(1, min(climb_idx, summit_idx))

    method = f"altitude-changepoint(max={alt_max:.0f}m, summit=day {summit_first.day_index})"
    bounds = [
        ActBoundary(1, act1_start, trek_start, "message-phase:planning"),
        ActBoundary(2, trek_start, days[climb_idx].start_utc, method),
        ActBoundary(3, days[climb_idx].start_utc, summit_first.start_utc, method),
        ActBoundary(4, summit_first.start_utc, summit_last.end_utc, method),
        ActBoundary(5, summit_last.end_utc, act5_end, method),
    ]
    return _repair(bounds)


def _even_split(days: Sequence[DayStat], planning_start, after_end,
                method: str) -> list[ActBoundary]:
    """The spec's stated fallback: even split by day count."""
    trek_start, trek_end = days[0].start_utc, days[-1].end_utc
    n = len(days)
    cuts = [days[min(n - 1, round(n * f))].start_utc for f in (0.25, 0.6, 0.85)]
    return _repair([
        ActBoundary(1, planning_start or trek_start - timedelta(days=30), trek_start, method),
        ActBoundary(2, trek_start, cuts[0], method),
        ActBoundary(3, cuts[0], cuts[1], method),
        ActBoundary(4, cuts[1], cuts[2], method),
        ActBoundary(5, cuts[2], after_end or trek_end, method),
    ])


def _repair(bounds: list[ActBoundary]) -> list[ActBoundary]:
    """Guarantee non-decreasing, non-empty, contiguous boundaries.

    A degenerate profile -- one trek day, or a summit on the first day -- can
    otherwise produce an act that starts after it ends, which would silently
    drop every shot in it.
    """
    out = sorted(bounds, key=lambda b: b.act)
    for i, b in enumerate(out):
        if b.end_utc < b.start_utc:
            b.end_utc = b.start_utc
        if i > 0 and b.start_utc < out[i - 1].end_utc:
            b.start_utc = out[i - 1].end_utc
            if b.end_utc < b.start_utc:
                b.end_utc = b.start_utc
    return out


def act_for(ts: datetime, bounds: Sequence[ActBoundary]) -> int | None:
    """Which act a moment falls in. Later acts win ties on a shared edge, so a
    shot exactly on a boundary belongs to the act it opens."""
    chosen = None
    for b in sorted(bounds, key=lambda b: b.act):
        if b.start_utc <= ts <= b.end_utc:
            chosen = b.act
    if chosen is not None:
        return chosen
    first, last = min(bounds, key=lambda b: b.act), max(bounds, key=lambda b: b.act)
    if ts < first.start_utc:
        return first.act
    if ts > last.end_utc:
        return last.act
    return None
