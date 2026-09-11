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
    asset_count: int = 0


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


def trek_window(days: Sequence[DayStat], *, max_gap_days: int = 3,
                min_days: int = 3) -> tuple[list[DayStat], list[DayStat]]:
    """Split days into the trek itself and everything else.

    A trek is a contiguous burst of intense capture. Everything around it --
    planning photos taken at home over the preceding months, a few shots after
    getting back, and any material whose timestamp is simply wrong -- is sparse
    and scattered by comparison.

    Segmenting over all of it produces nonsense. On real material: months of
    home photos carrying GPS stretched the envelope so that Act 1 collapsed to
    41 minutes while Act 2 swallowed eleven weeks, and a single day of footage
    dated eighteen months later (a camera whose clock had reset) pushed Act 4
    across the entire gap and left Act 5 with zero duration.

    So the trek is taken to be the run of days with the most material, where
    consecutive days are no more than ``max_gap_days`` apart. Returns
    (trek_days, excluded_days); the excluded days still supply Act 1's planning
    material and Act 5's homecoming, they just do not define the structure.
    """
    ordered = sorted(days, key=lambda d: d.day_index)
    if not ordered:
        return [], []

    runs: list[list[DayStat]] = [[ordered[0]]]
    for prev, cur in zip(ordered, ordered[1:]):
        if cur.day_index - prev.day_index <= max_gap_days:
            runs[-1].append(cur)
        else:
            runs.append([cur])

    def weight(run: list[DayStat]) -> tuple[int, int]:
        # material first, then length: a long sparse stretch of planning photos
        # must not outrank a short dense trek
        return (sum(d.asset_count for d in run), len(run))

    best = max(runs, key=weight)
    if len(best) < min_days and len(ordered) >= min_days:
        best = max(runs, key=lambda r: (len(r), weight(r)))

    excluded = [d for d in ordered if d not in best]
    if excluded:
        log.info("S02 trek window: days %d-%d (%s .. %s), %d asset(s); "
                 "%d day(s) outside it supply Act 1 and Act 5 but do not define structure",
                 best[0].day_index, best[-1].day_index, best[0].date, best[-1].date,
                 sum(d.asset_count for d in best), len(excluded))
    return best, excluded


def _contiguous_summit_run(days: Sequence[DayStat], dated: Sequence[DayStat],
                           alt_max: float, summit_band: float) -> list[DayStat]:
    """The unbroken run of high days around the actual maximum.

    Taking every day above the threshold is wrong whenever the corpus contains
    an outlier: a single stray day at altitude -- a camera whose clock is out by
    months, footage from a different trip, a revisit a year later -- becomes the
    last member of the band, so Act 4 stretches from the real summit to that
    day and Act 5 collapses to nothing. Observed on real material as an Act 4
    spanning eighteen months and an Act 5 of zero duration.

    The summit is one continuous period, so the run is grown outward from the
    day of maximum altitude and stops at the first day that drops below the
    band. A genuine multi-day summit push is kept; a stray day is not.
    """
    if not dated:
        return []
    threshold = alt_max * summit_band
    peak = max(dated, key=lambda d: d.alt_max)
    order = [d for d in days if d.alt_max is not None]
    try:
        i = order.index(peak)
    except ValueError:
        return [peak]

    lo = i
    while lo - 1 >= 0 and order[lo - 1].alt_max >= threshold:
        lo -= 1
    hi = i
    while hi + 1 < len(order) and order[hi + 1].alt_max >= threshold:
        hi += 1

    run = order[lo:hi + 1]
    excluded = [d for d in order if d.alt_max >= threshold and d not in run]
    if excluded:
        log.warning("S02 %d day(s) reach the summit band but are not contiguous with "
                    "the peak and are excluded from Act 4: %s. Check for a clock error "
                    "or footage from another trip.",
                    len(excluded), ", ".join(f"{d.date} ({d.alt_max:.0f}m)"
                                             for d in excluded[:5]))
    return run


def segment_acts(days: Sequence[DayStat], *,
                 planning_start: datetime | None = None,
                 after_end: datetime | None = None,
                 summit_band: float = 0.97,
                 max_gap_days: int = 3,
                 planning_window_days: int = 180,
                 after_window_days: int = 30) -> list[ActBoundary]:
    """Five act boundaries from the altitude profile.

    ``summit_band`` sets how wide Act 4 is: days reaching at least this
    fraction of the maximum altitude. Act 4 is only 1.5-2 minutes of film, so
    it should be a narrow band, not a whole phase.
    """
    all_days = sorted(days, key=lambda d: d.day_index)
    if not all_days:
        return []

    # Structure comes from the trek itself. Material outside it still lands in
    # Act 1 or Act 5 by time, but it must not move the boundaries.
    days, outside = trek_window(all_days, max_gap_days=max_gap_days)
    if not days:
        days, outside = all_days, []
    dated = [d for d in days if d.alt_max is not None]
    if not dated:
        return _even_split(days, planning_start, after_end, "fallback:even-split(no altitude)")

    trek_start, trek_end = days[0].start_utc, days[-1].end_utc

    # Act 1 reaches back to the first planning material, Act 5 forward to the
    # homecoming -- but both are bounded. The brief describes Act 1 as starting
    # with the first planning message, which is legitimately months out, and
    # Act 5 as "coming down, the last days, arrival home", which is days. Left
    # unbounded, one mis-timestamped clip eighteen months later turned Act 5
    # into a 566-day act.
    earliest = min([d.start_utc for d in outside] + [trek_start])
    latest = max([d.end_utc for d in outside] + [trek_end])
    plan_floor = trek_start - timedelta(days=planning_window_days)
    after_ceiling = trek_end + timedelta(days=after_window_days)

    act1_start = max(min(planning_start or earliest, earliest), plan_floor)
    act5_end = min(max(after_end or latest, latest), after_ceiling)

    if earliest < plan_floor:
        log.warning("S02 material exists %s, more than %d days before the trek -- "
                    "Act 1 starts at %s instead. Raise planning_window_days to include it.",
                    earliest.date(), planning_window_days, act1_start.date())
    if latest > after_ceiling:
        log.warning("S02 material exists %s, more than %d days after the trek -- "
                    "Act 5 ends at %s instead. A date that far out usually means a "
                    "clock error rather than a late homecoming.",
                    latest.date(), after_window_days, act5_end.date())

    alt_max = max(d.alt_max for d in dated)
    summit_days = _contiguous_summit_run(days, dated, alt_max, summit_band)
    if not summit_days:
        return _even_split(days, planning_start, after_end,
                           "fallback:even-split(no summit run)")
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
                method: str, *, planning_window_days: int = 180,
                after_window_days: int = 30) -> list[ActBoundary]:
    """The spec's stated fallback: even split by day count."""
    trek_start, trek_end = days[0].start_utc, days[-1].end_utc
    if planning_start:
        planning_start = max(planning_start, trek_start - timedelta(days=planning_window_days))
    if after_end:
        after_end = min(after_end, trek_end + timedelta(days=after_window_days))
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
