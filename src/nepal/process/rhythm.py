"""Film v2 step 4 -- re-time an act's slots against its own music (spec
section 4.3, task 8). Task 9 wires this from ``config/pipeline.yaml``; this
module reads nothing, shells out to nothing, and knows nothing about a
database -- every tunable is a keyword argument, the same discipline
pairs.py and longtake.py follow so the rhythm logic is testable with no
media present.

The idea: a section's energy, ranked against the rest of its own act, says
how long a cut should hold -- loud stays short, quiet stays long -- and the
act's single loudest swell gets its own treatment, a fast run of one-beat
cuts. Every length this module produces is a *candidate*; ``shot_available_s``
(from assemble.py) is consulted last and always wins, per the project's own
trap: a slot must never claim more footage than its shot has, because ffmpeg
simply stops at the end of the source and every number derived from the
timeline downstream of that claim would be wrong.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from nepal.process.assemble import shot_available_s

# The config comment's own thresholds ("< 33", "33-66", "> 66"): reused both
# for target_length's band choice and for retime's beat-vs-downbeat snap,
# since "high" energy is exactly the case the config asks to cut on downbeats.
LOW_MID_BOUNDARY_PCT = 33.0
MID_HIGH_BOUNDARY_PCT = 66.0


def _percentile_rank_pct(value: float, values: Sequence[float]) -> float:
    """Percentile rank of ``value`` within ``values``, on a 0-100 scale --
    the same mid-rank convention as spine/music.py's ``_percentile_rank``
    (entries strictly below count fully, ties count half), reimplemented
    here rather than imported: that function is a private helper of a module
    another task is editing concurrently, and this module must not couple to
    the spine package to stay pure. An empty act (no sections) has nothing
    to rank against; the caller never asks this with an empty list, but 0.0
    is the same "nothing to say" value used for a missing ``energy``."""
    if not values:
        return 0.0
    below = sum(1 for v in values if v < value)
    equal = sum(1 for v in values if v == value)
    return 100.0 * (below + 0.5 * equal) / len(values)


def energy_percentiles(sections: Sequence[Mapping[str, Any]]) -> list[float]:
    """Each section's energy percentile within the act's own cue. A section
    with no ``energy`` reads as 0.0 -- silence, not "unknown" -- so it always
    ranks at or below every measured section rather than skewing the rest
    of the act's ranking by dropping out of it."""
    energies = [float(s.get("energy") or 0.0) for s in sections]
    return [_percentile_rank_pct(e, energies) for e in energies]


def target_length(pct: float, table: Mapping[str, Sequence[float]]) -> tuple[float, float]:
    """The low/mid/high length range for a section at percentile ``pct``,
    by the config's own bands."""
    if pct < LOW_MID_BOUNDARY_PCT:
        band = "low"
    elif pct > MID_HIGH_BOUNDARY_PCT:
        band = "high"
    else:
        band = "mid"
    lo, hi = table[band]
    return float(lo), float(hi)


def _section_pct_at(t: float, section_pcts: Sequence[tuple[Mapping[str, Any], float]]
                    ) -> tuple[Mapping[str, Any] | None, float]:
    """The section covering instant ``t`` (half-open: t_in <= t < t_out) and
    its percentile. A gap the sections don't cover -- before the first cue,
    or a hole between them -- has no energy to read, so it ranks as 0.0,
    the same "nothing to say" value a missing ``energy`` gets."""
    for section, pct in section_pcts:
        if float(section["t_in"]) <= t < float(section["t_out"]):
            return section, pct
    return None, 0.0


def _normal_end(t_in: float, pct: float, table: Mapping[str, Sequence[float]],
                avail: float, beats: Sequence[float], downbeats: Sequence[float]) -> float:
    """An unlocked slot's end when it isn't part of the burst: the band's
    midpoint, clamped to what the shot actually has, then snapped to the
    nearest qualifying beat (downbeat above the 66th percentile). A beat
    that would overrun the footage is not a candidate at all -- the clamp is
    the last word, not something a snap gets to override -- and a beat that
    would land at or before ``t_in`` can't produce a slot with any length,
    so both leave the clamped, unsnapped end standing."""
    lo, hi = target_length(pct, table)
    length = min((lo + hi) / 2.0, avail)
    want_end = t_in + length
    grid = downbeats if pct > MID_HIGH_BOUNDARY_PCT else beats
    ceiling = t_in + avail
    candidates = [b for b in grid if b <= ceiling]
    if candidates:
        nearest = min(candidates, key=lambda b: abs(b - want_end))
        if nearest > t_in:
            return nearest
    return want_end


def _burst_end(t_in: float, beats: Sequence[float], ceiling: float) -> float | None:
    """The end of one beat starting at or after ``t_in``: the beat at or
    after ``t_in``, to the next beat after that -- clamped to ``ceiling``
    (``t_in`` plus what the shot has left). None when the grid can't supply
    one -- no beat at or after ``t_in``, no further beat after that, or the
    only next beat would overrun the footage -- so the caller falls back to
    treating this slot normally instead of inventing a length the grid never
    offered."""
    after = sorted(b for b in beats if b >= t_in)
    if len(after) < 2:
        return None
    end = after[1]
    return end if end <= ceiling else None


def _apply_held_shot(out: list[dict[str, Any]], *, shots: Mapping[str, Mapping[str, Any]],
                     held_shot_s: Sequence[float], silence_t: float) -> None:
    """Act 4's ending: the last unlocked slot before the silence holds up to
    it. The first choice is to reach exactly ``silence_t``, so the cut to
    silence lands clean; when the shot doesn't have that much left, it holds
    instead for ``held_shot_s``'s midpoint -- clamped by what's left, same as
    everywhere else in this module -- and falls short of the silence, which
    is the caller's gap to re-fill, not this function's to invent footage for."""
    candidates = [s for s in out if not s.get("locked") and float(s["t_in"]) < silence_t]
    if not candidates:
        return
    slot = candidates[-1]
    shot = shots.get(slot.get("shot_id"))
    avail = shot_available_s(shot) if shot is not None else math.inf
    t_in = float(slot["t_in"])
    reach = silence_t - t_in
    if avail >= reach:
        slot["t_out"] = silence_t
    else:
        held_mid = (float(held_shot_s[0]) + float(held_shot_s[1])) / 2.0
        slot["t_out"] = t_in + min(held_mid, avail)


def retime(slots: Sequence[Mapping[str, Any]], *, sections: Sequence[Mapping[str, Any]],
          beats: Sequence[float], downbeats: Sequence[float],
          table: Mapping[str, Sequence[float]], burst_slots: Sequence[int],
          is_act4: bool, held_shot_s: Sequence[float], silence_t: float | None,
          shots: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Walk an act's slots in order and give each an end that fits its
    section's energy, its shot's footage and the beat grid.

    A locked slot (the bridge crossing, a pair) keeps both its own bounds
    untouched -- the walk only resumes counting from its ``t_out``. Every
    other slot is contiguous: the act's first slot keeps its own ``t_in``,
    and each slot after it starts where the previous one ended, so the
    re-timing never opens a gap or an overlap on its own. The one exception
    the spec asks for is the act's loudest swell, which becomes a run of
    one-beat cuts (a "burst") instead of section-banded ones; Act 4 gets a
    second exception, a shot held up to the silence at the end.

    Order matters here: the nominal length is decided first (band midpoint,
    or one beat inside the burst), the beat snap is applied second, and
    ``shot_available_s`` is consulted at every step and is the final word --
    a snap is never allowed to hand a slot more footage than its shot has.
    """
    section_pcts = [(s, pct) for s, pct in zip(sections, energy_percentiles(sections))]
    swell = max(sections, key=lambda s: float(s.get("energy") or 0.0), default=None)
    act_end = float(sections[-1]["t_out"]) if sections else math.inf
    burst_count = round((float(burst_slots[0]) + float(burst_slots[1])) / 2.0)

    out: list[dict[str, Any]] = []
    t = float(slots[0]["t_in"]) if slots else 0.0
    burst_started = False
    burst_remaining = burst_count

    for slot in slots:
        if slot.get("locked"):
            new_slot = dict(slot)
            if float(new_slot["t_in"]) >= act_end:
                continue                    # off the end of the act; the caller re-fills the gap
            out.append(new_slot)
            t = float(new_slot["t_out"])
            continue

        t_in = t
        if t_in >= act_end:
            continue                        # this and every later slot fall off the act's end

        section, pct = _section_pct_at(t_in, section_pcts)
        # A slot whose shot never made it into `shots` can't be measured
        # against its footage -- it is left at its nominal length rather
        # than treated as having none, since inventing a clamp this module
        # was given no data for is worse than leaving the length alone.
        shot = shots.get(slot.get("shot_id"))
        avail = shot_available_s(shot) if shot is not None else math.inf

        if swell is not None and section is swell and not burst_started:
            burst_started = True

        t_out = None
        if burst_started and burst_remaining > 0:
            t_out = _burst_end(t_in, beats, t_in + avail)
            if t_out is not None:
                burst_remaining -= 1
        if t_out is None:
            t_out = _normal_end(t_in, pct, table, avail, beats, downbeats)

        new_slot = dict(slot)
        new_slot["t_in"], new_slot["t_out"] = round(t_in, 3), round(t_out, 3)
        out.append(new_slot)
        t = new_slot["t_out"]

    if is_act4 and silence_t is not None:
        _apply_held_shot(out, shots=shots, held_shot_s=held_shot_s, silence_t=float(silence_t))

    return out
