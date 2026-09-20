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
    so both leave the clamped, unsnapped end standing.

    Not ``assemble.snap_to_beat``/``snap_within`` (already imported into this
    module by way of ``shot_available_s``'s neighbours): ``snap_to_beat`` has
    no ceiling at all, and ``snap_within``'s own fallback is the ceiling
    itself, which would stretch every clamped-but-unsnapped slot out to the
    full available footage instead of to its own band's nominal length."""
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


def _slot_avail_s(slot: Mapping[str, Any], shot: Mapping[str, Any] | None) -> float:
    """How much footage this specific slot can actually claim.

    ``shot_available_s`` caps by the shot's own ``start_s``/``end_s``, but a
    slot's own ``src_in`` may already sit partway into that footage --
    pairs.py's split slots set ``src_in`` to ``start_s`` plus however far the
    two streams had to align to share an instant -- so the real ceiling is
    ``end_s - src_in``, which can only be tighter, never looser, than
    ``shot_available_s``. A split slot has a second, independent ceiling
    besides: the overlap ``find_pairs`` already measured between the two
    streams (``src_out - src_in``, computed once at pairing time), since a
    beat snap may stretch a split no further than either camera actually
    has at the aligned instant.
    """
    if shot is None:
        return math.inf
    avail = shot_available_s(shot)
    if avail == math.inf:
        return avail                        # a photo (or a shot with no end_s) has nothing to tighten
    src_in = slot.get("src_in")
    if src_in is not None:
        avail = min(avail, float(shot["end_s"]) - float(src_in))
        if slot.get("secondary_shot_id") is not None and slot.get("src_out") is not None:
            avail = min(avail, float(slot["src_out"]) - float(src_in))
    return max(0.0, avail)


def _apply_held_shot(out: list[dict[str, Any]], *, shots: Mapping[str, Mapping[str, Any]],
                     held_shot_s: Sequence[float], silence_t: float) -> None:
    """Act 4's ending: the last unlocked slot before the silence is
    stretched to end exactly on it.

    A locked slot carries a beat the rhythm pass may never cut, so a locked
    slot anywhere between a candidate and the silence disqualifies it --
    scanning forward, every locked slot before the silence resets the
    candidate, so only the very last qualifying slot (if any) is ever
    eligible.

    The hold's length is ``held_shot_s``'s own midpoint, clamped to what the
    shot has left -- not the distance back to the silence, which would make
    the config value dead weight in every act whose last shot has room to
    spare. ``t_out`` is always the silence point; only ``t_in`` (and so the
    hold's actual length) gives way -- first to the shot's own footage, then,
    if that still reaches earlier than the previous slot's own end, to the
    previous slot's territory, since this pass may not open an overlap to
    buy itself a longer last breath. Whatever the hold can't reach is the
    caller's gap to re-fill, not this function's to invent footage for.
    """
    candidate, idx = None, None
    for i, s in enumerate(out):
        if float(s["t_in"]) >= silence_t:
            continue
        if s.get("locked"):
            candidate, idx = None, None     # a locked slot sits between here and the silence
        else:
            candidate, idx = s, i
    if candidate is None:
        return

    shot = shots.get(candidate.get("shot_id"))
    avail = _slot_avail_s(candidate, shot)
    held_mid = (float(held_shot_s[0]) + float(held_shot_s[1])) / 2.0
    held = min(held_mid, avail)
    prev_end = float(out[idx - 1]["t_out"]) if idx > 0 else -math.inf
    if silence_t - held < prev_end:
        held = silence_t - prev_end         # shrink the hold rather than overlap the previous slot

    candidate["t_in"] = round(silence_t - held, 3)
    candidate["t_out"] = round(silence_t, 3)
    if candidate.get("src_in") is not None:
        candidate["src_out"] = round(float(candidate["src_in"]) + held, 3)


def retime(slots: Sequence[Mapping[str, Any]], *, sections: Sequence[Mapping[str, Any]],
          beats: Sequence[float], downbeats: Sequence[float],
          table: Mapping[str, Sequence[float]], burst_slots: Sequence[int],
          is_act4: bool, held_shot_s: Sequence[float], silence_t: float | None,
          shots: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Walk an act's slots in order and give each an end that fits its
    section's energy, its shot's footage and the beat grid.

    A locked slot (the bridge crossing) keeps both its own bounds
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
    # A `None` t_in on the very first slot is the shape S06's lay_out (via
    # SLOT_KEYS) hands a slot before anything has placed it: it starts the
    # act at the act's own first cue rather than crashing on `float(None)`.
    if not slots:
        t = 0.0
    elif slots[0].get("t_in") is not None:
        t = float(slots[0]["t_in"])
    elif sections:
        t = float(sections[0]["t_in"])
    else:
        t = 0.0
    burst_started = False
    burst_remaining = burst_count

    for slot in slots:
        if slot.get("locked"):
            if slot.get("t_in") is None or slot.get("t_out") is None:
                # A locked slot is placed by whoever built it (the bridge
                # crossing, a pair); one with no bounds at all is that
                # caller's bug, not something to paper over with a cast
                # that would otherwise fail deep inside float(None).
                raise ValueError(f"locked slot {slot.get('shot_id')!r} has no t_in/t_out")
            new_slot = dict(slot)
            if float(new_slot["t_in"]) >= act_end:
                continue                    # off the end of the act; the caller re-fills the gap
            out.append(new_slot)
            t = float(new_slot["t_out"])
            continue

        t_in = t
        if t_in >= act_end:
            # This one unlocked slot missed the act's end; the walk pointer
            # doesn't move, so every unlocked slot after it will too --
            # unless a later LOCKED slot (which ignores the walk pointer
            # entirely) lands before the act's end and pulls it back.
            continue

        section, pct = _section_pct_at(t_in, section_pcts)
        # A slot whose shot never made it into `shots` can't be measured
        # against its footage -- it is left at its nominal length rather
        # than treated as having none, since inventing a clamp this module
        # was given no data for is worse than leaving the length alone.
        shot = shots.get(slot.get("shot_id"))
        avail = _slot_avail_s(slot, shot)

        if swell is not None and section is swell and not burst_started:
            burst_started = True

        if burst_started and burst_remaining > 0:
            t_out = _burst_end(t_in, beats, t_in + avail)
            if t_out is None:
                # Inside the swell the burst budget counts cuts attempted,
                # not successful snaps -- a starved grid still spends one,
                # and takes the fastest length the config allows (the high
                # band's own floor) rather than the section's own, likely
                # much longer, band: a slow cut is exactly wrong at the
                # film's most intense moment.
                t_out = t_in + min(float(table["high"][0]), avail)
            burst_remaining -= 1
        else:
            t_out = _normal_end(t_in, pct, table, avail, beats, downbeats)

        if t_out <= t_in:
            continue                        # the clamp (or a starved grid) left no length at all; drop it rather than stall the walk

        new_slot = dict(slot)
        new_slot["t_in"], new_slot["t_out"] = round(t_in, 3), round(t_out, 3)
        if new_slot.get("src_in") is not None:
            # render.py cuts `-ss src_in -t (t_out - t_in)`; src_out is
            # descriptive, not consulted by ffmpeg, but it must still agree
            # with the film span this slot now actually occupies.
            new_slot["src_out"] = round(float(new_slot["src_in"]) +
                                        (new_slot["t_out"] - new_slot["t_in"]), 3)
        out.append(new_slot)
        t = new_slot["t_out"]

    if is_act4 and silence_t is not None:
        _apply_held_shot(out, shots=shots, held_shot_s=held_shot_s, silence_t=float(silence_t))

    return out
