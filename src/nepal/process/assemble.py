"""S06 -- choose the shots and lay them on the beat grid.

The spec is explicit that the model must not free-form the timeline: selection
is a greedy MMR fill under hard constraints, and Bedrock is used only to
refine ordering within an act and to place caption cards. Everything in this
module is the part that decides, so all of it is pure and none of it needs a
network.

MMR -- maximal marginal relevance -- picks ``argmax(score - lambda * max
similarity to what is already chosen)``. Without it a greedy fill takes the
ten best shots of the same ridge in the same light, because they all score
alike for the same reason. Similarity is the CLIP cosine from S04.1, so
"already said this" is measured on what the frame looks like rather than on
where or when it was taken.

Constraints are filters applied at each step, not a repair pass afterwards.
The spec's order of surrender when a constraint cannot be met is deliberate
and followed here: **relax diversity before chronology**. A film that jumps
backwards in time reads as broken; one that lingers in a place reads as slow.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

log = logging.getLogger(__name__)

MMR_LAMBDA = 0.30


def slot_budget(act_seconds: float, duration_range: Sequence[float]) -> int:
    """How many shots an act's runtime affords, at its mid duration."""
    lo, hi = float(duration_range[0]), float(duration_range[1])
    mid = (lo + hi) / 2.0
    return max(1, int(round(float(act_seconds) / mid))) if mid > 0 else 1


def snap_to_beat(t: float, beats: Sequence[float], *,
                 downbeats: Sequence[float] | None = None) -> float:
    """Nearest beat, or nearest downbeat when one is asked for.

    A cut that lands a few tens of milliseconds off the beat reads as a
    mistake rather than as syncopation, so this always snaps to the nearest
    rather than the next -- moving a cut forward to catch a beat would push
    every following shot and accumulate.
    """
    grid = list(downbeats) if downbeats else list(beats)
    if not grid:
        return float(t)
    arr = np.asarray(grid, dtype=float)
    return float(arr[int(np.argmin(np.abs(arr - float(t))))])


def _similarity(a: str, b: str, embeddings: Mapping[str, np.ndarray]) -> float:
    ea, eb = embeddings.get(a), embeddings.get(b)
    if ea is None or eb is None:
        return 0.0                     # unknown similarity must not veto a shot
    return float(np.dot(ea, eb))


def mmr_select(candidates: Sequence[Mapping[str, Any]], *, budget: int,
               embeddings: Mapping[str, np.ndarray],
               lam: float = MMR_LAMBDA,
               admissible=None) -> list[Mapping[str, Any]]:
    """Greedy MMR fill under a per-step admissibility filter.

    ``admissible(candidate, chosen)`` returns whether a shot may be taken
    given what is already there -- the hard constraints. It is consulted at
    every step rather than checked at the end, because a constraint that can
    only be repaired afterwards is a constraint that reorders the film.

    When nothing is admissible the fill stops rather than forcing a shot in:
    the caller decides what to relax, and the spec says what order to relax in.
    """
    chosen: list[Mapping[str, Any]] = []
    remaining = [c for c in candidates if c.get("score_total") is not None]
    while len(chosen) < budget and remaining:
        best, best_val = None, None
        for c in remaining:
            if admissible is not None and not admissible(c, chosen):
                continue
            penalty = max((_similarity(c["shot_id"], s["shot_id"], embeddings)
                           for s in chosen), default=0.0)
            val = float(c["score_total"]) - lam * penalty
            if best_val is None or val > best_val:
                best, best_val = c, val
        if best is None:
            break
        chosen.append(best)
        remaining.remove(best)
    return chosen


# -- the hard constraints ----------------------------------------------

def chronological(shots: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Order by capture time. Chronology is per act and non-decreasing.

    Shots with no timestamp keep their relative order at the end rather than
    being dropped: an unplaced shot is still footage, and the spec only
    demands chronology of what has a time.
    """
    timed = [s for s in shots if s.get("start_utc")]
    untimed = [s for s in shots if not s.get("start_utc")]
    return sorted(timed, key=lambda s: str(s["start_utc"])) + untimed


def place_count_ok(candidate: Mapping[str, Any], chosen: Sequence[Mapping[str, Any]],
                   *, limit: int) -> bool:
    """No more than ``limit`` shots from one place, per act."""
    place = candidate.get("place_name")
    if not place:
        return True                    # an unnamed place cannot crowd the act
    return sum(1 for s in chosen if s.get("place_name") == place) < limit


def needs_subject(chosen: Sequence[Mapping[str, Any]], *, runtime_s: float,
                  every_s: float) -> bool:
    """Whether the act is short of its person-every-N-seconds quota."""
    if every_s <= 0:
        return False
    want = int(float(runtime_s) // float(every_s))
    have = sum(1 for s in chosen if s.get("has_face"))
    return have < want


def missing_levity(chosen: Sequence[Mapping[str, Any]], *, minimum: int) -> int:
    """How many levity-tagged shots the act still owes."""
    have = sum(1 for s in chosen if s.get("tag_levity"))
    return max(0, int(minimum) - have)


def speech_first(candidates: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Speech-carrying shots ahead of the rest, each group by score.

    Section 1.4 makes speech the spine of the film, and the spec says every
    transcribed moment that survives Gate 2 is placed -- so speech is not a
    scoring bonus to be outweighed, it is a claim on a slot. Ordering the
    candidate list this way means the MMR fill reaches for silence only once
    the voices are in.
    """
    def key(s: Mapping[str, Any]) -> tuple:
        speaks = bool(s.get("has_speech") and (s.get("transcript") or "").strip())
        return (0 if speaks else 1, -float(s.get("score_total") or 0.0))
    return sorted(candidates, key=key)


def lay_out(shots: Sequence[Mapping[str, Any]], *, start_s: float,
            duration_range: Sequence[float], beats: Sequence[float],
            downbeats: Sequence[float] | None = None) -> list[dict[str, Any]]:
    """Give each shot a place on the timeline, snapped to the grid.

    Durations come from the act's range, scaled by the shot's own score so a
    stronger shot is held longer -- within the act's range, never outside it.
    The first cut of an act snaps to a downbeat; the rest to any beat.
    """
    lo, hi = float(duration_range[0]), float(duration_range[1])
    out: list[dict[str, Any]] = []
    t = float(start_s)
    for i, s in enumerate(shots):
        score = float(s.get("score_total") or 0.5)
        want = lo + (hi - lo) * max(0.0, min(1.0, score))
        end = snap_to_beat(t + want, beats,
                           downbeats=downbeats if i == 0 else None)
        if end <= t:                    # the grid was too coarse to advance
            end = t + want
        src_in = float(s.get("start_s") or 0.0)
        out.append({"shot_id": s["shot_id"], "act": s.get("act"),
                    "t_in": round(t, 3), "t_out": round(end, 3),
                    "src_in": round(src_in, 3),
                    "src_out": round(src_in + (end - t), 3),
                    "yaw": s.get("chosen_yaw")})
        t = end
    return out
