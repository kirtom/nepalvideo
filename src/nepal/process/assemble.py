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
import math
from datetime import datetime
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from nepal import db

log = logging.getLogger(__name__)

MMR_LAMBDA = 0.30

SLOT_KEYS = db.TIMELINE_V2_COLUMNS + ("locked",)


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


def parse_utc(iso: Any) -> datetime | None:
    """An ISO 8601 capture stamp as a ``datetime``, or None for anything that
    is not one -- a missing or malformed stamp is a fact about the shot, not
    an error the modules that place shots in time may raise: a shot with no
    clock still reaches the timeline, it just cannot be placed by chronology
    or paired with another camera.

    Two entry points over one parse because the callers want two different
    things back: ``epoch_utc`` for every comparison and fraction the
    placement math does, this one for the arithmetic that has to come back
    out as a stamp (``anchors.speech_anchors`` shifts a shot's start by the
    beat's offset into it and re-serialises the result).
    """
    if not iso:
        return None
    try:
        return datetime.fromisoformat(str(iso))
    except ValueError:
        return None


def epoch_utc(iso: Any) -> float | None:
    """``parse_utc`` in epoch seconds."""
    dt = parse_utc(iso)
    return None if dt is None else dt.timestamp()


def fallback_similarity(a: Mapping[str, Any], b: Mapping[str, Any], *,
                        weights: Mapping[str, float]) -> float:
    """What two shots share when nothing has looked at their pictures. The
    first draft's runs of one recording came from a similarity of 0.0 for
    every pair; this alone breaks them up.

    ``weights`` is indexed, not ``.get``-with-a-default: the three numbers
    live in `assemble.similarity_fallback` and nowhere else, so a key that
    goes missing there must raise with its own name rather than quietly
    restore a value this function happens to remember.
    """
    if a.get("recording_id") and a.get("recording_id") == b.get("recording_id"):
        ta, tb = epoch_utc(a.get("start_utc")), epoch_utc(b.get("start_utc"))
        if ta is not None and tb is not None and abs(ta - tb) <= 60.0:
            return float(weights["same_recording_60s"])
        return float(weights["same_recording"])
    if a.get("place_name") and a.get("place_name") == b.get("place_name"):
        ta, tb = epoch_utc(a.get("start_utc")), epoch_utc(b.get("start_utc"))
        if ta is not None and tb is not None and abs(ta - tb) <= 3600.0:
            return float(weights["same_place_hour"])
    return 0.0


def make_similarity(embeddings: Mapping[str, np.ndarray], *,
                    fallback_weights: Mapping[str, float]
                    ) -> Callable[[Mapping[str, Any], Mapping[str, Any]], float]:
    """CLIP cosine when both shots have an embedding, else the deterministic
    fallback -- so a shortlist that ran thin on embeddings still gets MMR's
    diversity pressure instead of silently degrading to score order."""
    def sim(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
        ea, eb = embeddings.get(a.get("shot_id")), embeddings.get(b.get("shot_id"))
        if ea is not None and eb is not None:
            return float(np.dot(ea, eb))
        return fallback_similarity(a, b, weights=fallback_weights)
    return sim


def mmr_select(candidates: Sequence[Mapping[str, Any]], *, budget: int,
               similarity: Callable[[Mapping[str, Any], Mapping[str, Any]], float],
               lam: float = MMR_LAMBDA,
               admissible=None,
               prefer=None,
               seed: Sequence[Mapping[str, Any]] = ()) -> list[Mapping[str, Any]]:
    """Greedy MMR fill under a per-step admissibility filter.

    ``admissible(candidate, chosen)`` returns whether a shot may be taken
    given what is already there -- the hard constraints. It is consulted at
    every step rather than checked at the end, because a constraint that can
    only be repaired afterwards is a constraint that reorders the film.

    ``prefer(candidate, chosen)`` is a soft bonus added to the MMR value --
    used for source alternation, which should nudge the fill rather than
    veto a shot the way ``admissible`` does.

    When nothing is admissible the fill stops rather than forcing a shot in:
    the caller decides what to relax, and the spec says what order to relax in.

    ``seed`` is what an earlier pass already chose: the diversity penalty is
    taken over it as well as over this call's own picks, so a relaxed second
    pass is still penalised for resembling the first, while the returned
    list and the ``admissible``/``prefer`` calls see this call's picks alone.
    """
    chosen: list[Mapping[str, Any]] = []
    remaining = [c for c in candidates if c.get("score_total") is not None]
    while len(chosen) < budget and remaining:
        best, best_val = None, None
        for c in remaining:
            if admissible is not None and not admissible(c, chosen):
                continue
            penalty = max((similarity(c, s) for s in (*seed, *chosen)), default=0.0)
            val = float(c["score_total"]) - lam * penalty
            if prefer is not None:
                val += prefer(c, chosen)
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


def recording_run_ok(candidate: Mapping[str, Any], chosen: Sequence[Mapping[str, Any]],
                     *, limit: int) -> bool:
    """No more than ``limit`` consecutive chosen shots from one recording.

    Only the tail of ``chosen`` counts as a run: a recording that appeared
    earlier and was already broken up by something else is not repeating.
    """
    rid = candidate.get("recording_id")
    if not rid:
        return True
    run = 0
    for s in reversed(chosen):
        if s.get("recording_id") == rid:
            run += 1
        else:
            break
    return run < limit


def source_alternation_bonus(candidate: Mapping[str, Any], chosen: Sequence[Mapping[str, Any]],
                             *, after: int, bonus: float = 0.05) -> float:
    """A soft nudge toward alternating source once one has run for a while.

    Soft, not a veto: a single source holding the only good shot of a moment
    must still be selectable, just without the bonus that would tip a
    near-tie toward it.
    """
    if len(chosen) < after:
        return 0.0
    tail = chosen[-after:]
    sources = {s.get("source") for s in tail}
    if len(sources) == 1 and candidate.get("source") not in sources:
        return bonus
    return 0.0


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


def source_share_repair(chosen: list, pool: Sequence, *, min_share: float,
                        phones: Sequence[str] = ("phone_keller", "phone_kulikov")) -> list:
    """Give a starved phone its share of an act after the fill, not during it.

    MMR fills by score and diversity; it has no notion of "belongs to
    Keller" or "belongs to Kulikov" and shouldn't gain one, because a phone
    that shot fewer good moments must not be padded with weak ones to look
    even. This runs once, after selection: for each phone below
    ``min_share`` of the act's phone slots, swap the weakest slot of the
    phone currently ahead for the best material the starved phone still has
    in the pool, until the share holds or the pool runs dry. A phone is
    never swapped down to zero slots -- that would erase it rather than
    balance it. Chronology is the caller's job (``chronological`` re-sorts
    what this returns); this only decides membership.
    """
    chosen = list(chosen)
    pool_by_phone = {p: sorted((s for s in pool if s.get("source") == p),
                               key=lambda s: -float(s.get("score_total") or 0.0))
                     for p in phones}

    for phone in phones:
        others = [p for p in phones if p != phone]
        while True:
            slots = [s for s in chosen if s.get("source") in phones]
            if not slots:
                break
            share = sum(1 for s in slots if s.get("source") == phone) / len(slots)
            if share >= min_share:
                break
            available = pool_by_phone.get(phone) or []
            if not available:
                break                   # the pool is dry; the share cannot be repaired
            # Raw count, not share: with two phones the one ahead by count is
            # the one ahead by share, and the trek had two. A third phone
            # would have to compare shares here.
            over_source = max(others, default=None,
                              key=lambda p: sum(1 for s in slots if s.get("source") == p))
            over_slots = [s for s in chosen if s.get("source") == over_source]
            if len(over_slots) <= 1:
                break                   # never drop the other phone to zero
            weakest = min(over_slots, key=lambda s: float(s.get("score_total") or 0.0))
            chosen.remove(weakest)
            chosen.append(available.pop(0))
    return chosen


# `speech_first` lived here until Film v2 step 3: speech-carrying shots ahead
# of the rest, so the fill reached for silence only once the voices were in.
# It placed two whisper hallucinations first in their acts. The voice of the
# film is now chosen by reading (the beat sheet, S04.5), not by a flag.


def shot_available_s(shot: Mapping[str, Any]) -> float:
    """How many seconds of source the shot actually has.

    A photograph has no limit: a still is held for as long as the slot asks.
    A video shot has exactly ``end_s - start_s``, and asking for more does not
    produce more -- ffmpeg simply stops, which is how 126 video slots wanted
    664.9 s of footage and the render delivered 593.3 s. The timeline must not
    claim material that does not exist, because every downstream number (act
    length, total runtime, the music layout) is computed from the claim.
    """
    if str(shot.get("media_kind") or "video") == "photo":
        return math.inf
    end = shot.get("end_s")
    if end is None:
        return math.inf
    return max(0.0, float(end) - float(shot.get("start_s") or 0.0))


def snap_within(t: float, ceiling: float, beats: Sequence[float]) -> float:
    """The latest beat strictly after ``t`` and no later than ``ceiling``.

    Used when the nearest beat would overrun the footage. Falling back to the
    raw ceiling would drop the cut off the grid entirely, so a beat that fits
    is preferred and the ceiling is only used when the shot is shorter than
    the gap between beats.
    """
    fits = [b for b in beats if t < b <= ceiling]
    return max(fits) if fits else float(ceiling)


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
        # The act's range says how long this shot deserves; the shot says how
        # long it can be. The shorter of the two wins, always.
        avail = shot_available_s(s)
        want = min(want, avail)
        end = snap_to_beat(t + want, beats,
                           downbeats=downbeats if i == 0 else None)
        if end <= t:                    # the grid was too coarse to advance
            end = t + want
        if end - t > avail:             # the nearest beat overran the footage
            end = snap_within(t, t + avail, beats)
        src_in = float(s.get("start_s") or 0.0)
        out.append({
            "act": s.get("act"),
            "t_in": round(t, 3), "t_out": round(end, 3),
            "kind": "photo" if s.get("media_kind") == "photo" else "video",
            "shot_id": s["shot_id"],
            "src_in": round(src_in, 3),
            "src_out": round(src_in + (end - t), 3),
            "secondary_shot_id": None, "secondary_src_in": None,
            "motion": None, "speed": 1.0, "transition": "cut",
            "beat_id": None, "scene_id": None, "msg_id": None,
            "yaw": s.get("chosen_yaw"), "locked": 0,
        })
        t = end
    return out
