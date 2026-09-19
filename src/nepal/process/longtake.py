"""Film v2 step 4 -- the suspension-bridge crossing as one unbroken slot
(spec section 4.3 / 13.5, task 5).

Every other slot in this pipeline is assembled from short shots; the bridge
crossing is the opposite move -- find the one recording that already *is*
the moment (someone said "bridge" while filming it) and let it run, locked,
rather than cutting it up like everything else. Pure dicts and arithmetic,
no ffmpeg, no database, the same discipline as pairs.py and anchors.py so
S05 (Task 9) can insert the result chronologically without this module
knowing anything about a timeline.

The DEM drop under the bridge (spec section 13.5) is not ranked here on
purpose: picking *which* recording crosses the bridge only needs the words
said over it, but scoring *where the crossing itself sits* against the
altitude profile needs the crossing point, and that only exists once step
5/6's peak geometry (spec section 13.3) has located it. Until then this
module has no crossing point to measure a drop against.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from nepal.process.assemble import SLOT_KEYS

# The brief's own number, not a measured one: a long take that ran past two
# minutes would swallow the film's pacing budget on its own, so ranking caps
# credit for length there even though a slot can still end up shorter.
DURATION_RANK_CAP_S = 120.0


def _has_keyword(shot: Mapping[str, Any], keywords: Sequence[str]) -> bool:
    """Case-insensitive substring match on transcript or caption, either of
    which may be None; casefold (not lower) so Cyrillic keywords match
    Cyrillic text regardless of case."""
    text = " ".join(str(shot[f]) for f in ("transcript", "caption") if shot.get(f)).casefold()
    return any(kw.casefold() in text for kw in keywords)


def bridge_candidates(shots: Sequence[Mapping[str, Any]],
                      recordings: Sequence[Mapping[str, Any]], *,
                      keywords: Sequence[str]) -> list[dict[str, Any]]:
    """Recordings that mention the bridge, ranked by how good a long take
    they'd make: capped duration first (a long silent recording is not
    automatically better), then how much of it is actually about the
    bridge, then a soft tie-break preferring fewer faces (the crossing
    itself, not someone's reaction to it)."""
    shots_by_recording: dict[Any, list[Mapping[str, Any]]] = {}
    for s in shots:
        shots_by_recording.setdefault(s["recording_id"], []).append(s)

    out: list[dict[str, Any]] = []
    for rec in recordings:
        # A recording with no measured duration can't be weighed against
        # length_range at all, so it can never become a candidate -- not
        # "duration 0", which would just make it lose the rank, since
        # long_take_slot would still need a real number to slice against.
        duration_s = rec.get("duration_s")
        if duration_s is None:
            continue
        rec_id = rec["recording_id"]
        rec_shots = shots_by_recording.get(rec_id, [])
        keyword_shots = [s for s in rec_shots if _has_keyword(s, keywords)]
        if not keyword_shots:
            continue
        first = min(keyword_shots, key=lambda s: float(s["start_s"]))
        n_faces = sum(1 for s in rec_shots if s.get("has_face"))
        face_share = n_faces / len(rec_shots)
        score = (min(float(duration_s), DURATION_RANK_CAP_S),
                len(keyword_shots), -face_share)
        out.append({"recording_id": rec_id, "first_shot_id": first["shot_id"],
                   "act": first.get("act"), "score": score})

    # sort() is stable, so recordings that tie on the whole score keep the
    # input order -- the ranking tuple's own descending sense.
    out.sort(key=lambda c: c["score"], reverse=True)
    return out


def long_take_slot(candidate: Mapping[str, Any],
                   recordings_by_id: Mapping[Any, Mapping[str, Any]],
                   shots_by_recording: Mapping[Any, Sequence[Mapping[str, Any]]], *,
                   length_range: Sequence[float]) -> dict[str, Any] | None:
    """One locked slot starting at the candidate's first keyword shot,
    ``length_range[1]`` seconds long when the recording has that much left,
    else whatever remains when that still clears ``length_range[0]``, else
    no slot at all -- a slot must never claim more footage than the
    recording actually has (ffmpeg would just stop short and every later
    number derived from the timeline would be wrong)."""
    lo, hi = float(length_range[0]), float(length_range[1])
    rec = recordings_by_id[candidate["recording_id"]]
    duration_s = rec.get("duration_s")
    if duration_s is None:                 # nothing to slice a slot from
        return None
    first_shot_id = candidate["first_shot_id"]
    first = next(s for s in shots_by_recording[candidate["recording_id"]]
                if s["shot_id"] == first_shot_id)
    src_in = float(first["start_s"])
    remaining = float(duration_s) - src_in

    if remaining >= hi:
        length = hi
    elif remaining >= lo:
        length = remaining
    else:
        return None

    slot = {k: None for k in SLOT_KEYS if k != "slot_index"}
    slot.update(kind="video", shot_id=first_shot_id, src_in=src_in,
               src_out=src_in + length, act=candidate["act"], locked=1,
               speed=1.0, transition="cut")
    return slot
