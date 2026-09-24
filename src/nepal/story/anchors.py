"""Film v2 step 4 -- anchors from the beat sheet (spec section 5.1, and 5.5
for the cold open) of docs/superpowers/specs/2026-09-16-film-v2-voice-spine-design.md.

`story_beats` says what is said and when it was said; the timeline needs to
know where each moment sits in the film and how much picture surrounds it.
An `Anchor` is that translation: a beat plus the span of source it needs, its
own-recording clock, and the chronological ``utc`` used to lay it out.

Kept pure on purpose (dataclasses, plain dicts, `datetime`) so the placement
math -- the part every act's shape depends on -- is unit-testable with no
database and no media, the same discipline the FOV and clock solvers follow.

``act_utc0``/``act_utc1`` are Unix timestamps (seconds), not ISO strings: an
anchor's own ``utc`` is ISO for readability everywhere else it travels, but
comparing two ISO strings for a fraction is just comparing their epoch
seconds with extra steps, so the caller (which already has the act's time
bounds as floats from the spine) is asked for exactly the number this
function needs.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Mapping, Sequence

from nepal.process.assemble import SLOT_KEYS, epoch_utc, parse_utc

# ``_epoch`` is the name seven call sites in s05_cut.py and gate3.py already
# spell (``anchors_mod._epoch``); the parse itself now lives in assemble.py,
# next to the fill that shares it. Renaming across those two files is another
# task's diff, so the old name stays bound to the shared helper here.
_epoch = epoch_utc


@dataclass(frozen=True)
class Anchor:
    beat_id: str
    act: int
    kind: str            # speech | quote | closing
    utc: str | None      # the beat's moment (shot start_utc + src_in) for chronology, ISO 8601
    recording_id: str | None
    shot_id: str | None
    src_in: float        # on the recording's clock, pre-roll already applied
    src_out: float
    duration_s: float    # src_out - src_in
    own_picture: bool    # True: the picture stays on the beat's recording (walking shot)
    face_hold_s: float   # seconds of the speaker's own shot before B-roll (0 when own_picture)
    text: str
    effect: str          # none | freeze | burst | ramp
    t_in: float = -1.0   # on the film timeline, set by place_anchors


def speech_anchors(beats: Sequence[Mapping[str, Any]],
                   shots_by_id: Mapping[str, Mapping[str, Any]], *,
                   face_hold_s: float, pre_roll_s: float,
                   own_picture_below: float) -> list[Anchor]:
    """One anchor per speech beat, pre-rolled into the shot it lives on."""
    out: list[Anchor] = []
    for b in beats:
        if b.get("kind") != "speech":
            continue
        shot = shots_by_id[b["shot_id"]]
        beat_src_in = float(b["src_in"])
        shot_start = float(shot["start_s"])
        src_in = max(shot_start, beat_src_in - float(pre_roll_s))
        src_out = float(b["src_out"])
        duration = src_out - src_in
        # A malformed or missing stamp is this anchor's fact, not the batch's:
        # parse_utc returns None and the anchor travels on without a moment.
        shot_utc = parse_utc(shot.get("start_utc"))
        utc = None if shot_utc is None else (
            shot_utc + timedelta(seconds=beat_src_in - shot_start)).isoformat()
        own_picture = float(shot.get("face_score") or 0.0) < float(own_picture_below)
        hold = 0.0 if own_picture else min(float(face_hold_s), duration)
        out.append(Anchor(beat_id=b["beat_id"], act=b.get("act"), kind="speech", utc=utc,
                          recording_id=shot.get("recording_id"), shot_id=b["shot_id"],
                          src_in=src_in, src_out=src_out, duration_s=duration,
                          own_picture=own_picture, face_hold_s=hold,
                          text=b.get("text") or "", effect=b.get("effect") or "none"))
    return out


def quote_anchors(beats: Sequence[Mapping[str, Any]]) -> list[Anchor]:
    """One anchor per act-1 quote or the closing line -- no picture, no
    source span; the card's length is a config value, not a fact this
    module knows."""
    out: list[Anchor] = []
    for b in beats:
        if b.get("kind") not in ("quote", "closing"):
            continue
        out.append(Anchor(beat_id=b["beat_id"], act=b.get("act"), kind=b["kind"],
                          utc=b.get("ts_utc"), recording_id=None, shot_id=None,
                          src_in=0.0, src_out=0.0, duration_s=0.0,
                          own_picture=False, face_hold_s=0.0,
                          text=b.get("text") or "", effect=b.get("effect") or "none"))
    return out


def place_anchors(anchors: Sequence[Anchor], *, act_t0: float, act_len_s: float,
                  act_utc0: float | None, act_utc1: float | None,
                  min_gap_s: float = 4.0) -> list[Anchor]:
    """Lay anchors on the act's slice of the film timeline.

    Chronology first (an anchor's ``utc`` fraction of the act's own time
    span), then two sweeps to make the placement playable: forward so
    nothing starts before the previous one has finished and rested for
    ``min_gap_s``, backward from the act's own end so the last anchor lands
    inside it. When the act is too short for all of them, the fix is not to
    violate the *end* of the act (the next act's anchors start counting from
    there) -- it is to let the crowd run up against the *start*, which is
    why the last step clamps at ``act_t0`` rather than the gap.
    """
    act_t1 = act_t0 + act_len_s
    fracs: list[float | None] = []
    unplaced: list[int] = []
    for i, a in enumerate(anchors):
        ts = _epoch(a.utc)
        frac = None
        if (ts is not None and act_utc0 is not None and act_utc1 is not None
                and act_utc1 != act_utc0):
            frac = max(0.0, min(1.0, (ts - act_utc0) / (act_utc1 - act_utc0)))
        else:
            unplaced.append(i)
        fracs.append(frac)
    # Anchors with no timestamp to place by keep the input's order (the only
    # ordering this pure function has access to) and are spread evenly.
    n = len(unplaced)
    for rank, i in enumerate(unplaced):
        fracs[i] = (rank + 0.5) / n if n else 0.5

    placed = [dataclasses.replace(a, t_in=act_t0 + act_len_s * frac)
             for a, frac in zip(anchors, fracs)]
    placed.sort(key=lambda a: a.t_in)

    for i in range(1, len(placed)):
        prev = placed[i - 1]
        floor = prev.t_in + prev.duration_s + min_gap_s
        if placed[i].t_in < floor:
            placed[i] = dataclasses.replace(placed[i], t_in=floor)

    if placed:
        last = placed[-1]
        overhang = (last.t_in + last.duration_s) - act_t1
        if overhang > 0:
            placed[-1] = dataclasses.replace(last, t_in=last.t_in - overhang)
        for i in range(len(placed) - 2, -1, -1):
            nxt = placed[i + 1]
            ceiling = nxt.t_in - placed[i].duration_s - min_gap_s
            if placed[i].t_in > ceiling:
                placed[i] = dataclasses.replace(placed[i], t_in=ceiling)

    placed = [dataclasses.replace(a, t_in=max(act_t0, a.t_in)) for a in placed]
    placed.sort(key=lambda a: a.t_in)
    return placed


def broll_candidates(anchor: Anchor, shots: Sequence[Mapping[str, Any]], *,
                     window_s: float) -> list[dict[str, Any]]:
    """Video from another recording, near the anchor in both time and act --
    the pool B-roll is cut from once ``face_hold_s`` runs out."""
    a_ts = _epoch(anchor.utc)
    if a_ts is None:
        return []
    out = []
    for s in shots:
        if s.get("act") != anchor.act:
            continue
        if s.get("recording_id") == anchor.recording_id:
            continue
        # B-roll under a voice has to move, so a still is not a candidate.
        # An absent ``media_kind`` reads as video because the column is
        # `NOT NULL DEFAULT 'video'` (db.py): a row that lacks the key came
        # from a query that did not select it, not from a shot of unknown
        # kind, and assemble.shot_available_s defaults it the same way.
        if str(s.get("media_kind") or "video") != "video":
            continue
        s_ts = _epoch(s.get("start_utc"))
        if s_ts is None or abs(s_ts - a_ts) > float(window_s):
            continue
        out.append(s)
    out.sort(key=lambda s: -float(s.get("score_total") or 0.0))
    return out


def cold_open_pick(beats: Sequence[Mapping[str, Any]],
                   shots_by_id: Mapping[str, Mapping[str, Any]], *,
                   length_range: Sequence[float]) -> dict[str, Any] | None:
    """The film's first shot before the title: the speech beat the sheet
    ranks highest that is late enough (act 3 or 4) to be a real cost, its
    utterance extended into a proper cold open rather than cut on the word.
    """
    candidates = [b for b in beats if b.get("kind") == "speech" and b.get("act") in (3, 4)]
    if not candidates:
        return None
    beat = min(candidates, key=lambda b: b["rank"])
    shot = shots_by_id[beat["shot_id"]]
    shot_start, shot_end = float(shot["start_s"]), float(shot["end_s"])
    avail = shot_end - shot_start
    src_in, src_out = float(beat["src_in"]), float(beat["src_out"])
    lo, hi = float(length_range[0]), float(length_range[1])
    target = min(max(src_out - src_in, lo), hi, avail)
    extend = max(0.0, (target - (src_out - src_in)) / 2.0)
    new_in, new_out = src_in - extend, src_out + extend
    if new_in < shot_start:                        # push the shortfall to the other side
        new_out = min(shot_end, new_out + (shot_start - new_in))
        new_in = shot_start
    if new_out > shot_end:
        new_in = max(shot_start, new_in - (new_out - shot_end))
        new_out = shot_end

    slot = {k: None for k in SLOT_KEYS if k != "slot_index"}
    slot.update(kind="video", shot_id=beat["shot_id"], src_in=round(new_in, 3),
               src_out=round(new_out, 3), beat_id=beat["beat_id"], act=beat.get("act"),
               speed=1.0, transition="dip_black", locked=1)
    return slot
