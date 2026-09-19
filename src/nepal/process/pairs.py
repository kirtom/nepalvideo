"""Film v2 step 4 -- pairing two phones (or a phone and the camera) into one
split-screen slot (spec section 3.2).

A trek has three cameras rolling independently; when two of them catch the
same instant it is worth more on screen split than either half is alone. This
module only decides *which* shots pair and *where their spans align* -- pure
dicts and arithmetic, no ffmpeg, no database -- so the ranking that decides
what the film looks like is unit-testable with no media present, the same
discipline the FOV and clock solvers follow. S05 (Task 9) turns the slot dict
into an actual split-screen render.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Sequence

from nepal.process.assemble import SLOT_KEYS

# The brief's own number: two shots of the same moment from two angles, one
# of them a genuine reaction rather than the same face twice, is worth more
# than the raw score sum says -- but it is a nudge, not a second signal, so
# it must never outweigh a real score gap.
FACE_CLUSTER_BONUS = 0.2


def is_portrait(row: Mapping[str, Any]) -> bool:
    """A phone held up for a moment, not the horizontal camera -- missing
    dimensions read as landscape rather than raising, since a shot with no
    measured frame is a fact this function did not cause."""
    width, height = row.get("width"), row.get("height")
    if width is None or height is None:
        return False
    return height > width


def _epoch(iso: str | None) -> float | None:
    """Parse an ISO 8601 capture stamp, exactly as anchors.py does: a
    malformed or missing stamp means this shot cannot pair, not an error."""
    if not iso:
        return None
    try:
        return datetime.fromisoformat(str(iso)).timestamp()
    except ValueError:
        return None


def _score(row: Mapping[str, Any]) -> float:
    return float(row.get("score_total") or 0.0)


def _is_phone(source: Any) -> bool:
    return source in ("phone_keller", "phone_kulikov")


def _can_pair_sources(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """Two different sources, at least one a phone; telegram (no clock of
    its own trustworthy enough to align a split) never pairs."""
    sa, sb = a.get("source"), b.get("source")
    if sa == sb or sa == "telegram" or sb == "telegram":
        return False
    return _is_phone(sa) or _is_phone(sb)


def find_pairs(rows: Sequence[Mapping[str, Any]], *, window_s: float,
              per_act: int) -> list[dict[str, Any]]:
    """Greedily pick the highest-ranked, non-overlapping-in-shots pairs.

    Overlap is on UTC wall-clock span (each shot's camera runs on its own
    drifted clock otherwise), not act membership; act is only where the cap
    and the resulting slot's ``act`` come from.
    """
    spans = []
    for r in rows:
        ts = _epoch(r.get("start_utc"))
        if ts is None:
            continue
        duration = float(r["end_s"]) - float(r["start_s"])
        spans.append((r, ts, ts + duration))

    candidates = []
    for i in range(len(spans)):
        a, a0, a1 = spans[i]
        for j in range(i + 1, len(spans)):
            b, b0, b1 = spans[j]
            if not _can_pair_sources(a, b):
                continue
            if not (is_portrait(a) or is_portrait(b)):
                continue
            # window_s bounds how far apart the two starts may be -- what
            # "the same moment" means for two people pulling out phones --
            # but a split screen also needs both clips to actually hold
            # footage at the aligned instant, so the spans must truly
            # overlap (common duration > 0), not just fall within window_s
            # of each other with a gap between them.
            if abs(a0 - b0) > float(window_s):
                continue
            common = min(a1, b1) - max(a0, b0)
            if common <= 0:
                continue
            rank = _score(a) + _score(b)
            fc_a, fc_b = a.get("face_cluster"), b.get("face_cluster")
            if fc_a is not None and fc_b is not None and fc_a != fc_b:
                rank += FACE_CLUSTER_BONUS
            candidates.append((rank, a, a0, b, b0, common))

    candidates.sort(key=lambda c: c[0], reverse=True)

    used_shots: set[Any] = set()
    per_act_count: dict[Any, int] = {}
    pairs: list[dict[str, Any]] = []
    for rank, a, a0, b, b0, common in candidates:
        if a["shot_id"] in used_shots or b["shot_id"] in used_shots:
            continue
        primary, secondary = (a, b) if _score(a) >= _score(b) else (b, a)
        act = primary.get("act")               # paired shots are simultaneous, so they share an act; the primary's is taken
        if per_act_count.get(act, 0) >= per_act:
            continue
        later0 = max(a0, b0)
        primary_src_in = float(primary["start_s"]) + (later0 - (a0 if primary is a else b0))
        secondary_src_in = float(secondary["start_s"]) + (later0 - (a0 if secondary is a else b0))
        # common is the true overlap duration, computed once at candidate
        # time -- by construction it never exceeds either shot's remaining
        # footage, so src_out can't run past what either clip actually has.
        src_out = primary_src_in + common

        slot = {k: None for k in SLOT_KEYS if k != "slot_index"}
        slot.update(kind="video", act=act, shot_id=primary["shot_id"],
                   secondary_shot_id=secondary["shot_id"],
                   src_in=primary_src_in, secondary_src_in=secondary_src_in,
                   src_out=src_out, motion='{"type":"split"}', speed=1.0,
                   transition="cut",  # a split needs no transition of its own; the cut is instant
                   locked=0)
        pairs.append(slot)
        used_shots.add(a["shot_id"])
        used_shots.add(b["shot_id"])
        per_act_count[act] = per_act_count.get(act, 0) + 1

    return pairs
