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

import json
from typing import Any, Mapping, Sequence

from nepal.process.assemble import SLOT_KEYS, epoch_utc

# The brief's own number: two shots of the same moment from two angles, one
# of them a genuine reaction rather than the same face twice, is worth more
# than the raw score sum says -- but it is a nudge, not a second signal, so
# it must never outweigh a real score gap.
FACE_CLUSTER_BONUS = 0.2


def display_dimensions(width: Any, height: Any, probe_json: str | None) -> tuple[Any, Any]:
    """What the frame looks like on screen, not what the stream stores.

    On the real corpus, 253 of 373 phone videos carry a 90/270 rotation on
    their video stream while ``width``x``height`` still say 1920x1080 --
    only 3 phone videos are portrait by the stored numbers alone. The phone
    records portrait behind a rotation flag that ffmpeg applies on decode
    (the same trap noted in reproject.py's ``build_flat_graph_clamped``), so
    a caller reading raw width/height sees a landscape corpus and
    ``is_portrait`` never fires. ffprobe reports the rotation two ways
    depending on version: the old ``tags.rotate`` string on the video
    stream, or a ``side_data_list`` entry with a numeric ``rotation`` (often
    negative). Either one at a quarter turn (90 or 270, sign ignored) means
    the stored width/height are swapped from what a viewer sees.
    """
    if width is None or height is None or not probe_json:
        return width, height
    try:
        data = json.loads(probe_json)
    except (TypeError, ValueError):
        return width, height
    streams = data.get("streams") if isinstance(data, dict) else None
    if not streams:
        return width, height
    video = next((s for s in streams if isinstance(s, dict) and s.get("codec_type") == "video"), None)
    if video is None:
        return width, height

    def _quarter_turn(value: Any) -> bool:
        try:
            return abs(int(float(value))) % 180 == 90
        except (TypeError, ValueError):
            return False

    rotated = _quarter_turn((video.get("tags") or {}).get("rotate"))
    if not rotated:
        for side_data in video.get("side_data_list") or []:
            if isinstance(side_data, dict) and _quarter_turn(side_data.get("rotation")):
                rotated = True
                break
    return (height, width) if rotated else (width, height)


def is_portrait(row: Mapping[str, Any]) -> bool:
    """A phone held up for a moment, not the horizontal camera -- missing
    dimensions read as landscape rather than raising, since a shot with no
    measured frame is a fact this function did not cause."""
    width, height = row.get("width"), row.get("height")
    if width is None or height is None:
        return False
    return height > width


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
        ts = epoch_utc(r.get("start_utc"))     # no clock, no alignment: this shot cannot pair
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
        # Paired shots are simultaneous, so they normally share an act -- but
        # at an act boundary the two clocks can straddle it, and this field is
        # what s05_cut routes the pair by (`p.get("act") == act`) before it
        # looks the *primary* up in that act's rows
        # (`plan_act`'s `shots_by_id[slot["shot_id"]]`). Filing the pair under
        # the secondary's act would hand plan_act a primary its act never saw.
        act = primary.get("act")
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
                   # `dip_black` is the film's punctuation and Act 0's cold
                   # open owns it -- the shot itself, from
                   # anchors.cold_open_pick or s05_cut's fallback when no beat
                   # opens the film; even the card after it cuts. A pair is
                   # an ordinary picture inside an act's run -- placed among
                   # the fill and re-timed by rhythm.retime like the rest --
                   # so it takes the same plain cut every other fill slot gets.
                   # The split itself is the event; announcing it with a
                   # transition would announce it a beat early.
                   transition="cut",
                   locked=0)
        pairs.append(slot)
        used_shots.add(a["shot_id"])
        used_shots.add(b["shot_id"])
        per_act_count[act] = per_act_count.get(act, 0) + 1

    return pairs
