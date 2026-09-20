"""Film v2 step 4, Part B -- what is heard under the picture, and which of
the chat's lines are laid over it (spec section 6).

The timeline says which picture plays when. This module turns that, the
placed speech anchors and the music map into the rows the mixer and step 6
read: three audio tracks -- ``speech`` (one cue per placed anchor, the voice
at its own level), ``location`` (one cue per slot: the picture's own sound,
or the last video's ambience held under a still), ``music`` (one cue per
segment of the map, never inside the silence) -- and the ``chat_card``
overlays. Nothing here decides; it lays out what the earlier steps chose.

Pure on purpose: rows in, rows out. The level rules are the part that goes
wrong at the ffmpeg boundary, so they are testable with no media, no
database and no config.
"""
from __future__ import annotations

import dataclasses
import json
import math
from typing import Any, Mapping, Sequence

from nepal.story.anchors import Anchor

# Every row carries every column: ``db.upsert`` takes the column list from
# the first row, so a cue missing a key would silently drop that column for
# the whole batch.
CUE_KEYS = ("cue_id", "track", "t_in", "t_out", "source", "src_in", "src_out",
            "gain_lufs", "fade_in_s", "fade_out_s", "beat_id")
OVERLAY_KEYS = ("overlay_id", "kind", "t_in", "t_out", "payload", "asset_path")

# The map and the timeline round to milliseconds; two edges that the writer
# meant to coincide can still differ by a float ulp.
_EDGE_TOL_S = 1e-3


def _cue(**fields: Any) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(CUE_KEYS)
    row.update(fields)
    return row


def _inside(t: float, spans: Sequence[tuple[float, float]]) -> bool:
    return any(a <= t <= b for a, b in spans)


def _touches(t0: float, t1: float, spans: Sequence[tuple[float, float]]) -> bool:
    """Overlapping or abutting: the cue on either side of an edge shares it."""
    return any(t0 <= b and t1 >= a for a, b in spans)


# -- speech ---------------------------------------------------------------

def _runs(slots: Sequence[Mapping[str, Any]]) -> list[tuple[str, Mapping[str, Any], float]]:
    """Each run of consecutive slots carrying the same ``beat_id``: the beat,
    the run's first slot and where the run ends. Runs, not one span per beat
    -- the cold open plays a late beat at t = 0 and its act plays it again,
    and one span for both would put the whole film in between "under
    speech"."""
    out: list[tuple[str, Mapping[str, Any], float]] = []
    prev = None
    for s in slots:
        beat = s.get("beat_id")
        if beat and beat == prev:
            out[-1] = (beat, out[-1][1], max(out[-1][2], float(s["t_out"])))
        elif beat:
            out.append((beat, s, float(s["t_out"])))
        prev = beat
    return out


def place_speech(anchors: Sequence[Anchor], slots: Sequence[Mapping[str, Any]]) -> list[Anchor]:
    """One placed anchor per run of slots carrying its beat, in slot order:
    the cold open is the film's device -- the line teases at t = 0 and is
    delivered in its act -- so both runs are heard, the same utterance each
    time. The voice starts where the run's first slot reaches the anchor's
    source: a locked slot opens on it, the cold open runs into it after its
    extension. A slot that opens after the anchor's pre-roll (a cold open
    longer than its range extends less than the pre-roll) starts the voice
    with the picture rather than before the film. The offset is read only
    when the slot is on the anchor's recording -- another clip's ``src_in``
    is on another clock -- so a run the retime left all B-roll starts the
    voice with its first cut. A beat with no run is absent from the result;
    the caller says so."""
    firsts: dict[str, list[Mapping[str, Any]]] = {}
    for beat, first, _ in _runs(slots):
        firsts.setdefault(beat, []).append(first)
    out: list[Anchor] = []
    for a in anchors:
        for slot in firsts.get(a.beat_id, []):
            same_clip = slot.get("recording_id") == a.recording_id
            lead = a.src_in - float(slot["src_in"] or 0.0) if same_clip else 0.0
            if lead < 0:
                a = dataclasses.replace(a, src_in=a.src_in - lead, duration_s=a.duration_s + lead)
                lead = 0.0
            out.append(dataclasses.replace(a, t_in=float(slot["t_in"]) + lead))
    return out


def speech_cues(anchors: Sequence[Anchor], *, lufs: float, fade_s: float) -> list[dict[str, Any]]:
    """One cue per placed speech anchor (``t_in`` already on film time). A
    beat placed more than once -- the cold open and its act -- keeps
    ``sp_<beat>`` for its first cue and numbers the rest from 2."""
    seen: dict[str, int] = {}
    out: list[dict[str, Any]] = []
    for a in anchors:
        if a.kind != "speech":
            continue
        n = seen[a.beat_id] = seen.get(a.beat_id, 0) + 1
        out.append(_cue(cue_id=f"sp_{a.beat_id}" if n == 1 else f"sp_{a.beat_id}_{n}", track="speech",
                        t_in=round(a.t_in, 3), t_out=round(a.t_in + a.duration_s, 3),
                        source=a.recording_id, src_in=a.src_in, src_out=a.src_out, gain_lufs=lufs,
                        fade_in_s=fade_s, fade_out_s=fade_s, beat_id=a.beat_id))
    return out


def speech_spans(slots: Sequence[Mapping[str, Any]]) -> list[tuple[float, float]]:
    """Where a voice is heard: each run's span."""
    return [(float(first["t_in"]), t1) for _, first, t1 in _runs(slots)]


# -- location -------------------------------------------------------------

def location_cues(slots: Sequence[Mapping[str, Any]], *, lufs_under_music: float, lufs_full: float,
                  lufs_under_speech: float, speech_spans: Sequence[tuple[float, float]],
                  windows: Sequence[Mapping[str, Any]], silence: Mapping[str, Any] | None,
                  fade_s: float, window_fade_s: float) -> list[dict[str, Any]]:
    """One cue per slot, in ``slot_index`` order, each slot carrying its
    shot's ``recording_id``. A video slot plays its own recording at its
    own span (a split slot: its primary); a photo or card holds the act's
    last video recording on from where that stopped, so the ambience does
    not drop out under a still. ``windows`` are the report's natural-sound
    windows (``t_in``/``t_out``), ``silence`` the map's window
    (``t_start``/``t_end``); inside either the location sound is the mix.

    A slot's level is read at its centre, so a window that cuts a slot
    claims it by the larger part rather than by a frame at the edge; every
    cue that overlaps or abuts a window takes the window fade at both ends,
    so the rise into full sound is one fade on each side of the cut.
    """
    full = [(float(w["t_in"]), float(w["t_out"])) for w in windows]
    if silence:
        full.append((float(silence["t_start"]), float(silence["t_end"])))
    out: list[dict[str, Any]] = []
    held: tuple[str, float] | None = None       # (recording, where its ambience got to)
    act = None
    for s in slots:
        if s.get("act") != act:
            act, held = s.get("act"), None
        t0, t1 = float(s["t_in"]), float(s["t_out"])
        if s.get("kind") == "video" and s.get("recording_id"):
            src_in, src_out = float(s["src_in"]), float(s["src_out"])
            held = (s["recording_id"], src_out)
        elif s.get("kind") in ("photo", "card") and held:
            src_in = held[1]
            src_out = round(src_in + (t1 - t0), 3)
            held = (held[0], src_out)
        else:
            continue
        mid = (t0 + t1) / 2
        if _inside(mid, full):
            gain = lufs_full
        elif _inside(mid, speech_spans):
            gain = lufs_under_speech
        else:
            gain = lufs_under_music
        fade = window_fade_s if _touches(t0, t1, full) else fade_s
        out.append(_cue(cue_id=f"lo_{s['slot_index']}", track="location", t_in=t0, t_out=t1,
                        source=held[0], src_in=src_in, src_out=src_out, gain_lufs=gain,
                        fade_in_s=fade, fade_out_s=fade))
    return out


# -- music ----------------------------------------------------------------

def map_on_film_time(mmap: Mapping[str, Any],
                     act_spans: Mapping[int, tuple[float, float]]) -> dict[str, Any]:
    """The map with every act on the span the timeline actually gave it.

    The map is built on the planned act spans and rebuilt only when an act's
    length moved by more than a scene; the rhythm pass then times each act's
    cuts against the map's grid shifted to where that act really starts. So
    the table is on film time and the map is not -- on the seeded corpus act
    4 opened 8.4 s after the map said -- and a cue laid at the map's
    ``t_start`` would drift against the picture by that much. ``act_spans``
    are the table's own first ``t_in`` and last ``t_out`` per act. The
    silence keeps its length and stays after the act it followed in the map,
    at that act's real end.
    """
    out = dict(mmap)
    out["acts"] = [dict(e, t_start=round(act_spans[int(e["act"])][0], 3),
                        t_end=round(act_spans[int(e["act"])][1], 3))
                   if int(e["act"]) in act_spans else dict(e) for e in mmap.get("acts", [])]
    q = mmap.get("silence_window") or {}
    if q:
        q0, q1 = float(q["t_start"]), float(q["t_end"])
        before = next((int(e["act"]) for e in mmap.get("acts", [])
                       if math.isclose(float(e["t_end"]), q0, abs_tol=_EDGE_TOL_S)), None)
        if before in act_spans:
            end = act_spans[before][1]
            out["silence_window"] = {"t_start": round(end, 3), "t_end": round(end + (q1 - q0), 3)}
    return out


def music_cues(mmap: Mapping[str, Any], *, lufs: float, xfade_s: float,
               window_fade_s: float) -> list[dict[str, Any]]:
    """One cue per segment of every act in the map, on film time. Adjacent
    cues carry the crossfade on both ends; the renderer overlaps them. The
    silence window gets no music: a segment inside it is dropped, one that
    runs into it stops at its edge and fades over the window fade, one that
    starts inside it resumes where the silence ends, the track advanced by
    the same amount so the map's beat grid still lines up.

    An act's last cue ends where the act ends. The map's segments follow the
    slots its scenes were grouped on, and those stop short of the act's end
    or run past it (on the seeded corpus 1.3 s short in act 1, 7.6 s over in
    act 3): short would be a hole in the bed right before the cut, over
    would be two beds under the next act's first shot. A segment that only
    begins past the end is dropped."""
    q = mmap.get("silence_window") or {}
    q0, q1 = (float(q["t_start"]), float(q["t_end"])) if q else (math.inf, math.inf)
    out: list[dict[str, Any]] = []
    for entry in mmap.get("acts", []):
        t_start, t_end = float(entry["t_start"]), float(entry["t_end"])
        segments = entry.get("segments") or []
        kept = [i for i, seg in enumerate(segments) if t_start + float(seg["t_in"]) < t_end - _EDGE_TOL_S]
        for i in kept:
            seg = segments[i]
            t0, t1 = t_start + float(seg["t_in"]), t_start + float(seg["t_end"])
            src_in, src_out = float(seg["src_in"]), float(seg["src_out"])
            if i == kept[-1]:
                src_out, t1 = src_out + (t_end - t1), t_end
            if t0 < q1 and t1 > q0:
                if t0 >= q0 and t1 <= q1:
                    continue
                if t0 < q0:
                    src_out, t1 = src_out - (t1 - q0), q0
                else:
                    src_in, t0 = src_in + (q1 - t0), q1
            fade_out = window_fade_s if math.isclose(t1, q0, abs_tol=_EDGE_TOL_S) else xfade_s
            out.append(_cue(cue_id=f"mu_{entry['act']}_{i}", track="music", t_in=round(t0, 3),
                            t_out=round(t1, 3), source=seg["track_id"], src_in=round(src_in, 3),
                            src_out=round(src_out, 3), gain_lufs=lufs, fade_in_s=xfade_s,
                            fade_out_s=fade_out))
    return out


# -- overlays -------------------------------------------------------------

def _overlay(beat: Mapping[str, Any], t_in: float, t_out: float, *, side: str,
             cast_tags: Mapping[str, str]) -> dict[str, Any]:
    # The tag, never the name: the author string stays in the database.
    tag = cast_tags.get(beat.get("author") or "") if beat.get("msg_id") else None
    payload = {"text": beat.get("text") or "", "author_tag": tag, "side": side,
               "msg_id": beat.get("msg_id")}
    return {"overlay_id": f"cc_{beat['beat_id']}", "kind": "chat_card", "t_in": round(t_in, 3),
            "t_out": round(t_out, 3), "payload": json.dumps(payload, ensure_ascii=False),
            "asset_path": None}


def overlay_rows(beats: Sequence[Mapping[str, Any]], slots: Sequence[Mapping[str, Any]], *,
                 chat_card_s: float, closing_card_s: float,
                 cast_tags: Mapping[str, str]) -> list[dict[str, Any]]:
    """The ``chat_card`` overlays: the quote beats spread evenly over Act 1's
    film span in rank order, sides alternating from the left, and the
    closing line over the last seconds before the end. ``beats`` are the
    ``story_beats`` rows with the message's ``author`` joined on; the tag
    written is ``cast_tags[author]``, the same lettering the beat sheet's
    prompt used. Stat cards are step 6's, from the sheet's ideas, not here.
    """
    act1 = [s for s in slots if s.get("act") == 1]
    quotes = sorted((b for b in beats if b.get("kind") == "quote"),
                    key=lambda b: (b.get("rank") if b.get("rank") is not None else math.inf, b["beat_id"]))
    out: list[dict[str, Any]] = []
    if act1 and quotes:
        a0 = min(float(s["t_in"]) for s in act1)
        a1 = max(float(s["t_out"]) for s in act1)
        step = (a1 - a0) / len(quotes)
        for k, b in enumerate(quotes):
            t_in = a0 + (k + 0.5) * step - chat_card_s / 2
            t_in = max(a0, min(t_in, a1 - chat_card_s))       # a card never leaves its act
            out.append(_overlay(b, t_in, t_in + chat_card_s, side=("left", "right")[k % 2],
                                cast_tags=cast_tags))
    end = max((float(s["t_out"]) for s in slots), default=0.0)
    for b in beats:
        if b.get("kind") == "closing":
            out.append(_overlay(b, max(0.0, end - closing_card_s), end, side="left", cast_tags=cast_tags))
    return out
