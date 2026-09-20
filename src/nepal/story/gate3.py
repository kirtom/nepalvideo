"""Gate 3 -- the page read next to the draft (Film v2 step 4, task 14).

The draft is watched; this is what the operator reads while watching it.
A Gate 3 note says "the shot at 06:45 on the 4th is wrong", and someone
then has to find the row -- so every slot is listed with the moment it
shows on the trek's own clock, beside its index and its film time. Under
it: the per-act source ratio the operator asked to see, the three audio
tracks cue by cue, the cards, and the run's numbers.

Pure: rows in, one string out. Nothing here reads a file or the database,
so the page renders in a test from a handful of hand-built rows and the
two promises it makes are checked on the string.

Nobody is named. The chat authors are the letters the beat sheet gave
them, and a phone is "phone of B", as ``Cast.source`` renders it -- this
page is what gets forwarded for a second opinion.
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from nepal.process.render import timecode
from nepal.process.timeline_io import AUDIO_TRACKS
from nepal.story import anchors as anchors_mod
from nepal.story.beats_input import Cast

# Nepal's clock, the offset ``effort.nepal_hour`` defaults to. A fact of
# the country rather than a tunable: the operator remembers the pass at
# dawn, not at 01:00 UTC.
NEPAL_UTC_OFFSET_H = 5.75

_STYLE = ("body{font:14px/1.4 system-ui,sans-serif;background:#111;color:#ddd;margin:2rem;max-width:80rem}"
          "h1,h2{font-weight:600}table{border-collapse:collapse;margin:.5rem 0 1.5rem}"
          "th,td{text-align:left;padding:.2rem .8rem .2rem 0;vertical-align:top;white-space:nowrap}"
          "th{color:#999;font-weight:500}.big{font-size:1.2rem}")


def wall_clock(slot: Mapping[str, Any]) -> str:
    """The trek's local time at the slot's first frame: the shot's stamp
    moved forward by how far into the shot the slot cuts. A card, or a shot
    with no stamp, has no moment to name."""
    ts = anchors_mod._epoch(slot.get("start_utc"))
    if ts is None:
        return "-"
    into = float(slot.get("src_in") or 0.0) - float(slot.get("start_s") or 0.0)
    local = datetime.fromtimestamp(ts + into, tz=timezone.utc) + timedelta(hours=NEPAL_UTC_OFFSET_H)
    return local.strftime("%Y-%m-%d %H:%M:%S")


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    e = html.escape
    head = "".join(f"<th>{e(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{e('-' if v is None else str(v))}</td>" for v in r) + "</tr>"
                   for r in rows)
    return f"<table><tr>{head}</tr>{body}</table>\n"


def _num(x: Any, digits: int = 1) -> str:
    return "-" if x is None else f"{float(x):.{digits}f}"


def _card(payload: Any) -> tuple[str, str]:
    """(tag, text) from an overlay's payload -- the JSON ``overlay_rows``
    wrote, which carries the author's tag and never the name."""
    try:
        p = json.loads(payload or "{}")
    except (TypeError, ValueError):
        return "-", str(payload or "")
    return str(p.get("author_tag") or "-"), str(p.get("text") or "")


def render_gate3(slots: Sequence[Mapping[str, Any]], cues: Sequence[Mapping[str, Any]],
                 overlays: Sequence[Mapping[str, Any]], *,
                 per_act_sources: Mapping[str, Mapping[str, Mapping[str, Any]]],
                 report: Mapping[str, Any], cast_tags: Mapping[str, str]) -> str:
    """The page. ``slots`` are the timeline rows with the shot's
    ``start_utc``/``start_s`` and its ``source_name`` joined on; ``cues``
    and ``overlays`` the tables as written; ``per_act_sources`` what
    ``db.per_act_sources`` counts; ``report`` the run's report so far;
    ``cast_tags`` the beat sheet's author -> letter map."""
    e = html.escape
    cast = Cast(tags=dict(cast_tags))
    end = max((float(s["t_out"]) for s in slots), default=0.0)
    n_cues = {t: sum(1 for c in cues if c.get("track") == t) for t in AUDIO_TRACKS}

    shot_rows = []
    for s in slots:
        kind = str(s.get("kind") or "")
        if s.get("secondary_shot_id"):
            kind += " (split)"
        shot_rows.append((s.get("slot_index"), timecode(s["t_in"]), wall_clock(s), s.get("act"), kind,
                          cast.source(s.get("source_name")) if s.get("source_name") else "-",
                          s.get("shot_id"), s.get("beat_id"), s.get("scene_id"),
                          _num(float(s["t_out"]) - float(s["t_in"]))))

    source_rows = []
    for act, sources in sorted(per_act_sources.items(), key=lambda kv: int(kv[0])):
        total = sum(int(v["slots"]) for v in sources.values())
        for src, v in sorted(sources.items()):
            share = f"{int(v['slots']) / total * 100:.0f}%" if total else "-"
            source_rows.append((act, cast.source(src), v["slots"], v["available"], share,
                                v.get("video_slots"), v.get("videos")))

    cue_tables = ""
    for track in AUDIO_TRACKS:
        mine = sorted((c for c in cues if c.get("track") == track), key=lambda c: float(c["t_in"]))
        cue_tables += f"<h2>cues: {track} ({len(mine)})</h2>" + _table(
            ("cue", "in", "out", "source", "src in", "src out", "LUFS", "beat"),
            [(c["cue_id"], timecode(c["t_in"]), timecode(c["t_out"]), c.get("source"),
              _num(c.get("src_in"), 2), _num(c.get("src_out"), 2), _num(c.get("gain_lufs")), c.get("beat_id"))
             for c in mine])

    card_rows = []
    for o in sorted(overlays, key=lambda o: float(o["t_in"])):
        tag, text = _card(o.get("payload"))
        card_rows.append((o.get("overlay_id"), o.get("kind"), timecode(o["t_in"]), timecode(o["t_out"]), tag, text))

    # The report's scalars only: its lists and maps (windows, spans, the
    # per-act tables) are either on this page already or not a number, and
    # a string with a path in it (the draft's, an ffmpeg error's) names the
    # machine and its user, which a forwarded page must not.
    report_rows = [(f"{section}.{k}", v) for section in ("score", "timeline", "cues", "draft")
                   for k, v in (report.get(section) or {}).items()
                   if isinstance(v, (bool, int, float, str)) and "/" not in str(v)]

    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8"><title>gate 3: the draft</title>\n'
        f"<style>{_STYLE}</style></head>\n"
        "<body><h1>Gate 3: the draft</h1>\n"
        f'<p class="big"><b>{len(slots)}</b> slot(s), <b>{timecode(end)}</b>; '
        f"<b>{sum(n_cues.values())}</b> cue(s): {', '.join(f'{t} {n}' for t, n in n_cues.items())}; "
        f"<b>{len(overlays)}</b> card(s)</p>\n"
        "<h2>sources per act</h2>"
        + _table(("act", "source", "slots", "available", "share", "video slots", "videos"), source_rows)
        + "<h2>shots</h2>"
        + _table(("#", "film", "wall clock", "act", "kind", "source", "shot", "beat", "scene", "s"), shot_rows)
        + cue_tables
        + "<h2>cards</h2>" + _table(("overlay", "kind", "in", "out", "tag", "text"), card_rows)
        + "<h2>the run</h2>" + _table(("number", "value"), report_rows)
        + f"<p>{e(str(report.get('finished_utc') or report.get('started_utc') or ''))}</p>\n"
        "</body></html>\n")
