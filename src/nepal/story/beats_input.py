"""What the beat-sheet call is given, assembled from the database (section 4.2).

Pure in the sense that matters: every function takes a connection or rows and
returns data or text, nothing here calls a model, and the whole prompt can be
written to a file and read before a token is bought.

Two rules the tests hold this to. Nobody is named: chat authors become A, B,
C by order of first appearance, and a face cluster that matches an author
(the cluster labels are surnames) takes the same tag, so the voice in the
chat and the face on screen are one person to the reader without either
being a name. And a hallucinated transcript is not in the input at all --
the model cannot pick what it does not see.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence

from nepal import db
from nepal.spine import acts as acts_mod
from nepal.story import brief

NEPAL_TZ = timezone(timedelta(hours=5, minutes=45))
CLAUSE_END = ".!?…;:,"
TAGS = "ABCDEFGH"
GUIDE_LABELS = ("guide", "dharma", "darma")


# -- who is who, without names ------------------------------------------

@dataclass
class Cast:
    tags: dict[str, str] = field(default_factory=dict)          # author -> tag
    cluster_tags: dict[str, str] = field(default_factory=dict)  # face label -> tag

    @classmethod
    def build(cls, authors_in_order: Iterable[str | None],
              clusters: Iterable[str | None]) -> "Cast":
        tags: dict[str, str] = {}
        for a in authors_in_order:
            if a and a not in tags:
                n = len(tags)
                tags[a] = TAGS[n] if n < len(TAGS) else f"P{n}"
        cluster_tags: dict[str, str] = {}
        for c in clusters:
            if not c:
                continue
            key = c.lower()
            if key in GUIDE_LABELS:
                cluster_tags[c] = "guide"
                continue
            # the labels are surnames; the author string holds the surname
            match = next((t for a, t in tags.items() if key in a.lower()), None)
            cluster_tags[c] = match or "other"
        return cls(tags, cluster_tags)

    def author(self, name: str | None) -> str:
        return self.tags.get(name or "", "?")

    def cluster(self, label: str | None) -> str | None:
        return self.cluster_tags.get(label or "")


# -- cut points -----------------------------------------------------------

def cut_points(segments: Sequence[Mapping[str, Any]]) -> list[float]:
    """Where a beat may start or end: segment edges, and where words exist,
    the end of every clause-ending word and the start of the word after it.

    Every word edge would be a valid cut too, but listing 40 numbers per shot
    buries the text; clause ends are where an editor cuts anyway.
    """
    pts: set[float] = set()
    for seg in segments:
        pts.add(round(float(seg["start_s"]), 2))
        pts.add(round(float(seg["end_s"]), 2))
        words = seg.get("words") or []
        for i, w in enumerate(words[:-1]):
            if (w.get("word") or "").rstrip().endswith(tuple(CLAUSE_END)):
                pts.add(round(float(w["end_s"]), 2))
                pts.add(round(float(words[i + 1]["start_s"]), 2))
    return sorted(pts)


def _marked_text(seg: Mapping[str, Any]) -> str:
    """The segment's text with ⟨t⟩ after each clause end, when words exist."""
    words = seg.get("words") or []
    if not words:
        return str(seg.get("text") or "")
    out: list[str] = []
    for i, w in enumerate(words):
        token = (w.get("word") or "").strip()
        if not token:
            continue
        out.append(token)
        if i < len(words) - 1 and token.endswith(tuple(CLAUSE_END)):
            end = round(float(w["end_s"]), 2)
            nxt = round(float(words[i + 1]["start_s"]), 2)
            out.append(f"⟨{end:g}⟩" if abs(nxt - end) <= 0.05 else f"⟨{end:g}|{nxt:g}⟩")
    return " ".join(out)


# -- the rows -------------------------------------------------------------

def transcript_rows(conn) -> list[dict[str, Any]]:
    """Every non-hallucinated transcript, with what the reader needs beside it."""
    rows = []
    for r in conn.execute(
            "SELECT s.shot_id, s.recording_id, s.start_s, s.end_s, s.start_utc, "
            "s.day_index, s.act, s.place_name, s.alt_dem_m, s.face_cluster, "
            "s.face_score, s.motion_mag, s.status, s.transcript_json, r.source "
            "FROM shots s JOIN recordings r ON r.recording_id = s.recording_id "
            "WHERE s.transcript_json IS NOT NULL AND COALESCE(s.hallucinated, 0) = 0 "
            "AND COALESCE(s.transcript, '') <> '' "
            "ORDER BY s.start_utc, s.recording_id, s.start_s"):
        d = dict(r)
        try:
            doc = json.loads(d.pop("transcript_json"))
        except ValueError:
            continue
        segs = [s for s in (doc.get("segments") or []) if not s.get("hallucinated_reasons")]
        if not segs:
            continue
        d["segments"] = segs
        d["synthetic"] = bool(doc.get("synthetic"))
        d["text"] = " ".join(str(s.get("text") or "") for s in segs)
        d["cuts"] = cut_points(segs)
        rows.append(d)
    return rows


def chat_rows(conn, *, max_chars: int) -> list[dict[str, Any]]:
    rows = []
    for r in conn.execute("SELECT msg_id, ts_utc, author, text, phase FROM messages "
                          "WHERE COALESCE(text, '') <> '' ORDER BY ts_utc, msg_id"):
        d = dict(r)
        text = str(d["text"]).replace("\n", " ").strip()
        d["clipped"] = len(text) > max_chars
        d["text"] = text[:max_chars] + ("…" if d["clipped"] else "")
        rows.append(d)
    return rows


def day_rows(conn) -> list[dict[str, Any]]:
    """One row per day with material, like `nepal report`, plus the watch."""
    from nepal.stages import s02_spine   # the day statistics live with S02
    days = s02_spine.day_stats(conn)
    bounds = [acts_mod.ActBoundary(b["act"], s02_spine._dt(b["start_utc"]),
                                   s02_spine._dt(b["end_utc"]), b.get("method", ""))
              for b in json.loads(db.get_decision(conn, "act_boundaries") or "[]")]
    out = []
    for d in days:
        midnight = datetime.fromisoformat(d.date).replace(tzinfo=NEPAL_TZ)
        lo = midnight.astimezone(timezone.utc).isoformat()
        hi = (midnight + timedelta(days=1)).astimezone(timezone.utc).isoformat()
        counts = conn.execute(
            "SELECT SUM(CASE WHEN kind IN ('video360','video_flat') THEN 1 ELSE 0 END) clips,"
            " SUM(CASE WHEN kind='photo' THEN 1 ELSE 0 END) photos "
            "FROM assets WHERE created_at_utc >= ? AND created_at_utc < ?", (lo, hi)).fetchone()
        msgs = conn.execute("SELECT COUNT(*) n FROM messages WHERE ts_utc >= ? AND ts_utc < ?",
                            (lo, hi)).fetchone()["n"]
        place = conn.execute(
            "SELECT place_name, COUNT(*) n FROM assets WHERE created_at_utc >= ? "
            "AND created_at_utc < ? AND place_name IS NOT NULL "
            "GROUP BY place_name ORDER BY n DESC LIMIT 1", (lo, hi)).fetchone()
        act = conn.execute(
            "SELECT name, distance_m, gain_m, hr_max, hr_avg, moving_s FROM activities "
            "WHERE start_utc >= ? AND start_utc < ? ORDER BY distance_m DESC LIMIT 1",
            (lo, hi)).fetchone()
        out.append({
            "day_index": d.day_index, "date": d.date,
            "act": acts_mod.act_for(d.start_utc, bounds) if bounds else None,
            "place": place["place_name"] if place else None,
            "alt_max": d.alt_max, "clips": counts["clips"] or 0,
            "photos": counts["photos"] or 0, "msgs": msgs,
            "stage": act["name"] if act else None,
            "km": round(float(act["distance_m"]) / 1000, 1) if act and act["distance_m"] else None,
            "gain_m": round(float(act["gain_m"])) if act and act["gain_m"] else None,
            "hr_max": act["hr_max"] if act else None,
            "moving_h": round(float(act["moving_s"]) / 3600, 1) if act and act["moving_s"] else None,
        })
    return out


# -- the text -------------------------------------------------------------

def _utc_local(value: str | None) -> str:
    if not value:
        return "?"
    try:
        d = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return str(value)[:16]
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.astimezone(NEPAL_TZ).strftime("%Y-%m-%d %H:%M")


def render_transcripts(rows: Sequence[Mapping[str, Any]], cast: Cast) -> str:
    lines = ["# Transcripts (times in seconds on the recording's clock; ⟨t⟩ marks a "
             "cut point; local time is Nepal's)", ""]
    for r in rows:
        who = cast.cluster(r.get("face_cluster"))
        face = "nobody" if not who else (
            f"{who} (face fills the frame)" if (r.get("face_score") or 0) >= 0.8 else who)
        alt = f"{r['alt_dem_m']:.0f} m" if r.get("alt_dem_m") is not None else "-"
        pic = "ok" if r.get("status") != "rejected" else "poor (audio still usable)"
        lines.append(
            f"## shot {r['shot_id']} | act {r.get('act') or '?'} | day {r.get('day_index') or '-'}"
            f" | {_utc_local(r.get('start_utc'))} | {r.get('place_name') or '-'} | {alt}"
            f" | {r.get('source')} | on screen: {face} | picture: {pic}")
        if r.get("synthetic"):
            lines.append("  (timing: whole shot only; no cut points inside it)")
        for i, seg in enumerate(r["segments"]):
            lines.append(f"  s{i} [{float(seg['start_s']):g}-{float(seg['end_s']):g}] "
                         f"{_marked_text(seg)}")
        lines.append("")
    return "\n".join(lines)


def render_chat(rows: Sequence[Mapping[str, Any]], cast: Cast) -> str:
    lines = ["# The chat (UTC; speakers are A, B, C by first appearance)", ""]
    for r in rows:
        ts = str(r["ts_utc"])[:16].replace("T", " ")
        lines.append(f"[{r['msg_id']} {ts} {r['phase']}] {cast.author(r['author'])}: {r['text']}")
    return "\n".join(lines)


def _cell(value: Any, fmt: str = "") -> str:
    if value is None or value == "":
        return "-"
    return format(value, fmt) if fmt else str(value)


def render_days(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = ["# The days", "",
             "| day | date | act | place | max alt m | stage (watch) | km | gain m | "
             "moving h | max HR | clips | photos | msgs |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in rows:
        cells = [d["day_index"], d["date"], _cell(d["act"]), _cell(d["place"]),
                 _cell(d["alt_max"], ".0f"), _cell(d["stage"]), _cell(d["km"]),
                 _cell(d["gain_m"]), _cell(d["moving_h"]), _cell(d["hr_max"], ".0f"),
                 d["clips"], d["photos"], d["msgs"]]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(lines)


def render_vocab(vocab: Sequence[str]) -> str:
    return ("# Vocabulary (names of places and things as the two of them write them)\n\n"
            + ", ".join(str(v) for v in vocab))


@dataclass
class Prompt:
    system: str
    user: str
    meta: dict[str, Any]

    def messages(self) -> list[dict[str, Any]]:
        return [{"role": "user", "content": self.user}]


def build_prompt(cfg, conn, *, rules: Mapping[str, Any]) -> Prompt:
    """Everything the call is given, and the facts the validator needs back."""
    trans = transcript_rows(conn)
    chat = chat_rows(conn, max_chars=int(cfg.get("beats.chat_max_chars", 500)))
    days = day_rows(conn)
    cast = Cast.build([r["author"] for r in chat],
                      [r.get("face_cluster") for r in trans])
    vocab_path = cfg.work_root / "vocab" / "telegram_vocab.json"
    vocab: list[str] = []
    if vocab_path.exists():
        try:
            vocab = [v for v in json.loads(vocab_path.read_text()) if isinstance(v, str)]
        except ValueError:
            vocab = []
    parts = [brief.BRIEF, brief.rules_text(rules), render_days(days),
             render_transcripts(trans, cast), render_chat(chat, cast)]
    if vocab:
        parts.append(render_vocab(vocab[:80]))
    parts.append(brief.OUTPUT_HINT)
    user = "\n\n".join(parts)
    shot_info = {r["shot_id"]: {"act": r.get("act"), "utc": r.get("start_utc") or "",
                                "cuts": r["cuts"], "text": r["text"]} for r in trans}
    msg_info = {r["msg_id"]: {"phase": r["phase"], "text": r["text"]} for r in chat}
    meta = {"n_shots": len(trans), "n_segments": sum(len(r["segments"]) for r in trans),
            "n_synthetic": sum(1 for r in trans if r["synthetic"]),
            "n_messages": len(chat), "n_days": len(days),
            "acts_present": sorted({int(r["act"]) for r in trans if r.get("act") is not None}),
            "chars": len(user), "cast": {"authors": len(cast.tags),
                                         "clusters": sorted(set(cast.cluster_tags.values()))},
            "shot_info": shot_info, "msg_info": msg_info}
    return Prompt(brief.SYSTEM, user, meta)
