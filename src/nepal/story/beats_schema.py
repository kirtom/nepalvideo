"""The beat sheet's shape and its rules (Film v2 section 4.2).

The schema constrains the answer's shape at the API; the rules here are the
ones a schema cannot state -- counts, chronology, that a cut point exists,
that a quoted text is in the transcript. The validator returns every
complaint at once so a single retry can fix them all, and it snaps a time
that is nearly on a cut point onto it rather than complaining about a
rounding.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

_BEAT = {
    "type": "object",
    "properties": {
        "beat_id": {"type": "string"},
        "kind": {"type": "string", "enum": ["speech", "quote"]},
        "act": {"type": "integer"},
        "shot_id": {"type": ["string", "null"]},
        "msg_id": {"type": ["string", "null"]},
        "src_in": {"type": ["number", "null"]},
        "src_out": {"type": ["number", "null"]},
        "text": {"type": "string"},
        "levity": {"type": "boolean"},
        "effect": {"type": "string", "enum": ["none", "freeze", "burst", "ramp"]},
        "rank": {"type": "integer"},
        "rationale": {"type": "string"},
    },
    "required": ["beat_id", "kind", "act", "shot_id", "msg_id", "src_in", "src_out",
                 "text", "levity", "effect", "rank", "rationale"],
    "additionalProperties": False,
}

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "beats": {"type": "array", "items": _BEAT},
        "pairs": {"type": "array", "items": {
            "type": "object",
            "properties": {"planning_msg_id": {"type": "string"},
                           "trek_beat_id": {"type": "string"},
                           "why": {"type": "string"}},
            "required": ["planning_msg_id", "trek_beat_id", "why"],
            "additionalProperties": False}},
        "closing": {"type": "object",
                    "properties": {"msg_id": {"type": "string"}, "text": {"type": "string"}},
                    "required": ["msg_id", "text"], "additionalProperties": False},
        "act_notes": {"type": "array", "items": {
            "type": "object",
            "properties": {"act": {"type": "integer"}, "note": {"type": "string"}},
            "required": ["act", "note"], "additionalProperties": False}},
        "stat_card_ideas": {"type": "array", "items": {"type": "string"}},
        "trailer": {"type": "object",
                    "properties": {"hook_beat_id": {"type": "string"},
                                   "cliffhanger_beat_id": {"type": "string"},
                                   "cards": {"type": "array", "items": {"type": "string"}}},
                    "required": ["hook_beat_id", "cliffhanger_beat_id", "cards"],
                    "additionalProperties": False},
    },
    "required": ["title", "beats", "pairs", "closing", "act_notes", "stat_card_ideas",
                 "trailer"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Rules:
    min_speech: int = 10
    max_speech: int = 16
    max_speech_s: float = 25.0
    min_quotes: int = 4
    max_quotes: int = 8
    max_act4: int = 1
    max_pairs: int = 5
    cut_tolerance_s: float = 0.05
    snap_s: float = 0.5
    card_max_chars: int = 180
    title_max_chars: int = 80
    acts_from: int = 2

    @classmethod
    def from_cfg(cls, cfg) -> "Rules":
        g = cfg.get
        return cls(min_speech=int(g("beats.min_speech_beats", 10)),
                   max_speech=int(g("beats.max_speech_beats", 16)),
                   max_speech_s=float(g("beats.max_speech_s", 25)),
                   min_quotes=int(g("beats.min_act1_quotes", 4)),
                   max_quotes=int(g("beats.max_act1_quotes", 8)),
                   max_act4=int(g("beats.max_act4_beats", 1)),
                   max_pairs=int(g("beats.max_pairs", 5)),
                   cut_tolerance_s=float(g("beats.cut_tolerance_s", 0.05)),
                   snap_s=float(g("beats.snap_s", 0.5)),
                   card_max_chars=int(g("spine.card_max_chars", 180)),
                   title_max_chars=int(g("beats.title_max_chars", 80)))

    def as_prompt_dict(self) -> dict[str, Any]:
        return {"min_speech": self.min_speech, "max_speech": self.max_speech,
                "max_speech_s": self.max_speech_s, "min_quotes": self.min_quotes,
                "max_quotes": self.max_quotes, "max_act4": self.max_act4,
                "max_pairs": self.max_pairs, "card_max_chars": self.card_max_chars,
                "acts_from": self.acts_from}


_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalise(text: str) -> str:
    """Case, punctuation and spacing do not make a quote a different quote."""
    return " ".join(_PUNCT.sub(" ", (text or "").lower().replace("ё", "е")).split())


def _snap(t: float, cuts: Sequence[float], rules: Rules) -> tuple[float | None, float]:
    """The nearest cut point and its distance; None when there are no cuts."""
    if not cuts:
        return None, float("inf")
    best = min(cuts, key=lambda c: abs(c - t))
    return best, abs(best - t)


def validate(doc: Mapping[str, Any], *, shot_info: Mapping[str, Mapping[str, Any]],
             msg_info: Mapping[str, Mapping[str, Any]], rules: Rules) -> list[str]:
    """Every way the sheet breaks the rules, as sentences the model can act on.

    Mutates ``doc`` only to snap a time onto the cut point it is within
    ``rules.snap_s`` of: that is a rounding, not a disagreement.
    """
    errs: list[str] = []
    beats = doc.get("beats") or []
    ids: dict[str, dict[str, Any]] = {}
    for b in beats:
        bid = str(b.get("beat_id") or "")
        if not bid:
            errs.append("a beat has no beat_id")
        elif bid in ids:
            errs.append(f"beat_id {bid} is used twice")
        else:
            ids[bid] = b

    speech = [b for b in beats if b.get("kind") == "speech"]
    quotes = [b for b in beats if b.get("kind") == "quote"]

    # -- speech beats ----------------------------------------------------
    order: list[tuple[str, float]] = []
    per_act: dict[int, int] = {}
    levity_act: dict[int, int] = {}
    for b in speech:
        bid = b.get("beat_id")
        sid = b.get("shot_id")
        info = shot_info.get(sid or "")
        if info is None:
            errs.append(f"{bid}: shot_id {sid!r} is not in the input")
            continue
        try:
            t_in, t_out = float(b.get("src_in")), float(b.get("src_out"))
        except (TypeError, ValueError):
            errs.append(f"{bid}: src_in and src_out must be numbers")
            continue
        cuts = list(info.get("cuts") or [])
        for key, t in (("src_in", t_in), ("src_out", t_out)):
            near, dist = _snap(t, cuts, rules)
            if near is None:
                continue
            if dist > rules.snap_s:
                errs.append(f"{bid}: {key}={t:g} is not a cut point of {sid} "
                            f"(nearest {near:g}; cut points {cuts})")
            elif dist > rules.cut_tolerance_s:
                b[key] = near                    # a rounding, snapped
        t_in, t_out = float(b["src_in"]), float(b["src_out"])
        if t_out <= t_in:
            errs.append(f"{bid}: src_out must be after src_in")
        elif t_out - t_in > rules.max_speech_s:
            errs.append(f"{bid}: {t_out - t_in:.1f} s is longer than "
                        f"{rules.max_speech_s:.0f} s")
        act = info.get("act")
        if act is not None and b.get("act") != act:
            errs.append(f"{bid}: shot {sid} is in act {act}, not {b.get('act')}")
        if normalise(b.get("text") or "") not in normalise(info.get("text") or ""):
            errs.append(f"{bid}: the text is not a verbatim part of {sid}'s transcript")
        order.append((str(info.get("utc") or ""), t_in))
        a = int(b.get("act") or 0)
        per_act[a] = per_act.get(a, 0) + 1
        if b.get("levity"):
            levity_act[a] = levity_act.get(a, 0) + 1

    n = len(speech)
    if not rules.min_speech <= n <= rules.max_speech:
        errs.append(f"{n} speech beats; the film needs {rules.min_speech} to "
                    f"{rules.max_speech}")
    if order != sorted(order):
        errs.append("speech beats are not in chronological order")
    acts_present = sorted({int(i["act"]) for i in shot_info.values()
                           if i.get("act") is not None and int(i["act"]) >= rules.acts_from})
    for a in acts_present:
        if not per_act.get(a):
            errs.append(f"act {a} has no speech beat")
        elif not levity_act.get(a) and a != 4:
            # Act 4 is one beat and the peak; asking it to be funny too
            # would contradict the brief's hard cut to silence.
            errs.append(f"act {a} has no beat with levity")
    if per_act.get(4, 0) > rules.max_act4:
        errs.append(f"act 4 has {per_act[4]} speech beats; at most {rules.max_act4}")

    # -- quotes and the closing line --------------------------------------
    def _check_msg(label: str, mid: Any, text: Any, phase: str) -> None:
        info = msg_info.get(mid or "")
        if info is None:
            errs.append(f"{label}: msg_id {mid!r} is not in the input")
            return
        if info.get("phase") != phase:
            errs.append(f"{label}: message {mid} is from the {info.get('phase')} phase, "
                        f"not {phase}")
        if len(str(text or "")) > rules.card_max_chars:
            errs.append(f"{label}: {len(str(text))} characters; a card holds "
                        f"{rules.card_max_chars}")
        if normalise(text or "") not in normalise(info.get("text") or ""):
            errs.append(f"{label}: the text is not a verbatim part of message {mid}")

    for b in quotes:
        _check_msg(str(b.get("beat_id")), b.get("msg_id"), b.get("text"), "planning")
        if b.get("act") != 1:
            errs.append(f"{b.get('beat_id')}: a quote belongs to act 1")
    if not rules.min_quotes <= len(quotes) <= rules.max_quotes:
        errs.append(f"{len(quotes)} act 1 quotes; the film needs {rules.min_quotes} to "
                    f"{rules.max_quotes}")
    closing = doc.get("closing") or {}
    _check_msg("closing", closing.get("msg_id"), closing.get("text"), "after")

    title = str(doc.get("title") or "").strip()
    if not title or len(title) > rules.title_max_chars:
        errs.append(f"the title must be 1 to {rules.title_max_chars} characters")

    # -- pairs and the trailer -------------------------------------------
    speech_ids = {str(b.get("beat_id")) for b in speech}
    pairs = doc.get("pairs") or []
    if len(pairs) > rules.max_pairs:
        errs.append(f"{len(pairs)} pairs; at most {rules.max_pairs}")
    for p in pairs:
        info = msg_info.get(p.get("planning_msg_id") or "")
        if info is None or info.get("phase") != "planning":
            errs.append(f"pair: planning_msg_id {p.get('planning_msg_id')!r} is not a "
                        f"planning-phase message in the input")
        if p.get("trek_beat_id") not in speech_ids:
            errs.append(f"pair: trek_beat_id {p.get('trek_beat_id')!r} is not one of the "
                        f"speech beats")
    trailer = doc.get("trailer") or {}
    for key in ("hook_beat_id", "cliffhanger_beat_id"):
        if trailer.get(key) not in speech_ids:
            errs.append(f"trailer.{key} {trailer.get(key)!r} is not one of the speech beats")
    for i, card in enumerate(trailer.get("cards") or []):
        if len(str(card)) > rules.card_max_chars:
            errs.append(f"trailer card {i} is longer than {rules.card_max_chars} characters")
    return errs
