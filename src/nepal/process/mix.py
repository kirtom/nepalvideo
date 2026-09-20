"""Film v2 step 4, Part B -- the mix as gain envelopes (spec section 6, task
12). Task 11's cues (``cues.py``) say when speech, location sound and music
play and at what level; this module turns spans and cues into an
``Envelope`` -- a piecewise-linear gain curve over the whole film -- and
``volume_expr`` renders one as an ffmpeg ``volume`` expression. Task 13's
renderer consumes ``volume_expr``'s output as an opaque string; Task 14
wires the numbers below from ``config/pipeline.yaml``.

Pure on purpose, like ``rhythm.py``: no DB, no config, no file reads, no
shelling out. Every tunable is a keyword argument so a change to the config
never means a change here.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

Envelope = list[tuple[float, float]]

# Not "quieter" -- off. ffmpeg's volume filter treats anything this low as
# inaudible, so a hard duck (a natural-sound window, Act 4's silence) can
# aim at one fixed floor instead of a per-scene "quiet enough" judgement.
FLOOR_DB = -70.0

# Priority among the three levels a moment of music can be at: the floor
# (a window or the silence) always wins over a speech dip, which always
# wins over the base bed -- "a window inside a speech span is a window."
_PRIORITY = {"base": 0, "speech": 1, "floor": 2}


def _merge_spans(spans: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sorted, non-overlapping spans -- two overlapping speech spans become
    one dip, not two."""
    ordered = sorted((float(a), float(b)) for a, b in spans if b > a)
    out: list[tuple[float, float]] = []
    for a, b in ordered:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _floor_spans(windows: Sequence[Mapping[str, Any]],
                  silence: Mapping[str, Any] | None) -> list[tuple[float, float]]:
    spans = [(float(w["t_in"]), float(w["t_out"])) for w in windows]
    if silence:
        spans.append((float(silence["t_start"]), float(silence["t_end"])))
    return _merge_spans(spans)


def _pieces(total_s: float, speech: Sequence[tuple[float, float]],
            floor: Sequence[tuple[float, float]]) -> list[tuple[float, float, str]]:
    """The timeline cut into maximal runs of one kind ("base", "speech" or
    "floor"), covering [0, total_s] with no gaps. Adjacent same-kind pieces
    are merged here rather than left for the caller, so a boundary in the
    output always means a real change of level."""
    bounds = {0.0, total_s}
    for a, b in speech:
        bounds.add(a); bounds.add(b)
    for a, b in floor:
        bounds.add(a); bounds.add(b)
    ordered = sorted(t for t in bounds if 0.0 <= t <= total_s)
    pieces: list[tuple[float, float, str]] = []
    for a, b in zip(ordered, ordered[1:]):
        if b <= a:
            continue
        mid = (a + b) / 2.0
        if any(f0 <= mid <= f1 for f0, f1 in floor):
            kind = "floor"
        elif any(s0 <= mid <= s1 for s0, s1 in speech):
            kind = "speech"
        else:
            kind = "base"
        if pieces and pieces[-1][2] == kind:
            pieces[-1] = (pieces[-1][0], b, kind)
        else:
            pieces.append((a, b, kind))
    return pieces


def _finalize(points: Sequence[tuple[float, float]]) -> Envelope:
    """Sort and drop duplicate times. A duplicate only ever arises from two
    adjacent boundaries computing the same shared instant (always with the
    same value, since the curve is continuous by construction), so keeping
    the first is never a choice between two different answers."""
    ordered = sorted(points, key=lambda p: p[0])
    out: Envelope = []
    for t, v in ordered:
        if out and math.isclose(t, out[-1][0], abs_tol=1e-9):
            continue
        out.append((t, v))
    return out


def _breakpoints_from_pieces(pieces: Sequence[tuple[float, float, str]], *,
                              cue_fade_s: float, window_fade_s: float,
                              base_db: float, under_speech_db: float) -> list[tuple[float, float]]:
    """One ramp per boundary between pieces, owned by whichever side is
    deeper (``_PRIORITY``): entering a deeper piece ramps inside its own
    start, leaving one ramps inside its own end -- so a window nested in a
    speech span dips to the floor and returns to the speech level, not to
    the base bed either side of it. Each ramp is clamped to half its own
    piece's length so two ramps at either end of a very short piece meet in
    the middle rather than cross (which would put breakpoints out of
    order)."""
    db = {"floor": FLOOR_DB, "speech": under_speech_db, "base": base_db}
    fade_for = {"speech": cue_fade_s, "floor": window_fade_s}
    pts: list[tuple[float, float]] = [(pieces[0][0], db[pieces[0][2]])]
    for i in range(len(pieces) - 1):
        a, b, kind = pieces[i]
        a2, b2, kind2 = pieces[i + 1]
        if _PRIORITY[kind] > _PRIORITY[kind2]:
            fade = min(fade_for[kind], (b - a) / 2.0)
            pts.append((b - fade, db[kind]))
            pts.append((b, db[kind2]))
        else:
            fade = min(fade_for[kind2], (b2 - a2) / 2.0)
            pts.append((b, db[kind]))
            pts.append((b + fade, db[kind2]))
    return pts


def music_envelope(*, total_s: float, speech_spans: Sequence[tuple[float, float]],
                    windows: Sequence[Mapping[str, Any]], silence: Mapping[str, Any] | None,
                    base_db: float = 0.0, under_speech_db: float = -8.0,
                    window_fade_s: float = 1.0, cue_fade_s: float = 0.15) -> Envelope:
    """The music bed's gain over the whole film: ``base_db`` everywhere,
    ``under_speech_db`` inside a speech span (ramped over ``cue_fade_s``),
    ``FLOOR_DB`` inside a natural-sound window or the silence window
    (ramped over ``window_fade_s``) -- the floor always wins where a window
    falls inside a speech span."""
    total_s = float(total_s)
    pieces = _pieces(total_s, _merge_spans(speech_spans), _floor_spans(windows, silence))
    if not pieces:
        return [(0.0, base_db)]
    pts = _breakpoints_from_pieces(pieces, cue_fade_s=cue_fade_s, window_fade_s=window_fade_s,
                                    base_db=base_db, under_speech_db=under_speech_db)
    return _finalize(pts)


def location_envelope(*, total_s: float, cues: Sequence[Mapping[str, Any]],
                       full_lufs: float, cue_fade_s: float = 0.15) -> Envelope:
    """The location track's gain, relative to ``full_lufs`` (0 dB): before
    the first cue, 0 dB; at each cue whose level differs from what is
    already held, a ``cue_fade_s`` ramp to it; between cues -- even across a
    gap -- the previous cue's level is simply held, since two points at the
    same value interpolate flat on their own. A cue with the same
    ``gain_lufs`` as what's already playing draws no new ramp at all.

    Fades are clamped to half the gap to the neighbouring transition on
    either side, the same guard ``music_envelope`` uses, so two cues placed
    closer together than ``2 * cue_fade_s`` still produce ordered
    breakpoints instead of crossing ramps.
    """
    ordered = sorted(cues, key=lambda c: float(c["t_in"]))
    transitions: list[tuple[float, float]] = []
    level = 0.0
    for c in ordered:
        target = float(c["gain_lufs"]) - float(full_lufs)
        if not math.isclose(target, level, abs_tol=1e-9):
            transitions.append((float(c["t_in"]), target))
            level = target

    pts: Envelope = [(0.0, 0.0)]
    for i, (t, target) in enumerate(transitions):
        prev_t = transitions[i - 1][0] if i > 0 else 0.0
        next_t = transitions[i + 1][0] if i + 1 < len(transitions) else math.inf
        fade = min(cue_fade_s, (t - prev_t) / 2.0, (next_t - t) / 2.0)
        held = pts[-1][1]
        pts.append((t, held))
        pts.append((t + fade, target))
    return _finalize(pts)


# -- ffmpeg expression -----------------------------------------------------

def _fmt(x: float) -> str:
    """A plain decimal ffmpeg's expression parser accepts -- repr()/str()
    can fall back to scientific notation for very small magnitudes, which
    the parser does not."""
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s if s and s != "-0" else "0"


def _gain(db: float) -> str:
    return f"pow(10,{db:.3f}/20)"


def volume_expr(env: Envelope) -> str:
    """``env`` as an ffmpeg ``volume`` expression, in linear gain
    (``pow(10, dB/20)``): a flat sum of terms, one per segment, plus a final
    term that holds the last value forever after. ``eval=frame`` is added by
    the renderer.

    Not the nested ``if(lt(t,T1), ..., if(lt(t,T2), ...))`` form: ffmpeg's
    expression parser recurses per nested ``if``, and a film's envelope runs
    to hundreds of breakpoints -- deep nesting has failed in practice around
    a few hundred levels, exactly the range this module produces. A flat sum
    has no nesting that grows with the breakpoint count at all.

    ``between(t,T0,T1)`` is closed on both ends in ffmpeg, so two adjacent
    segments built on it would both fire, and so both get summed, at the
    exact instant they share -- doubling that one instant's value. An
    earlier version tried to dodge this by shrinking each segment's upper
    bound by a fixed epsilon, which instead opened a real epsilon-wide gap
    of near-zero gain just before every interior breakpoint. Each segment
    is therefore ``gte(t,T0)*lt(t,T1)`` -- the exact, epsilon-free
    half-open interval ``[T0, T1)`` -- so the shared instant belongs to
    exactly one segment (or to the final held tail, for the very last
    breakpoint) with no gap and no double count.
    """
    if len(env) == 1:
        return _gain(env[0][1])
    terms = []
    for (t0, d0), (t1, d1) in zip(env, env[1:]):
        g0, g1 = _gain(d0), _gain(d1)
        terms.append(f"gte(t,{_fmt(t0)})*lt(t,{_fmt(t1)})*"
                     f"({g0}+({g1}-{g0})*(t-{_fmt(t0)})/({_fmt(t1)}-{_fmt(t0)}))")
    t_last, d_last = env[-1]
    terms.append(f"gte(t,{_fmt(t_last)})*{_gain(d_last)}")
    return "+".join(terms)
