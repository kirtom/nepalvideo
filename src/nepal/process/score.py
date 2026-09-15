"""S05 -- score every surviving shot, then shortlist.

Three scores per the spec, combined into one. Everything here is pure: given
a shot's measurements it returns a number, so the weights can be tuned against
a draft cut without re-measuring anything.

Two decisions the spec leaves open, both settled here because they decide what
the film is made of:

**A missing term must not read as a zero.** ``score_sem`` depends on
captioning, which on this account is gated behind AWS verification. Scoring a
shot with no caption as 0.0 semantic would not rank it low -- it would rank
every uncaptioned shot below every captioned one, which is a fact about the
pipeline's progress rather than about the film. Missing terms are therefore
dropped and the remaining weights renormalised, so a shot is judged on what is
actually known about it.

**Normalisation is against the corpus, not an absolute.** ``sharpness`` is a
log variance whose useful range depends on the camera and the light; the spec
says ``norm(sharpness)`` without saying norm against what. Percentile rank
within the corpus is the honest reading: it asks "is this sharp for this
material", which is the question a cut actually asks.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

log = logging.getLogger(__name__)

DURATION_PEAK_S = (4.0, 8.0)


def duration_fitness(seconds: float, peak: Sequence[float] = DURATION_PEAK_S) -> float:
    """1.0 inside the plateau, falling off smoothly either side.

    The spec asks for a peak at 4-8 s. Below the plateau the fall is linear to
    zero at 0 s; above it the fall is gentler, because a long shot can still be
    cut down while a short one cannot be extended.
    """
    lo, hi = float(peak[0]), float(peak[1])
    s = float(seconds)
    if s <= 0:
        return 0.0
    if s < lo:
        return s / lo
    if s <= hi:
        return 1.0
    return float(max(0.0, hi / s))          # 16 s -> 0.5, 24 s -> 0.33


def percentile_rank(values: Sequence[float]) -> dict[int, float]:
    """Map each index to its rank in 0..1 among the non-null values.

    Ties share the mean rank, so a corpus of identical values scores 0.5
    rather than a meaningless spread.
    """
    idx = [i for i, v in enumerate(values) if v is not None]
    if not idx:
        return {}
    arr = np.array([float(values[i]) for i in idx])
    order = arr.argsort()
    ranks = np.empty(len(arr), dtype=float)
    ranks[order] = np.arange(len(arr), dtype=float)
    # average the ranks of ties
    for v in np.unique(arr):
        mask = arr == v
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    if len(arr) == 1:
        return {idx[0]: 0.5}
    return {i: float(r / (len(arr) - 1)) for i, r in zip(idx, ranks)}


def weighted(terms: Mapping[str, float | None], weights: Mapping[str, float]) -> float | None:
    """Weighted mean over the terms that exist, renormalised.

    Returns None when nothing is known, which the caller must distinguish from
    a genuine zero -- see the module docstring.
    """
    num = den = 0.0
    for key, w in weights.items():
        v = terms.get(key)
        if v is None:
            continue
        num += float(w) * float(v)
        den += float(w)
    if den <= 0:
        return None
    return float(num / den)


def score_tech(shot: Mapping[str, Any], *, sharpness_rank: float | None,
               weights: Mapping[str, float],
               peak: Sequence[float] = DURATION_PEAK_S) -> float | None:
    dur = shot.get("end_s")
    start = shot.get("start_s")
    fit = (duration_fitness(float(dur) - float(start), peak)
           if dur is not None and start is not None else None)
    pen = shot.get("exposure_pen")
    return weighted({
        "sharpness": sharpness_rank,
        "exposure": None if pen is None else 1.0 - float(pen),
        "stability": shot.get("stability"),
        "duration_fit": fit,
    }, weights)


def score_sem(shot: Mapping[str, Any], *, clip_similarity: float | None,
              weights: Mapping[str, float]) -> float | None:
    vlm = shot.get("vlm_interest")
    return weighted({
        "vlm_interest": None if vlm is None else float(vlm) / 10.0,
        "clip_act_sim": clip_similarity,
    }, weights)


def score_ctx(shot: Mapping[str, Any], *, new_max_alt: bool, first_at_place: bool,
              msg_proximity: float, weights: Mapping[str, float]) -> float | None:
    return weighted({
        "has_face": 1.0 if shot.get("has_face") else 0.0,
        "has_speech": 1.0 if shot.get("has_speech") else 0.0,
        "new_max_alt": 1.0 if new_max_alt else 0.0,
        "first_at_place": 1.0 if first_at_place else 0.0,
        "msg_proximity": float(msg_proximity),
    }, weights)


def combine(tech: float | None, sem: float | None, ctx: float | None,
            weights: Mapping[str, float]) -> float | None:
    return weighted({"tech": tech, "sem": sem, "ctx": ctx}, weights)


def shortlist(scored: Sequence[Mapping[str, Any]], *, size: int,
              min_per_act: int) -> list[str]:
    """Top ``size`` by total, with a floor per act.

    The floor comes first: an act that photographs badly is still an act, and
    a purely global top-N can leave one with nothing to cut from. Acts are
    filled to the floor in rank order, then the remaining places go to the
    best of what is left regardless of act.
    """
    ranked = sorted((s for s in scored if s.get("score_total") is not None),
                    key=lambda s: -float(s["score_total"]))
    chosen: list[str] = []
    seen: set[str] = set()
    by_act: dict[Any, list[Mapping[str, Any]]] = {}
    for s in ranked:
        by_act.setdefault(s.get("act"), []).append(s)
    for act, rows in sorted(by_act.items(), key=lambda kv: (kv[0] is None, kv[0])):
        if act is None:
            continue                        # unplaced shots earn no floor
        for s in rows[:min_per_act]:
            if s["shot_id"] not in seen:
                chosen.append(s["shot_id"])
                seen.add(s["shot_id"])
    floor_n = len(chosen)
    for s in ranked:
        if len(chosen) >= size:
            break
        if s["shot_id"] not in seen:
            chosen.append(s["shot_id"])
            seen.add(s["shot_id"])
    if floor_n > size:
        # The floors are the point of stratifying, so they win over the cap:
        # trimming back to `size` would take shots from exactly the thin acts
        # the floor exists to protect. On this corpus it cannot happen
        # (5 acts x 40 = 200 against 400) but a config change could.
        log.warning("S05 per-act floors take %d shots, above shortlist_size %d; "
                    "keeping the floors", floor_n, size)
    return chosen
