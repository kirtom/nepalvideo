"""Selection rules shared by S05 scoring and S06 assembly.

Today this holds the photo/video mix. The film is made of motion and stills are
punctuation: the brief is a documentary, not a slideshow, and a still that has
to compete on equal terms with footage either loses everywhere or wins too
often. So the two are not ranked against each other at all. Stills are given a
fixed, small share of the runtime and made to compete among themselves for it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PhotoBudget:
    """How much runtime stills may take in one act, and which ones take it."""
    act: int
    act_duration_s: float
    budget_s: float
    chosen: tuple[str, ...]
    chosen_s: float
    n_candidates: int
    note: str = ""

    @property
    def spent_share(self) -> float:
        """The share of the act actually spent on stills, which is what the
        viewer experiences -- the budget is only an upper bound."""
        return (self.chosen_s / self.act_duration_s) if self.act_duration_s else 0.0


def budget_for_act(act_duration_s: float, share: float) -> float:
    """The seconds of an act that stills may occupy."""
    return max(0.0, float(act_duration_s)) * max(0.0, min(1.0, float(share)))


def allocate_photo_budget(candidates: Sequence[dict[str, Any]],
                          act_duration_s: float, *,
                          share: float = 0.10,
                          min_gap_slots: int = 0) -> PhotoBudget:
    """Choose which stills fill one act's share of the runtime, best first.

    ``candidates`` are dicts with ``shot_id``, ``score`` and ``duration_s``,
    for one act. They are taken in descending score until the next one would
    overrun the budget -- and then the loop keeps going rather than stopping, so
    a short strong still can still fit where a long one could not. That is the
    whole rule: no still competes with footage, and the ones that get in are the
    ones that were worth holding on screen.

    A budget too small for even the shortest candidate yields nothing, which is
    correct: one still in a ninety-second act is a glitch, not punctuation.
    """
    act = int(candidates[0].get("act", 0)) if candidates else 0
    budget = budget_for_act(act_duration_s, share)
    if not candidates:
        return PhotoBudget(act, act_duration_s, budget, (), 0.0, 0,
                           "no photo candidates")
    if budget <= 0:
        return PhotoBudget(act, act_duration_s, 0.0, (), 0.0, len(candidates),
                           "photo share is zero for this act")

    ordered = sorted(candidates, key=lambda c: (-float(c.get("score", 0.0)),
                                                str(c.get("shot_id", ""))))
    chosen: list[str] = []
    spent = 0.0
    for c in ordered:
        d = float(c.get("duration_s", 0.0))
        if d <= 0:
            continue
        if spent + d <= budget + 1e-9:
            chosen.append(str(c["shot_id"]))
            spent += d
    note = ""
    if not chosen:
        shortest = min((float(c.get("duration_s", 0.0)) for c in candidates
                        if float(c.get("duration_s", 0.0)) > 0), default=0.0)
        note = (f"budget {budget:.1f}s is shorter than the shortest still "
                f"({shortest:.1f}s)")
    return PhotoBudget(act, act_duration_s, round(budget, 3), tuple(chosen),
                       round(spent, 3), len(candidates), note)


def plan_photo_slots(candidates_by_act: dict[int, Sequence[dict[str, Any]]],
                     act_durations: dict[int, float], *,
                     share: float = 0.10,
                     share_by_act: dict[int, float] | None = None
                     ) -> dict[int, PhotoBudget]:
    """Run the budget per act.

    Per act rather than per film, so the share cannot pool into one place: ten
    percent of a twenty-minute film spent entirely inside Act 1 would make the
    planning section a slideshow and leave the trek without a single still.

    ``share_by_act`` overrides the global share for named acts. Act 4 is the
    one worth a deliberate decision -- the brief makes it a single swell into a
    hard cut to silence, and whether a held frame belongs in that is a creative
    call rather than something this function should assume.
    """
    over = share_by_act or {}
    return {act: allocate_photo_budget(candidates_by_act.get(act, ()),
                                       act_durations.get(act, 0.0),
                                       share=float(over.get(act, share)))
            for act in sorted(act_durations)}
