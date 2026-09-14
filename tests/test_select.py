"""The photo/video mix: stills are punctuation, and only the best get in."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest

from nepal.select import budget_for_act, allocate_photo_budget, plan_photo_slots

ACT_DURATIONS = {1: 167.0, 2: 287.0, 3: 407.0, 4: 113.0, 5: 227.0}


def cands(n, act=3, dur=3.0, best_first=True):
    return [{"shot_id": f"p{i}", "act": act, "duration_s": dur,
             "score": (1.0 - i * 0.05) if best_first else (i * 0.05)}
            for i in range(n)]


def test_the_budget_is_a_share_of_the_act():
    assert budget_for_act(400.0, 0.10) == pytest.approx(40.0)
    assert budget_for_act(400.0, 0.0) == 0.0


def test_the_budget_clamps_a_nonsense_share():
    assert budget_for_act(400.0, 5.0) == pytest.approx(400.0)
    assert budget_for_act(400.0, -1.0) == 0.0
    assert budget_for_act(-10.0, 0.5) == 0.0


def test_stills_never_exceed_their_share():
    b = allocate_photo_budget(cands(100), 407.0, share=0.10)
    assert b.chosen_s <= b.budget_s
    assert b.spent_share <= 0.10 + 1e-9


def test_the_best_stills_are_the_ones_that_get_in():
    b = allocate_photo_budget(cands(40), 200.0, share=0.10)
    assert b.chosen[0] == "p0"          # highest score first
    assert "p39" not in b.chosen        # worst does not displace better


def test_a_candidate_that_overruns_is_skipped_not_a_full_stop():
    """The loop keeps going past a candidate that will not fit, so a later,
    shorter still can use the budget the long one could not."""
    mixed = [
        {"shot_id": "too_long", "act": 1, "score": 1.00, "duration_s": 12.0},
        {"shot_id": "b",        "act": 1, "score": 0.90, "duration_s": 4.0},
        {"shot_id": "c",        "act": 1, "score": 0.80, "duration_s": 5.0},
        {"shot_id": "no_room",  "act": 1, "score": 0.70, "duration_s": 3.0},
        {"shot_id": "squeeze",  "act": 1, "score": 0.60, "duration_s": 1.0},
    ]
    b = allocate_photo_budget(mixed, 100.0, share=0.10)   # 10 s budget
    assert b.chosen == ("b", "c", "squeeze")
    assert b.chosen_s == pytest.approx(10.0)


def test_a_budget_too_small_for_any_still_takes_none():
    """One still in a ninety-second act is a glitch, not punctuation."""
    b = allocate_photo_budget(cands(5, dur=6.0), 20.0, share=0.10)
    assert b.chosen == ()
    assert "shorter than the shortest still" in b.note


def test_no_candidates_is_not_an_error():
    b = allocate_photo_budget([], 400.0)
    assert b.chosen == () and b.n_candidates == 0
    assert b.note == "no photo candidates"


def test_a_zero_share_excludes_stills_from_the_act():
    b = allocate_photo_budget(cands(10), 113.0, share=0.0)
    assert b.chosen == ()
    assert "zero" in b.note


def test_zero_length_candidates_are_ignored():
    junk = [{"shot_id": "z", "act": 1, "score": 9.9, "duration_s": 0.0}]
    assert allocate_photo_budget(junk + cands(3), 400.0).chosen[0] != "z"


def test_the_share_cannot_pool_into_one_act():
    """Ten percent of the film spent entirely inside Act 1 would make the
    planning section a slideshow and leave the trek without a single still."""
    plan = plan_photo_slots({a: cands(100, act=a) for a in ACT_DURATIONS},
                            ACT_DURATIONS, share=0.10)
    for act, b in plan.items():
        assert b.spent_share <= 0.10 + 1e-9, f"act {act} over its share"
        assert b.chosen, f"act {act} got no stills at all"


def test_the_whole_film_stays_near_the_target_share():
    plan = plan_photo_slots({a: cands(100, act=a) for a in ACT_DURATIONS},
                            ACT_DURATIONS, share=0.10)
    total = sum(b.chosen_s for b in plan.values())
    assert 0.085 <= total / sum(ACT_DURATIONS.values()) <= 0.10


def test_an_act_can_be_excluded_by_override():
    """Act 4 is one swell into a hard cut to silence -- whether a held frame
    belongs there is the operator's call, not this function's assumption."""
    plan = plan_photo_slots({a: cands(50, act=a) for a in ACT_DURATIONS},
                            ACT_DURATIONS, share=0.10, share_by_act={4: 0.0})
    assert plan[4].chosen == ()
    assert plan[3].chosen, "overriding one act must not disturb the others"


def test_an_override_can_also_raise_an_acts_share():
    plan = plan_photo_slots({a: cands(90, act=a) for a in ACT_DURATIONS},
                            ACT_DURATIONS, share=0.10, share_by_act={1: 0.30})
    assert plan[1].spent_share > 0.2
    assert plan[2].spent_share <= 0.10 + 1e-9


def test_every_act_is_reported_even_with_no_candidates():
    plan = plan_photo_slots({}, ACT_DURATIONS, share=0.10)
    assert sorted(plan) == sorted(ACT_DURATIONS)
    assert all(b.chosen == () for b in plan.values())
