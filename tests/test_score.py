"""S05 -- the scoring arithmetic, which decides what the film is made of."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest

from nepal.process import score

TECH = {"sharpness": 0.35, "exposure": 0.25, "stability": 0.30, "duration_fit": 0.10}
SEM = {"vlm_interest": 0.60, "clip_act_sim": 0.40}
CTX = {"has_face": 0.25, "has_speech": 0.20, "new_max_alt": 0.20,
       "first_at_place": 0.20, "msg_proximity": 0.15}
TOTAL = {"tech": 0.40, "sem": 0.35, "ctx": 0.25}


# -- duration fitness --------------------------------------------------

@pytest.mark.parametrize("s", [4.0, 5.5, 8.0])
def test_a_shot_in_the_plateau_is_a_perfect_length(s):
    assert score.duration_fitness(s) == 1.0


def test_shorter_than_the_plateau_falls_off_linearly():
    assert score.duration_fitness(2.0) == pytest.approx(0.5)
    assert score.duration_fitness(0.0) == 0.0


def test_longer_than_the_plateau_falls_off_more_gently_than_shorter():
    """A long shot can be trimmed; a short one cannot be extended."""
    over = score.duration_fitness(8.0 * 2)      # twice the plateau top
    under = score.duration_fitness(4.0 / 2)     # half the plateau bottom
    assert over == pytest.approx(0.5)
    assert under == pytest.approx(0.5)
    assert score.duration_fitness(24.0) > 0.0   # never a hard zero


# -- the missing-term rule ---------------------------------------------

def test_a_missing_term_is_dropped_not_scored_zero():
    """The rule that keeps an uncaptioned corpus rankable: a shot with no
    caption must be judged on what IS known, not pushed below every
    captioned shot."""
    both = score.weighted({"a": 1.0, "b": 0.0}, {"a": 0.5, "b": 0.5})
    only_a = score.weighted({"a": 1.0, "b": None}, {"a": 0.5, "b": 0.5})
    assert both == pytest.approx(0.5)
    assert only_a == pytest.approx(1.0), "renormalised, not halved"


def test_nothing_known_is_none_not_zero():
    assert score.weighted({"a": None}, {"a": 1.0}) is None


def test_sem_survives_having_no_caption_at_all():
    """Exactly the state the corpus is in while Bedrock is gated."""
    s = score.score_sem({"vlm_interest": None}, clip_similarity=0.8, weights=SEM)
    assert s == pytest.approx(0.8)
    assert score.score_sem({}, clip_similarity=None, weights=SEM) is None


def test_total_ignores_a_whole_missing_score():
    with_sem = score.combine(1.0, 0.0, 1.0, TOTAL)
    without = score.combine(1.0, None, 1.0, TOTAL)
    assert without == pytest.approx(1.0)
    assert with_sem < without


# -- percentile rank ---------------------------------------------------

def test_rank_spreads_the_corpus_over_zero_to_one():
    r = score.percentile_rank([10.0, 20.0, 30.0])
    assert r[0] == 0.0 and r[2] == 1.0 and r[1] == pytest.approx(0.5)


def test_identical_values_all_score_the_middle_not_a_spread():
    r = score.percentile_rank([5.0, 5.0, 5.0])
    assert list(r.values()) == pytest.approx([0.5, 0.5, 0.5])


def test_nulls_are_skipped_and_keep_their_index():
    r = score.percentile_rank([1.0, None, 9.0])
    assert set(r) == {0, 2} and r[0] == 0.0 and r[2] == 1.0


def test_a_single_value_is_the_middle_not_the_top():
    assert score.percentile_rank([42.0]) == {0: 0.5}


# -- shortlist ---------------------------------------------------------

def _rows(n_per_act):
    out, k = [], 0
    for act, n in n_per_act.items():
        for i in range(n):
            k += 1
            out.append({"shot_id": f"s{k}", "act": act, "score_total": 1.0 - k * 1e-4})
    return out


def test_the_best_shots_are_taken_first():
    rows = _rows({1: 50})
    got = score.shortlist(rows, size=10, min_per_act=0)
    assert got == [r["shot_id"] for r in rows[:10]]


def test_a_thin_act_still_gets_its_floor():
    """Act 2 scores worse than everything in act 1; without the floor a
    global top-N would leave it with nothing to cut from."""
    rows = _rows({1: 100, 2: 20})
    got = score.shortlist(rows, size=50, min_per_act=10)
    act2 = [r["shot_id"] for r in rows if r["act"] == 2]
    assert sum(1 for s in got if s in act2) >= 10


def test_an_act_with_less_than_the_floor_contributes_what_it_has():
    rows = _rows({1: 100, 2: 3})
    got = score.shortlist(rows, size=50, min_per_act=10)
    assert sum(1 for s in got if s in [r["shot_id"] for r in rows if r["act"] == 2]) == 3


def test_the_shortlist_has_no_duplicates():
    got = score.shortlist(_rows({1: 30, 2: 30}), size=40, min_per_act=10)
    assert len(got) == len(set(got))


def test_unscored_shots_are_not_shortlisted():
    rows = [{"shot_id": "a", "act": 1, "score_total": None},
            {"shot_id": "b", "act": 1, "score_total": 0.5}]
    assert score.shortlist(rows, size=10, min_per_act=5) == ["b"]


def test_an_unplaced_shot_earns_no_floor_but_can_still_win_on_merit():
    rows = [{"shot_id": "top", "act": None, "score_total": 0.99},
            {"shot_id": "a1", "act": 1, "score_total": 0.5}]
    got = score.shortlist(rows, size=10, min_per_act=1)
    assert "top" in got and "a1" in got
