"""Photo shots: measuring a still, and how long to hold it."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from nepal.process.stills import (sharpness, exposure_penalty, slot_duration_s,
                                  technical_score, passes_gate)

rng = np.random.default_rng(11)
CURVE = {"min_sharpness": 3.5, "max_exposure_pen": 0.20}


def flat(value=128, shape=(120, 160, 3)):
    return np.full(shape, value, dtype=np.uint8)


def noisy(shape=(120, 160, 3)):
    return (rng.random(shape) * 255).astype(np.uint8)


def test_detail_scores_sharper_than_a_flat_field():
    assert sharpness(noisy()) > sharpness(flat())


def test_sharpness_reads_greyscale_and_colour_alike():
    a = noisy()
    assert sharpness(a) == pytest.approx(sharpness(a[..., :3].mean(axis=2)), rel=1e-6)


def test_a_blown_out_frame_is_penalised():
    """Snow and sky make the highlight end the one that matters: a blown ridge
    carries no detail however sharp the lens was."""
    assert exposure_penalty(flat(253)) > exposure_penalty(flat(128))
    assert exposure_penalty(flat(253)) == pytest.approx(1.0, abs=0.01)


def test_a_crushed_frame_is_penalised():
    assert exposure_penalty(flat(2)) > 0.5


def test_a_well_exposed_frame_is_barely_penalised():
    assert exposure_penalty(flat(128)) < 0.05


def test_exposure_penalty_stays_in_range():
    for v in (0, 40, 128, 200, 255):
        assert 0.0 <= exposure_penalty(flat(v)) <= 1.0


def test_an_empty_frame_is_unusable_rather_than_a_crash():
    assert exposure_penalty(np.zeros((0, 0, 3), dtype=np.uint8)) == 1.0


def test_technical_score_rewards_sharp_and_well_exposed():
    assert technical_score(40.0, 0.02) > technical_score(40.0, 0.6)
    assert technical_score(40.0, 0.02) > technical_score(1.0, 0.02)


def test_technical_score_is_bounded():
    for s, e in ((0.0, 0.0), (1e9, 0.0), (1e9, 1.0), (-5.0, -1.0)):
        assert 0.0 <= technical_score(s, e) <= 1.0


def test_one_macro_shot_does_not_rescale_every_landscape():
    """Sharpness is heavy-tailed, so it is squashed against a reference rather
    than normalised across the set."""
    landscape = technical_score(12.0, 0.05)
    assert technical_score(1e6, 0.05) - landscape < 0.35


def test_the_quality_gate_uses_the_source_curve():
    assert passes_gate(10.0, 0.05, CURVE)
    assert not passes_gate(1.0, 0.05, CURVE)      # too soft
    assert not passes_gate(10.0, 0.9, CURVE)      # blown out


def test_a_still_is_held_long_enough_to_read_but_not_to_stall():
    assert 2.0 <= slot_duration_s(1.5) <= 4.5


def test_a_panorama_is_held_longer_because_the_eye_travels_further():
    assert slot_duration_s(3.0) > slot_duration_s(1.5)


def test_the_hold_never_escapes_its_bounds():
    assert slot_duration_s(99.0, base_s=10.0, max_s=4.5) == 4.5
    assert slot_duration_s(1.0, base_s=0.1, min_s=2.0) == 2.0
