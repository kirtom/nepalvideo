"""S03.3 -- the four technical metrics."""
from __future__ import annotations

import numpy as np
import pytest

from nepal.process import metrics


def flat(shape=(64, 128)):
    return np.full(shape, 128.0, dtype=np.float32)


def noisy(shape=(64, 128), seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=shape).astype(np.float32)


def shifted(img, dx: int, dy: int = 0):
    return np.roll(np.roll(img, dx, axis=1), dy, axis=0)


# -- sample placement ------------------------------------------------
def test_samples_avoid_both_cuts():
    """The first and last frames of a shot are the ones most likely to be a
    dissolve or a smear, so no sample may land on either boundary."""
    ts = metrics.sample_times(10.0, 20.0, 5)
    assert len(ts) == 5
    assert ts[0] > 10.0 and ts[-1] < 20.0
    assert ts == sorted(ts)


def test_samples_of_a_zero_length_shot_do_not_escape_it():
    assert metrics.sample_times(7.0, 7.0, 5) == [7.0] * 5


# -- sharpness -------------------------------------------------------
def test_sharpness_is_logged_not_raw():
    """The curves' min_sharpness of 4.0 is a log value. A raw Laplacian
    variance would sit in the thousands and pass everything."""
    assert 0.0 < metrics.sharpness(noisy()) < 20.0


def test_a_flat_frame_has_no_detail():
    assert metrics.sharpness(flat()) == 0.0
    assert metrics.sharpness(noisy()) > metrics.sharpness(flat())


# -- exposure --------------------------------------------------------
def test_exposure_penalty_counts_only_clipping():
    """Not drift from mid-grey: a snowfield is the subject of this film, not a
    badly exposed photograph."""
    bright = np.full((64, 128), 200.0, dtype=np.float32)
    assert metrics.exposure_penalty(bright) == 0.0
    blown = np.full((64, 128), 255.0, dtype=np.float32)
    assert metrics.exposure_penalty(blown) == pytest.approx(1.0)


def test_exposure_penalty_is_the_clipped_fraction():
    img = flat()
    img[:32] = 255.0
    assert metrics.exposure_penalty(img) == pytest.approx(0.5)


# -- the equirect band -----------------------------------------------
def test_the_poles_of_an_equirect_frame_are_excluded():
    """Sky and the photographer's feet occupy half an equirect proxy and blur
    every measurement toward the same value."""
    img = np.zeros((100, 200), dtype=np.float32)
    img[25:75] = 200.0
    assert metrics.central_band(img).mean() == pytest.approx(200.0)


def test_a_frame_too_short_to_band_is_kept_whole():
    img = np.ones((6, 12), dtype=np.float32)
    assert metrics.central_band(img).shape == img.shape


# -- motion and shake ------------------------------------------------
def test_a_steady_pan_is_stable_and_a_shaken_camera_is_not():
    """Shake is not motion: it is the change in motion. A pan at a constant
    rate must score as stable, or every walking shot in the film is rejected."""
    base = noisy((128, 256))
    pan = [base, shifted(base, 4), shifted(base, 8)]
    shake = [base, shifted(base, 6), shifted(base, -6)]
    _, pan_jerk = metrics.motion_of(pan, width=256)
    _, shake_jerk = metrics.motion_of(shake, width=256)
    assert pan_jerk < shake_jerk
    assert metrics.stability_from_jerk(pan_jerk) > metrics.stability_from_jerk(shake_jerk)


def test_a_locked_off_camera_is_perfectly_stable():
    base = noisy((128, 256))
    _, jerk = metrics.motion_of([base, base, base], width=256)
    assert metrics.stability_from_jerk(jerk) == pytest.approx(1.0, abs=1e-3)


def test_motion_needs_two_frames_and_shake_needs_three():
    base = noisy((64, 128))
    assert metrics.motion_of([base]) == (0.0, 0.0)
    mag, jerk = metrics.motion_of([base, shifted(base, 3)], width=128)
    assert mag > 0 and jerk == 0.0    # nothing observed, not "no shake"


def test_stability_falls_as_jerk_rises():
    xs = [metrics.stability_from_jerk(j) for j in (0, 1, 4, 16)]
    assert xs == sorted(xs, reverse=True)
    assert xs[0] == 1.0 and 0.0 < xs[-1] < 0.3


# -- folding samples into one verdict --------------------------------
def test_measure_samples_reports_all_four_metrics():
    base = noisy((128, 256))
    groups = [[base, shifted(base, 2), shifted(base, 4)] for _ in range(3)]
    m = metrics.measure_samples(groups, width=256)
    assert set(m) >= {"sharpness", "exposure_pen", "motion_mag", "stability"}
    assert m["n_samples"] == 3
    assert 0.0 <= m["stability"] <= 1.0


def test_one_bad_sample_does_not_decide_a_shot():
    """Medians, not means: one frame of lens flare should not reject a shot."""
    good = noisy((64, 128))
    blown = np.full((64, 128), 255.0, dtype=np.float32)
    groups = [[good, good], [good, good], [blown, blown], [good, good], [good, good]]
    m = metrics.measure_samples(groups, width=128)
    assert m["exposure_pen"] == pytest.approx(metrics.exposure_penalty(good), abs=1e-4)
    assert m["exposure_pen"] < 0.5


def test_a_shot_with_no_readable_frames_measures_nothing():
    assert metrics.measure_samples([[], []]) is None


def test_percentiles_of_nothing_is_empty_not_an_error():
    assert metrics.percentiles([]) == {}
    assert metrics.percentiles([1.0, 2.0, 3.0])["p50"] == 2.0
