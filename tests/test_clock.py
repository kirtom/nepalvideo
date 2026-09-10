import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from nepal.probe.clock import (gcc_phat, offset_from_pair, reduce_measurements,
                               PairMeasurement, Clip, find_candidate_pairs, naive_offset)

FS = 16000
rng = np.random.default_rng(11)


def delayed_pair(delay_samples, n=FS * 8, snr_db=None):
    """Two windows onto one source, offset by ``delay_samples``.

    Returns (a, b) such that a is b delayed by delay_samples.
    """
    pad = FS * 5
    source = rng.normal(0, 1, n + 2 * pad)
    base = pad
    b = source[base: base + n].copy()
    a = source[base - delay_samples: base - delay_samples + n].copy()
    if snr_db is not None:
        for sig in (a, b):
            noise = rng.normal(0, 10 ** (-snr_db / 20), sig.shape)
            sig += noise
    return a, b


@pytest.mark.parametrize("delay", [0, 1, 160, 1600, -160, -1600, 8000, -8000])
def test_recovers_known_delay_and_sign(delay):
    a, b = delayed_pair(delay)
    lag_s, conf = gcc_phat(a, b, FS, max_lag_s=2.0)
    assert lag_s == pytest.approx(delay / FS, abs=1.5 / FS), f"got {lag_s*FS:.1f} samples"
    assert conf > 3.0


def test_sign_convention_is_documented_direction():
    """Positive lag == the sound lands LATER in `a` than in `b`."""
    a, b = delayed_pair(1600)          # a is b delayed by 0.1 s
    lag_s, _ = gcc_phat(a, b, FS, max_lag_s=2.0)
    assert lag_s > 0
    lag_back, _ = gcc_phat(b, a, FS, max_lag_s=2.0)
    assert lag_back == pytest.approx(-lag_s, abs=2.0 / FS)


def test_survives_different_mic_gain_and_colouration():
    """Two devices, two microphones -- PHAT should not care."""
    a, b = delayed_pair(2400)
    from scipy.signal import lfilter
    a = lfilter([1.0, -0.7], [1.0], a) * 0.05      # tinny + quiet
    b = lfilter([0.3, 0.4, 0.3], [1.0], b) * 8.0   # muffled + loud
    lag_s, conf = gcc_phat(a, b, FS, max_lag_s=2.0)
    assert lag_s == pytest.approx(2400 / FS, abs=3.0 / FS)
    assert conf > 2.0


def test_survives_moderate_additive_noise():
    a, b = delayed_pair(3200, snr_db=6)
    lag_s, conf = gcc_phat(a, b, FS, max_lag_s=2.0)
    assert lag_s == pytest.approx(3200 / FS, abs=3.0 / FS)
    assert conf > 1.5


def test_unrelated_audio_yields_low_confidence():
    a = rng.normal(0, 1, FS * 8)
    b = rng.normal(0, 1, FS * 8)
    _, conf = gcc_phat(a, b, FS, max_lag_s=2.0)
    assert conf < 1.5, "unrelated clips must fall under the acceptance threshold"


def test_empty_input_is_safe():
    assert gcc_phat(np.array([]), np.ones(10), FS) == (0.0, 0.0)


def test_lag_search_is_bounded_by_max_lag():
    a, b = delayed_pair(FS * 3)                     # 3 s of delay
    lag_s, _ = gcc_phat(a, b, FS, max_lag_s=1.0)    # but only search +/- 1 s
    assert abs(lag_s) <= 1.0


# -- offset arithmetic -------------------------------------------------

def _t(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def test_offset_from_pair_zero_case():
    t = _t("2023-10-15T10:00:00")
    assert offset_from_pair(t, t, 0.0) == 0.0


def test_phone_clock_running_late_gives_positive_offset():
    """Phone says 09:59:00 for a moment the camera calls 10:00:00.

    The phone is 60 s slow, so +60 s must be added to its timestamps.
    """
    cam = _t("2023-10-15T10:00:00")
    phone = _t("2023-10-15T09:59:00")
    assert offset_from_pair(cam, phone, 0.0) == pytest.approx(60.0)


def test_measured_lag_is_subtracted():
    cam = _t("2023-10-15T10:00:00")
    phone = _t("2023-10-15T10:00:00")
    # the shared sound appears 2 s later into the phone recording, so the phone
    # actually started 2 s earlier in real time than it claims
    assert offset_from_pair(cam, phone, 2.0) == pytest.approx(-2.0)


def test_roundtrip_offset_recovers_truth():
    """End to end: inject a known clock error, measure it back."""
    true_offset = -37.0                      # phone clock runs 37 s fast
    cam_start = _t("2023-10-15T10:00:00")
    phone_start_reported = cam_start - timedelta(seconds=true_offset)
    a, b = delayed_pair(0)
    lag_s, _ = gcc_phat(a, b, FS, max_lag_s=2.0)
    got = offset_from_pair(cam_start, phone_start_reported, lag_s)
    assert got == pytest.approx(true_offset, abs=0.01)


# -- reduction ---------------------------------------------------------

def _m(offset, conf=3.0, cid="c", pid="p"):
    return PairMeasurement(cid, pid, lag_s=0.0, confidence=conf, offset_s=offset)


def test_median_of_confident_pairs():
    r = reduce_measurements("keller", [_m(-37.1), _m(-36.9), _m(-37.0), _m(-37.2)])
    assert r.offset_s == pytest.approx(-37.05, abs=0.01)
    assert r.n_pairs_accepted == 4
    assert not r.needs_manual


def test_median_rejects_a_wind_gust_outlier():
    """One pair that locked onto the wrong peak must not move the answer."""
    r = reduce_measurements("keller", [_m(-37.0), _m(-37.1), _m(-36.9), _m(412.0)])
    assert r.offset_s == pytest.approx(-37.0, abs=0.11)


def test_low_confidence_pairs_are_excluded():
    r = reduce_measurements("keller", [_m(-37.0), _m(-37.0), _m(-37.0), _m(900.0, conf=1.1)])
    assert r.n_pairs_accepted == 3
    assert r.offset_s == pytest.approx(-37.0)
    excluded = [m for m in r.measurements if not m.accepted]
    assert len(excluded) == 1 and "confidence" in excluded[0].note


def test_too_few_pairs_falls_back_to_manual_with_prefill():
    r = reduce_measurements("kulikov", [_m(-37.0), _m(-37.0)],
                            min_confident_pairs=3, naive_offset_s=-40.0)
    assert r.needs_manual
    assert r.confidence == 0.0
    assert r.offset_s == -40.0, "Gate 1 field must be prefilled with the naive delta"
    assert "manual" in r.method


def test_disagreeing_pairs_lower_the_confidence():
    tight = reduce_measurements("k", [_m(-37.0), _m(-37.0), _m(-37.05)])
    loose = reduce_measurements("k", [_m(-37.0), _m(-25.0), _m(-49.0)])
    assert tight.confidence > loose.confidence
    assert loose.spread_s == pytest.approx(24.0)


# -- pairing -----------------------------------------------------------

def test_finds_overlapping_pairs_and_locates_the_overlap():
    cam = [Clip("c1", "camera", _t("2023-10-15T10:00:00"), 600, "c1.insv")]
    ph = [Clip("p1", "phone_keller", _t("2023-10-15T10:05:00"), 300, "p1.mp4")]
    pairs = find_candidate_pairs(cam, ph, window_s=600)
    assert len(pairs) == 1
    _, _, cam_off, ph_off = pairs[0]
    assert cam_off == pytest.approx(300.0), "read from 5 min into the camera clip"
    assert ph_off == pytest.approx(0.0)


def test_silent_clips_are_not_paired():
    cam = [Clip("c1", "camera", _t("2023-10-15T10:00:00"), 600, "c1.insv", has_audio=False)]
    ph = [Clip("p1", "phone_keller", _t("2023-10-15T10:00:00"), 300, "p1.mp4")]
    assert find_candidate_pairs(cam, ph) == []


def test_distant_clips_are_not_paired():
    cam = [Clip("c1", "camera", _t("2023-10-15T10:00:00"), 60, "c1.insv")]
    ph = [Clip("p1", "phone_keller", _t("2023-10-15T14:00:00"), 60, "p1.mp4")]
    assert find_candidate_pairs(cam, ph, window_s=600) == []


def test_naive_offset_is_the_earliest_timestamp_difference():
    cam = [Clip("c1", "camera", _t("2023-10-15T10:00:00"), 60, "c")]
    ph = [Clip("p1", "phone_keller", _t("2023-10-15T09:59:00"), 60, "p")]
    assert naive_offset(cam, ph) == pytest.approx(60.0)


# -- the Gate 1 prefill ------------------------------------------------

def test_naive_prefill_uses_pairs_not_corpus_extremes():
    """Guards a real defect: comparing the earliest camera clip to the earliest
    phone clip measures the gap between two unrelated days. On the fixture
    corpus that produced -169239 s as the value offered to the operator."""
    cam = [Clip("c_early", "camera", _t("2023-10-15T06:00:00"), 60, "a"),
           Clip("c_pair", "camera", _t("2023-10-19T02:00:00"), 60, "b")]
    ph = [Clip("p_pair", "phone_keller", _t("2023-10-19T02:00:39"), 60, "c")]
    pairs = find_candidate_pairs(cam, ph, window_s=600)
    got = naive_offset(cam, ph, pairs)
    assert got == pytest.approx(-39.0), "must reflect the overlapping pair"
    assert abs(got) < 120, "must not span days"


def test_naive_prefill_without_pairs_uses_nearest_clips():
    cam = [Clip("c1", "camera", _t("2023-10-15T06:00:00"), 60, "a"),
           Clip("c2", "camera", _t("2023-10-19T02:00:00"), 60, "b")]
    ph = [Clip("p1", "phone_keller", _t("2023-10-19T02:00:39"), 60, "c")]
    assert naive_offset(cam, ph, None) == pytest.approx(-39.0)


def test_naive_prefill_median_ignores_one_odd_pair():
    cam = [Clip(f"c{i}", "camera", _t(f"2023-10-1{5+i}T02:00:00"), 60, "a") for i in range(3)]
    ph = [Clip(f"p{i}", "phone_keller", _t(f"2023-10-1{5+i}T02:00:39"), 60, "c") for i in range(3)]
    pairs = find_candidate_pairs(cam, ph, window_s=600)
    assert naive_offset(cam, ph, pairs) == pytest.approx(-39.0)


def test_naive_prefill_empty_is_zero():
    assert naive_offset([], [], []) == 0.0
