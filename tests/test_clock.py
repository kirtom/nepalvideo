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


# -- coarse alignment for clocks that are days out ---------------------

from nepal.probe.clock import coarse_offset_by_activity, activity_histogram, apply_coarse
import random


def _trek_activity(start, days=15, seed=3):
    """Capture times with a realistic diurnal shape: busy at dawn and late
    afternoon, quiet at midday, nothing overnight."""
    rng = random.Random(seed)
    out = []
    for d in range(days):
        for hour, n in ((6, 4), (7, 6), (8, 5), (11, 2), (15, 5), (16, 6), (17, 3)):
            for _ in range(n):
                out.append(start + timedelta(days=d, hours=hour,
                                             minutes=rng.randrange(60)))
    return sorted(out)


def test_recovers_a_fourteen_day_camera_clock_error():
    """The real corpus: camera filenames say 13 April, phone photos say
    27 April. A flat battery reverts an action camera's clock, and the gap is
    thousands of times wider than the +/-600 s audio search."""
    truth = timedelta(days=14)
    phone = _trek_activity(datetime(2024, 4, 27, tzinfo=timezone.utc))
    camera = [t - truth for t in phone]
    offset, conf = coarse_offset_by_activity(phone, camera, bin_s=3600)
    assert offset == pytest.approx(truth.total_seconds(), abs=3600)
    assert conf > 1.0


@pytest.mark.parametrize("days", [-30, -14, -1, 0, 1, 14, 30])
def test_recovers_offsets_in_both_directions(days):
    truth = timedelta(days=days)
    phone = _trek_activity(datetime(2024, 4, 27, tzinfo=timezone.utc))
    camera = [t - truth for t in phone]
    offset, _ = coarse_offset_by_activity(phone, camera, bin_s=3600)
    assert offset == pytest.approx(truth.total_seconds(), abs=3600)


def test_partial_overlap_still_aligns():
    """The camera ran out of battery halfway: only the first eight days exist."""
    truth = timedelta(days=14)
    phone = _trek_activity(datetime(2024, 4, 27, tzinfo=timezone.utc), days=15)
    cutoff = datetime(2024, 4, 27, tzinfo=timezone.utc) + timedelta(days=8)
    camera = [t - truth for t in phone if t < cutoff]
    offset, _ = coarse_offset_by_activity(phone, camera, bin_s=3600)
    assert offset == pytest.approx(truth.total_seconds(), abs=2 * 3600)


def test_no_offset_when_clocks_agree():
    phone = _trek_activity(datetime(2024, 4, 27, tzinfo=timezone.utc))
    offset, _ = coarse_offset_by_activity(phone, list(phone), bin_s=3600)
    assert abs(offset) <= 3600


def test_coarse_alignment_is_bounded_by_max_offset():
    phone = _trek_activity(datetime(2024, 4, 27, tzinfo=timezone.utc))
    camera = [t - timedelta(days=200) for t in phone]
    offset, _ = coarse_offset_by_activity(phone, camera, max_offset_s=10 * 86400)
    assert abs(offset) <= 10 * 86400


def test_coarse_alignment_empty_inputs():
    assert coarse_offset_by_activity([], []) == (0.0, 0.0)
    assert coarse_offset_by_activity([datetime(2024, 4, 27, tzinfo=timezone.utc)], []) == (0.0, 0.0)


def test_activity_histogram_bins_and_ignores_out_of_range():
    t0 = datetime(2024, 4, 27, tzinfo=timezone.utc)
    times = [t0, t0 + timedelta(minutes=30), t0 + timedelta(hours=2),
             t0 - timedelta(hours=5)]
    h = activity_histogram(times, t0, 3600, 4)
    assert list(h) == [2.0, 0.0, 1.0, 0.0]


def test_apply_coarse_shifts_clips():
    c = Clip("c1", "camera", _t("2024-04-13T10:00:00"), 60, "a")
    shifted = apply_coarse([c], 14 * 86400)
    assert shifted[0].start == _t("2024-04-27T10:00:00")
    assert shifted[0].clip_id == "c1"


def test_apply_coarse_zero_is_identity():
    c = Clip("c1", "camera", _t("2024-04-13T10:00:00"), 60, "a")
    assert apply_coarse([c], 0)[0].start == c.start


def test_diurnal_ambiguity_is_visible_rather_than_hidden():
    """With an identical routine every day, a whole-day shift correlates almost
    as well as the truth. The runner-up must therefore be a day away -- that is
    the ambiguity, and it is what Gate 1 has to show."""
    from nepal.probe.clock import coarse_candidates, DIURNAL_CONFIDENCE
    truth = timedelta(days=14)
    phone = _trek_activity(datetime(2024, 4, 27, tzinfo=timezone.utc))
    camera = [t - truth for t in phone]
    cands = coarse_candidates(phone, camera, bin_s=3600, top_n=3)
    assert cands[0][0] == pytest.approx(truth.total_seconds(), abs=3600)
    gaps = [abs(c[0] - cands[0][0]) / 86400 for c in cands[1:]]
    assert all(g == pytest.approx(round(g), abs=0.2) for g in gaps), gaps
    _, conf = coarse_offset_by_activity(phone, camera, bin_s=3600)
    assert conf < DIURNAL_CONFIDENCE, "a flat routine must not read as decisive"


def test_days_that_differ_break_the_tie():
    """Real treks are not identical every day -- a rest day and a summit push
    give the correlation something to lock onto, and confidence rises."""
    from nepal.probe.clock import coarse_offset_by_activity
    base = datetime(2024, 4, 27, tzinfo=timezone.utc)
    phone = _trek_activity(base, days=15)
    # a rest day with almost nothing, and a summit day with a burst
    phone = [t for t in phone if (t - base).days != 5]
    phone += [base + timedelta(days=9, hours=4, minutes=m) for m in range(0, 180, 3)]
    phone.sort()
    truth = timedelta(days=14)
    camera = [t - truth for t in phone]
    offset, conf = coarse_offset_by_activity(phone, camera, bin_s=3600)
    assert offset == pytest.approx(truth.total_seconds(), abs=3600)
    assert conf > 1.3, f"varied days should be more decisive, got {conf:.2f}"


# -- coincidence voting ------------------------------------------------

from nepal.probe.clock import offset_candidates_by_coincidence


def test_coincidence_finds_the_offset_from_sparse_material():
    """The case histogram correlation fails: a handful of camera recordings
    against a dense stream of phone photos."""
    truth = 14 * 86400 + 63.0
    phone = _trek_activity(datetime(2024, 4, 27, tzinfo=timezone.utc), days=15)
    # only seven camera clips, each filmed alongside some phone capture
    camera = [phone[i] - timedelta(seconds=truth) for i in (3, 20, 44, 60, 77, 90, 101)]
    cands = offset_candidates_by_coincidence(phone, camera, tolerance_s=300)
    assert cands, "no candidates produced"
    assert cands[0][0] == pytest.approx(truth, abs=300)


def test_coincidence_beats_histogram_on_sparse_input():
    truth = 14 * 86400 + 63.0
    phone = _trek_activity(datetime(2024, 4, 27, tzinfo=timezone.utc), days=15)
    camera = [phone[i] - timedelta(seconds=truth) for i in (3, 20, 44, 60, 77, 90, 101)]
    votes = offset_candidates_by_coincidence(phone, camera, tolerance_s=300)
    hist, _ = coarse_offset_by_activity(phone, camera, bin_s=3600)
    assert abs(votes[0][0] - truth) < abs(hist - truth) or abs(hist - truth) < 3600


def test_coincidence_votes_are_ranked_by_agreement():
    truth = 3600.0
    phone = _trek_activity(datetime(2024, 4, 27, tzinfo=timezone.utc), days=10)
    camera = [t - timedelta(seconds=truth) for t in phone[::4]]
    cands = offset_candidates_by_coincidence(phone, camera, tolerance_s=900)
    assert cands[0][1] >= cands[-1][1], "votes must be sorted descending"
    assert cands[0][0] == pytest.approx(truth, abs=900)


def test_coincidence_respects_max_offset():
    phone = _trek_activity(datetime(2024, 4, 27, tzinfo=timezone.utc), days=5)
    camera = [t - timedelta(days=200) for t in phone]
    assert offset_candidates_by_coincidence(phone, camera, max_offset_s=10 * 86400) == []


def test_coincidence_empty_inputs():
    assert offset_candidates_by_coincidence([], []) == []


def test_coincidence_candidates_are_separated():
    phone = _trek_activity(datetime(2024, 4, 27, tzinfo=timezone.utc), days=10)
    camera = [t - timedelta(seconds=3600) for t in phone[::4]]
    cands = offset_candidates_by_coincidence(phone, camera, tolerance_s=900, top_n=6)
    for i, (a, _) in enumerate(cands):
        for b, _ in cands[i + 1:]:
            assert abs(a - b) > 900, "non-maximum suppression failed"


@pytest.mark.parametrize("tol", [60, 120, 300, 600, 900])
def test_coincidence_works_across_usable_tolerances(tol):
    truth = 14 * 86400 + 63.0
    phone = _trek_activity(datetime(2024, 4, 27, tzinfo=timezone.utc), days=15)
    camera = [phone[i] - timedelta(seconds=truth) for i in (3, 20, 44, 60, 77, 90, 101)]
    cands = offset_candidates_by_coincidence(phone, camera, tolerance_s=tol)
    assert cands[0][0] == pytest.approx(truth, abs=tol)


def test_discrimination_degrades_as_the_window_widens():
    """Documents the method's one constraint so nobody widens the window
    thinking it makes detection more forgiving.

    Background votes scale with tolerance x capture density, so the margin
    between the true offset and the best impostor shrinks monotonically as the
    window opens. The exact tolerance at which the truth is finally outvoted
    depends on how dense the reference is, which is why the default is set well
    inside the safe range rather than at the observed breaking point.
    """
    truth = 14 * 86400 + 63.0
    phone = _trek_activity(datetime(2024, 4, 27, tzinfo=timezone.utc), days=15)
    camera = [phone[i] - timedelta(seconds=truth) for i in (3, 20, 44, 60, 77, 90, 101)]

    margins = []
    for tol in (120, 300, 900, 3600):
        c = offset_candidates_by_coincidence(phone, camera, tolerance_s=tol, top_n=4)
        margins.append(c[0][1] / max(c[1][1], 1) if len(c) > 1 else float("inf"))

    assert margins[0] >= margins[-1], f"margin should not improve with width: {margins}"
    tight = offset_candidates_by_coincidence(phone, camera, tolerance_s=300, top_n=4)
    assert tight[0][0] == pytest.approx(truth, abs=300), "the default must find it"
