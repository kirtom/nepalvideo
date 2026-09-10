"""S01.5 -- auto-solve per-phone clock offsets by audio cross-correlation.

Replaces "find a moment captured on both devices". The camera is the reference
clock; each phone gets one offset measured against it.

    created_at_utc = created_at + clock_offset_<device>_s

A wrong offset here misaligns every downstream join -- geotagging, message
anchoring, act assignment -- and yields a plausible-looking but wrong film.
The spec requires it be surfaced at Gate 1 regardless of confidence, so this
module reports its full working (per-pair lags, per-pair confidence, spread)
rather than a bare number.

GCC-PHAT rather than plain cross-correlation: whitening by magnitude makes the
peak depend on phase alignment alone, which survives two devices with very
different microphones, gain and codecs recording the same wind and voices.
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable, Sequence

import numpy as np

log = logging.getLogger(__name__)
EPS = 1e-12

# Below this, a coarse alignment is not decisive against a whole-day shift.
DIURNAL_CONFIDENCE = 1.35


def gcc_phat(a: np.ndarray, b: np.ndarray, fs: int,
             max_lag_s: float = 600.0,
             guard_samples: int | None = None) -> tuple[float, float]:
    """Generalised cross-correlation with phase transform.

    Returns ``(lag_s, confidence)`` where **a is b delayed by lag_s**: a
    positive lag means the sound arrives later in ``a`` than in ``b``.

    ``confidence`` is the ratio of the correlation peak to the next highest
    peak outside a guard window around it (spec: "peak / second_highest_local
    _peak"). 1.0 means the winner is no better than an also-ran; the spec
    accepts a pair at 1.5.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0:
        return 0.0, 0.0
    a = a - a.mean()
    b = b - b.mean()

    n = 1 << int(np.ceil(np.log2(a.size + b.size)))
    A = np.fft.rfft(a, n)
    B = np.fft.rfft(b, n)
    R = A * np.conj(B)
    R /= np.abs(R) + EPS                    # the phase transform
    cc = np.fft.irfft(R, n)

    max_shift = min(int(fs * max_lag_s), n // 2)
    # reorder to [-max_shift .. +max_shift]
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
    mag = np.abs(cc)

    peak_i = int(np.argmax(mag))
    peak = float(mag[peak_i])
    lag_s = (peak_i - max_shift) / float(fs)

    if guard_samples is None:
        guard_samples = max(8, int(fs * 0.05))   # 50 ms either side
    masked = mag.copy()
    lo = max(0, peak_i - guard_samples)
    hi = min(masked.size, peak_i + guard_samples + 1)
    masked[lo:hi] = 0.0
    runner_up = float(masked.max()) if masked.size else 0.0

    confidence = peak / (runner_up + EPS) if runner_up > 0 else float("inf")
    return float(lag_s), float(confidence)


def offset_from_pair(cam_seg_start: datetime, phone_seg_start: datetime,
                     lag_s: float) -> float:
    """Convert one measured lag into a phone clock offset, in seconds.

    The same real instant sits at segment-relative time t in the camera audio
    and t + lag in the phone audio. Solving for the correction that maps the
    phone's reported clock onto the camera's::

        phone_offset = (cam_seg_start - phone_seg_start) - lag
    """
    delta = (cam_seg_start - phone_seg_start).total_seconds()
    return delta - lag_s


@dataclass
class PairMeasurement:
    camera_id: str
    phone_id: str
    lag_s: float
    confidence: float
    offset_s: float
    accepted: bool = False
    note: str = ""


@dataclass
class ClockResult:
    device: str
    offset_s: float
    confidence: float
    method: str
    n_pairs_total: int = 0
    n_pairs_accepted: int = 0
    spread_s: float | None = None
    measurements: list[PairMeasurement] = field(default_factory=list)
    needs_manual: bool = False


def reduce_measurements(device: str, measurements: Sequence[PairMeasurement], *,
                        min_pair_confidence: float = 1.5,
                        min_confident_pairs: int = 3,
                        naive_offset_s: float = 0.0) -> ClockResult:
    """Median of confident pairs, per spec step 5.

    Median rather than mean: one pair that locked onto a wind gust instead of
    the real event lands hundreds of seconds out and would drag any mean with
    it. Below ``min_confident_pairs`` the spec requires confidence 0 and a
    manual field at Gate 1, pre-filled with the naive timestamp difference.
    """
    measurements = list(measurements)
    for m in measurements:
        m.accepted = m.confidence > min_pair_confidence
        if not m.accepted:
            m.note = f"confidence {m.confidence:.2f} <= {min_pair_confidence}"

    good = [m for m in measurements if m.accepted]

    if len(good) < min_confident_pairs:
        return ClockResult(
            device=device, offset_s=round(naive_offset_s, 3), confidence=0.0,
            method=(f"fallback:manual({len(good)} confident pairs, "
                    f"need {min_confident_pairs}) -- prefilled with naive delta"),
            n_pairs_total=len(measurements), n_pairs_accepted=len(good),
            measurements=measurements, needs_manual=True,
        )

    offsets = [m.offset_s for m in good]
    median = float(statistics.median(offsets))
    spread = float(max(offsets) - min(offsets))

    # Agreement between independent pairs is the real evidence the number is
    # right; the per-pair peak ratio only says the correlation was decisive.
    mad = float(statistics.median([abs(o - median) for o in offsets]))
    confidence = 1.0 / (1.0 + mad)

    return ClockResult(
        device=device, offset_s=round(median, 3), confidence=round(confidence, 4),
        method=f"gcc-phat:median({len(good)}/{len(measurements)} pairs, MAD={mad:.2f}s)",
        n_pairs_total=len(measurements), n_pairs_accepted=len(good),
        spread_s=round(spread, 3), measurements=measurements,
    )


# -- coarse alignment --------------------------------------------------

def activity_histogram(times: Sequence[datetime], t0: datetime, bin_s: float,
                       n_bins: int) -> "np.ndarray":
    """Capture counts per time bin, on an absolute grid starting at ``t0``."""
    h = np.zeros(n_bins, dtype=float)
    for t in times:
        i = int((t - t0).total_seconds() // bin_s)
        if 0 <= i < n_bins:
            h[i] += 1.0
    return h


def coarse_offset_by_activity(reference: Sequence[datetime], target: Sequence[datetime],
                              *, bin_s: float = 3600.0,
                              max_offset_s: float = 45 * 86400.0) -> tuple[float, float]:
    """Bulk clock error between two devices, for clocks that are days apart.

    GCC-PHAT searches +/- 600 s. An action camera whose battery went flat
    reverts its clock to the firmware epoch and can come back days or years
    out -- a gap thousands of times wider than that window, which the audio
    stage would simply never find.

    Both devices were carried by people doing the same thing at the same
    times: filming in the mornings, resting at midday, asleep at night. So
    their capture-activity histograms have the same shape, and the lag that
    best aligns them is the bulk clock error. Cross-correlating counts per
    hour recovers it without decoding a single frame.

    Returns ``(offset_s, confidence)`` where offset_s is what to ADD to the
    target's timestamps, and confidence is the peak-to-runner-up ratio.

    Note the one ambiguity this method cannot resolve on its own: trek activity
    is diurnal, so a whole-day shift correlates nearly as well as the truth and
    confidence stays modest (around 1.2) even when the answer is right. Days
    that differ from each other -- a rest day, the summit push, a travel day --
    are what break the tie, so real corpora score better than a synthetic
    one. A one-day error is beyond what the +/-600 s audio stage can repair,
    so anything below ``DIURNAL_CONFIDENCE`` must reach the operator at
    Gate 1 rather than being applied silently.
    """
    if not reference or not target:
        return 0.0, 0.0

    t0 = min(min(reference), min(target)) - timedelta(seconds=max_offset_s)
    t1 = max(max(reference), max(target)) + timedelta(seconds=max_offset_s)
    n_bins = int((t1 - t0).total_seconds() // bin_s) + 1
    if n_bins > 2_000_000:                     # guard a pathological span
        bin_s = (t1 - t0).total_seconds() / 500_000
        n_bins = int((t1 - t0).total_seconds() // bin_s) + 1

    ref_h = activity_histogram(reference, t0, bin_s, n_bins)
    tgt_h = activity_histogram(target, t0, bin_s, n_bins)
    if ref_h.sum() == 0 or tgt_h.sum() == 0:
        return 0.0, 0.0

    ref_h = (ref_h - ref_h.mean()) / (ref_h.std() + EPS)
    tgt_h = (tgt_h - tgt_h.mean()) / (tgt_h.std() + EPS)

    corr = np.correlate(ref_h, tgt_h, mode="full")
    lags = np.arange(-(len(tgt_h) - 1), len(ref_h))
    keep = np.abs(lags * bin_s) <= max_offset_s
    corr, lags = corr[keep], lags[keep]
    if corr.size == 0:
        return 0.0, 0.0

    peak_i = int(np.argmax(corr))
    peak = float(corr[peak_i])
    offset_s = float(lags[peak_i] * bin_s)

    guard = max(1, int(6 * 3600 / bin_s))      # ignore the peak's own shoulders
    masked = corr.copy()
    masked[max(0, peak_i - guard): peak_i + guard + 1] = -np.inf
    runner = float(masked.max()) if np.isfinite(masked).any() else 0.0
    confidence = peak / abs(runner) if runner not in (0.0, -np.inf) and abs(runner) > EPS \
        else float("inf")
    return offset_s, float(confidence)


def coarse_candidates(reference: Sequence[datetime], target: Sequence[datetime],
                      *, bin_s: float = 3600.0, max_offset_s: float = 45 * 86400.0,
                      top_n: int = 3) -> list[tuple[float, float]]:
    """The best few coarse alignments, as (offset_s, correlation).

    When the diurnal ambiguity bites, the runner-up is a whole-day shift of
    the winner. Showing the operator "14 days, or 13, or 15" at Gate 1 is a
    question they can answer in seconds from memory; showing one number they
    cannot check is how a wrong film gets made.
    """
    if not reference or not target:
        return []
    t0 = min(min(reference), min(target)) - timedelta(seconds=max_offset_s)
    t1 = max(max(reference), max(target)) + timedelta(seconds=max_offset_s)
    n_bins = int((t1 - t0).total_seconds() // bin_s) + 1
    ref_h = activity_histogram(reference, t0, bin_s, n_bins)
    tgt_h = activity_histogram(target, t0, bin_s, n_bins)
    if ref_h.sum() == 0 or tgt_h.sum() == 0:
        return []
    ref_h = (ref_h - ref_h.mean()) / (ref_h.std() + EPS)
    tgt_h = (tgt_h - tgt_h.mean()) / (tgt_h.std() + EPS)
    corr = np.correlate(ref_h, tgt_h, mode="full")
    lags = np.arange(-(len(tgt_h) - 1), len(ref_h))
    keep = np.abs(lags * bin_s) <= max_offset_s
    corr, lags = corr[keep], lags[keep]

    out: list[tuple[float, float]] = []
    work = corr.copy()
    guard = max(1, int(6 * 3600 / bin_s))
    for _ in range(top_n):
        if not np.isfinite(work).any():
            break
        i = int(np.argmax(work))
        out.append((float(lags[i] * bin_s), float(work[i])))
        work[max(0, i - guard): i + guard + 1] = -np.inf
    return out


def apply_coarse(clips: Sequence["Clip"], offset_s: float) -> list["Clip"]:
    """Shift a device's clips by a coarse offset, so the fine stage searches
    around the right place."""
    if not offset_s:
        return list(clips)
    shift = timedelta(seconds=offset_s)
    return [Clip(c.clip_id, c.device, c.start + shift, c.duration_s, c.path, c.has_audio)
            for c in clips]


# -- candidate pairing -------------------------------------------------

@dataclass
class Clip:
    clip_id: str
    device: str
    start: datetime
    duration_s: float
    path: str
    has_audio: bool = True

    @property
    def end(self) -> datetime:
        return self.start + timedelta(seconds=self.duration_s)


def find_candidate_pairs(camera: Sequence[Clip], phone: Sequence[Clip], *,
                         window_s: float = 600.0) -> list[tuple[Clip, Clip, float, float]]:
    """Pairs whose nominal times overlap within +/- ``window_s``.

    Returns (camera_clip, phone_clip, cam_offset_into_clip, phone_offset_into_clip)
    describing the region to extract from each. The offsets locate the
    overlapping span, so the extracted audio actually contains shared sound
    instead of two unrelated minutes that merely sat near each other.
    """
    pairs = []
    for c in camera:
        if not c.has_audio:
            continue
        for p in phone:
            if not p.has_audio:
                continue
            # widen each clip by the window, then intersect
            lo = max(c.start - timedelta(seconds=window_s), p.start - timedelta(seconds=window_s))
            hi = min(c.end + timedelta(seconds=window_s), p.end + timedelta(seconds=window_s))
            if hi <= lo:
                continue
            # the region we can actually read from both files
            ov_lo = max(c.start, p.start)
            ov_hi = min(c.end, p.end)
            if ov_hi > ov_lo:
                cam_off = (ov_lo - c.start).total_seconds()
                ph_off = (ov_lo - p.start).total_seconds()
            else:
                # no true overlap under nominal clocks, but within the search
                # window -- read from the start of each and let the +/-600 s lag
                # search find the alignment.
                cam_off, ph_off = 0.0, 0.0
            pairs.append((c, p, max(0.0, cam_off), max(0.0, ph_off)))
    return pairs


def naive_offset(camera: Sequence[Clip], phone: Sequence[Clip],
                 pairs: Sequence[tuple] | None = None) -> float:
    """Difference of device-reported timestamps -- the Gate 1 manual prefill.

    Taken over clips that plausibly recorded the same moment, not over the
    corpus as a whole. Comparing the globally earliest camera clip against the
    globally earliest phone clip measures the gap between two unrelated days
    and lands tens of thousands of seconds out, which is worse than useless as
    the value a tired operator is asked to confirm at 1 a.m.

    With candidate pairs available the median nominal delta is used; it differs from
    the true offset only by how long after the camera the phone started
    rolling, so it is right to within seconds. Otherwise the closest camera and
    phone clips in time are compared.
    """
    if pairs:
        deltas = [(c.start - p.start).total_seconds() for c, p, *_ in pairs]
        return float(statistics.median(deltas))
    if not camera or not phone:
        return 0.0
    best = min(((c, p) for c in camera for p in phone),
               key=lambda cp: abs((cp[0].start - cp[1].start).total_seconds()))
    return (best[0].start - best[1].start).total_seconds()
