"""S03.3 -- technical metrics per shot.

Four numbers decide whether a shot is worth a human looking at it: is it sharp,
is it exposed, how much is moving, and is the camera shaking. They are measured
on the proxy, at five points inside each shot, which is the whole of the spec's
S03.3 and costs minutes against S03.1's hours.

Two choices here are not the obvious ones:

*Jerk, not motion, is what shake means.* The spec asks for stability as
``1/(1+var(motion vectors))``. Taken literally over frames seconds apart that
measures how much the camera moved, and on this corpus almost everything moved
-- the camera was on a pole, on a walking person, for four hours. What separates
a usable walking shot from an unusable one is not the motion but its second
derivative: a steady pan has a large, constant flow field, a shaken camera has a
flow field that changes direction between one frame and the next. So each sample
reads three consecutive frames and compares the two flow fields they give.

*The poles of an equirect frame are not evidence.* A 2:1 equirectangular proxy
stretches the top and bottom rows across the full width, so sky and the
photographer's own feet occupy half the pixels and blur every sharpness
measurement toward the same value. Sharpness and exposure are read from the
central band, which is where the horizon -- and the film -- is.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

log = logging.getLogger(__name__)

SAMPLES_PER_SHOT = 5
FRAMES_PER_SAMPLE = 3
FLOW_WIDTH = 320
# Jerk, in pixels per frame squared at FLOW_WIDTH, at which a shot is judged
# half stable. A locked-off camera sits near zero; a hand-held phone while
# walking runs a few pixels. Calibration point, not a law of nature -- S03.3
# reports the distribution it measured so this can be moved against real
# material rather than against intuition.
JERK_REF_PX = 4.0
# The band of an equirect frame that carries the horizon.
EQUIRECT_BAND = (0.25, 0.75)


def sample_times(start_s: float, end_s: float,
                 n: int = SAMPLES_PER_SHOT) -> list[float]:
    """``n`` points spread through the shot, none of them on a cut.

    Offset by half a step at each end: the first and last frames of a shot are
    the ones most likely to be a dissolve, a rolling-shutter smear or the tail
    of whatever the camera was doing before the scene detector called it.
    """
    span = max(0.0, float(end_s) - float(start_s))
    n = max(1, int(n))
    return [float(start_s) + span * (i + 0.5) / n for i in range(n)]


def central_band(gray: np.ndarray, band: tuple[float, float] = EQUIRECT_BAND) -> np.ndarray:
    """The horizon band of an equirect frame; the whole frame if it is small."""
    h = gray.shape[0]
    lo, hi = int(h * band[0]), int(h * band[1])
    return gray[lo:hi] if hi - lo >= 8 else gray


def to_gray(frame: np.ndarray) -> np.ndarray:
    return frame if frame.ndim == 2 else frame[..., :3].mean(axis=2)


def sharpness(gray: np.ndarray) -> float:
    """``log(var(Laplacian))`` -- the spec's measure, and the curves' units.

    Logged because Laplacian variance is heavy-tailed over two orders of
    magnitude: one frame of lichen at arm's length would otherwise sit a
    thousand times above a ridge line and make every landscape look soft.
    """
    from scipy.ndimage import laplace
    if gray.size == 0:
        return 0.0
    var = float(np.var(laplace(gray.astype(np.float32))))
    return float(max(0.0, math.log(var))) if var > 1.0 else 0.0


def exposure_penalty(gray: np.ndarray, clip_lo: float = 5.0,
                     clip_hi: float = 250.0) -> float:
    """Fraction of the frame lost to clipping (spec S03.3).

    Snow and sky make the highlight end the one that matters: a blown-out ridge
    carries no detail however sharp the lens was.
    """
    g = gray.astype(np.float32)
    if g.size == 0:
        return 1.0
    return float(((g <= clip_lo) | (g >= clip_hi)).mean())


def _resize(gray: np.ndarray, width: int) -> np.ndarray:
    h, w = gray.shape[:2]
    if w <= width:
        return gray.astype(np.float32)
    step = max(1, int(round(w / width)))
    return gray[::step, ::step].astype(np.float32)


def flow(a: np.ndarray, b: np.ndarray, *, width: int = FLOW_WIDTH) -> np.ndarray:
    """Farneback dense flow between two frames, at reduced width.

    Cost is per-pixel and shake is a low-frequency signal, so measuring it at
    proxy resolution buys nothing but time.
    """
    import cv2
    fa, fb = _resize(to_gray(a), width), _resize(to_gray(b), width)
    return cv2.calcOpticalFlowFarneback(
        fa.astype(np.uint8), fb.astype(np.uint8), None,
        0.5, 3, 15, 3, 5, 1.2, 0)


def motion_of(frames: Sequence[np.ndarray], *, width: int = FLOW_WIDTH
              ) -> tuple[float, float]:
    """``(mean |flow|, jerk)`` for one sample point.

    ``jerk`` is the length of the change in the mean flow vector between the
    two consecutive pairs -- how much the camera's motion changed within a
    fifteenth of a second. With fewer than three frames it cannot be measured
    and is reported as 0.0, which is honest: nothing was observed.
    """
    if len(frames) < 2:
        return 0.0, 0.0
    fields = [flow(frames[i], frames[i + 1], width=width)
              for i in range(len(frames) - 1)]
    mags = [float(np.mean(np.hypot(f[..., 0], f[..., 1]))) for f in fields]
    means = [(float(np.mean(f[..., 0])), float(np.mean(f[..., 1]))) for f in fields]
    jerk = 0.0
    if len(means) >= 2:
        jerk = max(math.hypot(means[i + 1][0] - means[i][0],
                              means[i + 1][1] - means[i][1])
                   for i in range(len(means) - 1))
    return float(np.median(mags)), float(jerk)


def stability_from_jerk(jerk: float, *, ref: float = JERK_REF_PX) -> float:
    """0..1, where 1 is a camera that did not shake."""
    return float(1.0 / (1.0 + max(0.0, jerk) / max(ref, 1e-6)))


def measure_samples(samples: Iterable[Sequence[np.ndarray]], *,
                    equirect: bool = False, width: int = FLOW_WIDTH,
                    jerk_ref: float = JERK_REF_PX) -> dict[str, Any] | None:
    """Fold per-sample frame groups into the four metrics for one shot.

    Medians, not means: one frame of lens flare or one footstep should not
    decide a shot, and five samples is too few for anything cleverer.
    """
    sharps: list[float] = []
    pens: list[float] = []
    mags: list[float] = []
    jerks: list[float] = []
    for frames in samples:
        frames = [f for f in frames if f is not None and getattr(f, "size", 0)]
        if not frames:
            continue
        gray = to_gray(frames[0])
        band = central_band(gray) if equirect else gray
        sharps.append(sharpness(band))
        pens.append(exposure_penalty(band))
        mag, jerk = motion_of(frames, width=width)
        mags.append(mag)
        jerks.append(jerk)
    if not sharps:
        return None
    jerk = float(np.median(jerks)) if jerks else 0.0
    return {
        "sharpness": round(float(np.median(sharps)), 4),
        "exposure_pen": round(float(np.median(pens)), 4),
        "motion_mag": round(float(np.median(mags)) if mags else 0.0, 4),
        "stability": round(stability_from_jerk(jerk, ref=jerk_ref), 4),
        "n_samples": len(sharps),
    }


def read_samples(path: Path, times: Sequence[float], *,
                 frames_per_sample: int = FRAMES_PER_SAMPLE,
                 cap: Any = None) -> list[list[np.ndarray]]:
    """Decode ``frames_per_sample`` consecutive frames at each of ``times``.

    The one place this module touches a video file. Seeking is by millisecond
    and the times arrive in order, so a proxy is opened once per recording and
    walked forward rather than re-opened per shot.
    """
    import cv2
    own = cap is None
    cap = cap if cap is not None else cv2.VideoCapture(str(path))
    if not cap.isOpened():
        if own:
            cap.release()
        raise OSError(f"cannot open {path}")
    out: list[list[np.ndarray]] = []
    try:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(t)) * 1000.0)
            group: list[np.ndarray] = []
            for _ in range(max(1, frames_per_sample)):
                ok, frame = cap.read()
                if not ok:
                    break
                group.append(frame)
            out.append(group)
    finally:
        if own:
            cap.release()
    return out


def percentiles(values: Sequence[float], qs=(5, 25, 50, 75, 95)) -> dict[str, float]:
    """The distribution behind a threshold.

    The spec expects the quality gate to reject roughly 70% of camera material.
    Whether it does is a fact about this corpus, not about the thresholds, and
    the only way to know before running the gate is to look.
    """
    vals = [v for v in values if v is not None]
    if not vals:
        return {}
    arr = np.asarray(vals, dtype=float)
    return {f"p{q}": round(float(np.percentile(arr, q)), 3) for q in qs}
