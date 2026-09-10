"""S01.4 -- auto-solve the dual-fisheye field of view.

Replaces "eyeball the seam". In an equirectangular projection of width W, the
two fisheye hemispheres meet at x = W/4 and x = 3W/4. When ih_fov is wrong the
image content does not line up across those meridians, leaving a vertical
discontinuity. Measure it, sweep the candidate FOVs, take the argmin.

The scoring is deliberately scale-free: the seam gradient is divided by the
gradient of the surrounding image, so a frame full of rock texture and a frame
of flat sky both contribute a comparable number.

Pure functions here take arrays; ffmpeg lives in ``render_candidate``.
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

log = logging.getLogger(__name__)
EPS = 1e-6


def to_gray(img: np.ndarray) -> np.ndarray:
    """(H,W[,C]) uint8 -> (H,W) float32 luma."""
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 3:
        a = a[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return a


def horizontal_gradient(gray: np.ndarray) -> np.ndarray:
    """|dI/dx|, shape (H, W-1)."""
    return np.abs(np.diff(gray, axis=1))


def seam_discontinuity(img: np.ndarray, seam_band_px: int = 6,
                       local_band_px: int = 60) -> float:
    """Mean seam-to-local gradient ratio across both hemisphere meridians.

    1.0 means the seam is indistinguishable from its surroundings (ideal).
    Higher means a visible vertical join.
    """
    gray = to_gray(img)
    grad = horizontal_gradient(gray)
    w = grad.shape[1]
    if w < 2 * (seam_band_px + local_band_px):
        raise ValueError(f"image too narrow ({w + 1}px) for the requested bands")

    ratios = []
    for seam_x in (round(w * 0.25), round(w * 0.75)):
        half = seam_band_px // 2
        lo, hi = seam_x - half, seam_x + half + 1
        seam = grad[:, max(0, lo):min(w, hi)]
        left = grad[:, max(0, lo - local_band_px):max(0, lo)]
        right = grad[:, min(w, hi):min(w, hi + local_band_px)]
        local = np.concatenate([left, right], axis=1)
        if seam.size == 0 or local.size == 0:
            continue
        ratios.append(float(seam.mean()) / (float(local.mean()) + EPS))
    if not ratios:
        raise ValueError("no usable seam bands")
    return float(np.mean(ratios))


def texture_score(img: np.ndarray) -> float:
    """Laplacian variance -- used to prefer textured frames for sampling.

    scipy rather than OpenCV keeps Milestone 1 installable without a vision
    stack.
    """
    from scipy.ndimage import laplace
    gray = to_gray(img)
    return float(np.var(laplace(gray)))


@dataclass
class FovResult:
    fov_deg: float
    confidence: float
    method: str
    scores: dict[int, float]
    n_frames: int
    used_fallback: bool = False

    def as_decision(self) -> tuple[str, float, float, str]:
        return ("fov_deg", self.fov_deg, self.confidence, self.method)


def solve_from_scores(per_frame: Sequence[dict[int, float]], *,
                      min_confidence: float = 0.05,
                      fallback_deg: float = 193.0) -> FovResult:
    """Reduce per-frame per-FOV discontinuity scores to one FOV.

    ``per_frame`` is one dict {fov: discontinuity} per sampled frame. The
    median across frames is taken per FOV (step 4 of the spec), then argmin.
    Confidence is 1 - best/second_best: 0 when two candidates tie, approaching
    1 when the winner is far clearer than the runner-up.
    """
    if not per_frame:
        return FovResult(fallback_deg, 0.0, "fallback:no-frames", {}, 0, True)

    fovs = sorted({f for frame in per_frame for f in frame})
    medians: dict[int, float] = {}
    for f in fovs:
        vals = [frame[f] for frame in per_frame if f in frame]
        if vals:
            medians[f] = float(statistics.median(vals))
    if not medians:
        return FovResult(fallback_deg, 0.0, "fallback:no-scores", {}, len(per_frame), True)

    ranked = sorted(medians.items(), key=lambda kv: kv[1])
    best_fov, best = ranked[0]
    if len(ranked) < 2:
        return FovResult(float(best_fov), 0.0, "seam-min:single-candidate",
                         medians, len(per_frame), False)
    _, second = ranked[1]
    confidence = 1.0 - (best / (second + EPS))

    if confidence < min_confidence:
        # No clear winner. Spec: fall back and surface at Gate 1 with thumbnails.
        return FovResult(fallback_deg, round(confidence, 4),
                         f"fallback:low-confidence(argmin={best_fov})",
                         medians, len(per_frame), True)
    return FovResult(float(best_fov), round(confidence, 4), "seam-min",
                     medians, len(per_frame), False)


# -- orchestration (needs ffmpeg) --------------------------------------

def candidate_fovs(start: int, stop: int, step: int) -> list[int]:
    return list(range(start, stop, step))


def render_candidate(src: Path, t_s: float, fov: int, dest: Path,
                     width: int = 1024) -> Path:
    """Reproject one frame at one candidate FOV."""
    from nepal.util.proc import extract_frame
    vf = (f"v360=input=dfisheye:output=e:ih_fov={fov}:iv_fov={fov},"
          f"scale={width}:{width // 2}")
    return extract_frame(src, t_s, dest, vf=vf)


def load_image(path: Path) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def sample_timestamps(duration_s: float, n: int) -> list[float]:
    """Evenly spaced interior timestamps, avoiding the first/last 5%."""
    if duration_s <= 0 or n <= 0:
        return []
    lo, hi = duration_s * 0.05, duration_s * 0.95
    if n == 1:
        return [(lo + hi) / 2]
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def pick_textured_frames(candidates: Sequence[tuple[Path, float, float]],
                         n_frames: int, min_distinct_files: int) -> list[tuple[Path, float]]:
    """Choose sampling points, preferring texture but spreading across files.

    Spec S01.4 step 1: 20 frames from at least 8 different files, preferring
    frames whose Laplacian variance is above the median.

    The file-count floor outranks the texture preference. Filtering to
    above-median texture first can strand the sample on a handful of busy
    clips -- and a FOV solved from one clip's parallax is a FOV fitted to one
    subject distance. So frames are taken round-robin across every file
    (each file's sharpest first, files ordered by their best frame), in two
    passes: above-median frames first, then below-median only if the quota is
    still short.
    """
    if not candidates or n_frames <= 0:
        return []

    median_tex = statistics.median([c[2] for c in candidates])

    by_file: dict[Path, list[tuple[Path, float, float]]] = {}
    for c in candidates:
        by_file.setdefault(c[0], []).append(c)
    for v in by_file.values():
        v.sort(key=lambda c: -c[2])

    files = sorted(by_file, key=lambda p: -by_file[p][0][2])
    if len(files) < min_distinct_files:
        log.warning("FOV sampling: only %d distinct files available, spec wants %d",
                    len(files), min_distinct_files)

    picked: list[tuple[Path, float]] = []
    seen: set[tuple[Path, float]] = set()

    for above_median_only in (True, False):
        depth = 0
        while len(picked) < n_frames:
            added = False
            for f in files:
                if len(picked) >= n_frames:
                    break
                frames = by_file[f]
                if depth >= len(frames):
                    continue
                path, t_s, tex = frames[depth]
                if above_median_only and tex < median_tex:
                    continue
                key = (path, t_s)
                if key in seen:
                    continue
                picked.append(key)
                seen.add(key)
                added = True
            if not added:
                break
            depth += 1
        if len(picked) >= n_frames:
            break

    return picked[:n_frames]
