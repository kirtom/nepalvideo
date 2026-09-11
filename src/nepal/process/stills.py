"""Photo shots -- the stills that earn a place in a film made of motion.

Roughly half the corpus is photographs, and until now none of it could reach
the cut: a timeline slot points at a shot, and every shot belonged to a video
recording. A trek is photographed as much as it is filmed, and Act 1 in
particular is planning material that was never filmed at all.

Stills are a seasoning. The film is motion; a held frame is punctuation, and
too many turn a documentary into a slideshow. So the share of the runtime they
may take is capped (``film.photo_share``) and only the best of them compete for
it -- see ``nepal.select.allocate_photo_budget``.

Scoring is deliberately narrow. Sharpness and exposure carry over from video
unchanged. Stability and motion do not apply to a still and are left NULL
rather than given a flattering default, because a photo would otherwise win
every stability comparison against real footage by virtue of not moving.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Sequence

import numpy as np

log = logging.getLogger(__name__)


def sharpness(img: np.ndarray) -> float:
    """Laplacian variance -- the same measure the FOV frame picker uses."""
    from scipy.ndimage import laplace
    gray = img if img.ndim == 2 else img[..., :3].mean(axis=2)
    return float(np.var(laplace(gray.astype(np.float32))))


def exposure_penalty(img: np.ndarray, clip_lo: float = 8.0,
                     clip_hi: float = 247.0) -> float:
    """How much of the frame is lost to clipping, plus drift from mid-grey.

    0 is a well-exposed frame; 1 is unusable. Snow and sky make the highlight
    end the one that matters here -- a blown-out ridge carries no detail no
    matter how sharp the lens was.
    """
    gray = img if img.ndim == 2 else img[..., :3].mean(axis=2)
    g = gray.astype(np.float32)
    if g.size == 0:
        return 1.0
    clipped = float(((g <= clip_lo) | (g >= clip_hi)).mean())
    drift = abs(float(g.mean()) - 128.0) / 128.0
    return float(min(1.0, clipped + 0.5 * drift))


def slot_duration_s(aspect: float | None = None, *, base_s: float = 3.0,
                    min_s: float = 2.0, max_s: float = 4.5) -> float:
    """How long a still is held.

    Long enough to read, short enough not to stall the cut. A panorama earns a
    little longer because the eye has further to travel across it; S06 snaps
    the value to the beat grid afterwards, so this is a preference rather than
    a final duration.
    """
    if aspect and aspect > 2.0:
        base_s *= 1.25
    return float(min(max_s, max(min_s, base_s)))


def technical_score(sharp: float, exposure_pen: float, *,
                    sharp_ref: float = 8.0) -> float:
    """One 0..1 number per still, for ranking stills against each other.

    Sharpness is unbounded above and heavy-tailed, so it is squashed against a
    reference rather than normalised across the set -- otherwise one macro shot
    of lichen rescales every landscape into looking soft.
    """
    s = math.tanh(max(0.0, sharp) / max(sharp_ref, 1e-6))
    return float(max(0.0, min(1.0, s * (1.0 - max(0.0, min(1.0, exposure_pen))))))


def passes_gate(sharp: float, exposure_pen: float, curve: dict[str, float]) -> bool:
    """The quality floor for the asset's source, reusing the video curves.

    A photo is held on screen for three seconds with nothing moving to distract
    from it, so it is judged at least as strictly as a frame of video.
    """
    return (sharp >= float(curve.get("min_sharpness", 0.0))
            and exposure_pen <= float(curve.get("max_exposure_pen", 1.0)))
