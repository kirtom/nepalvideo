import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from nepal.probe.fov import (seam_discontinuity, texture_score, solve_from_scores,
                             sample_timestamps, pick_textured_frames, candidate_fovs)

rng = np.random.default_rng(7)


def textured(h=256, w=1024, corr=8):
    """Smooth low-frequency texture -- stands in for landscape.

    ``corr`` is the spatial correlation length in pixels: small values look
    like scree, large values like sky and distant ridgelines.
    """
    from scipy.ndimage import zoom
    base = rng.normal(0, 1, (max(2, h // corr), max(2, w // corr)))
    img = zoom(base, corr, order=3)[:h, :w]
    img = (img - img.min()) / (np.ptp(img) + 1e-9) * 255
    return img.astype(np.float32)


def inject_seam(img, offset_px):
    """Shift content on one side of each seam -- what a wrong FOV looks like."""
    out = img.copy()
    w = img.shape[1]
    for seam_x in (w // 4, 3 * w // 4):
        out[:, seam_x:] = np.roll(out[:, seam_x:], offset_px, axis=1)
    return out


def test_clean_image_scores_near_one():
    img = textured()
    assert seam_discontinuity(img) < 2.0  # unmodified texture has no seam


def test_seam_raises_the_score():
    img = textured()
    clean = seam_discontinuity(img)
    broken = seam_discontinuity(inject_seam(img, 12))
    assert broken > clean * 1.5, f"clean={clean:.3f} broken={broken:.3f}"


def test_zero_misalignment_is_the_argmin():
    """The property the solver depends on: the correct alignment scores lowest.

    Not global monotonicity -- on a smooth field the seam gradient saturates
    once the offset passes the texture correlation length, so score(16px) may
    sit below score(4px). Only the location of the minimum matters to argmin.
    """
    img = textured()
    offsets = [0, 1, 2, 4, 8, 16, 32]
    scores = {k: seam_discontinuity(inject_seam(img, k)) for k in offsets}
    assert min(scores, key=scores.get) == 0, scores


@pytest.mark.parametrize("corr", [4, 8, 32, 64])
def test_aligned_seam_wins_at_every_texture_scale(corr):
    """Correct alignment must be the minimum regardless of what is in frame."""
    img = textured(corr=corr)
    aligned = seam_discontinuity(inject_seam(img, 0))
    others = [seam_discontinuity(inject_seam(img, k)) for k in (1, 2, 4, 8, 16, 32)]
    assert aligned < min(others), (aligned, others)


@pytest.mark.parametrize("corr", [8, 32, 64])
def test_coarse_texture_gives_a_wide_margin(corr):
    """Why S01.4 samples high-texture frames rather than arbitrary ones.

    Discrimination is a function of what is in frame. On fine texture (corr=4)
    the aligned/misaligned margin narrows to roughly 1.3x, which leaves
    ``confidence = 1 - best/second`` hovering near the 0.05 fallback
    threshold. On coarser structure it opens to 1.5x and beyond. That is the
    whole reason ``pick_textured_frames`` exists -- it is a precision lever,
    not a nicety.
    """
    img = textured(corr=corr)
    aligned = seam_discontinuity(inject_seam(img, 0))
    others = [seam_discontinuity(inject_seam(img, k)) for k in (1, 2, 4, 8, 16, 32)]
    assert aligned < 0.75 * min(others), (aligned, others)


def test_aligned_seam_scores_near_unity():
    """1.0 means the seam is statistically invisible against its surroundings."""
    for corr in (4, 8, 32, 64):
        assert 0.7 < seam_discontinuity(inject_seam(textured(corr=corr), 0)) < 1.3


def test_narrow_image_rejected():
    with pytest.raises(ValueError):
        seam_discontinuity(np.zeros((10, 50), dtype=np.float32))


def test_rgb_and_gray_agree():
    img = textured()
    rgb = np.repeat(img[:, :, None], 3, axis=2).astype(np.uint8)
    assert seam_discontinuity(rgb) == pytest.approx(seam_discontinuity(img), rel=0.05)


def test_texture_score_orders_flat_below_detailed():
    flat = np.full((128, 512), 128.0, dtype=np.float32)
    assert texture_score(flat) < texture_score(textured(128, 512, corr=8))


# -- solve_from_scores -------------------------------------------------

def test_argmin_is_chosen():
    frames = [{188: 3.0, 190: 2.0, 192: 1.0, 194: 2.5} for _ in range(5)]
    r = solve_from_scores(frames)
    assert r.fov_deg == 192
    assert r.confidence == pytest.approx(1 - 1.0 / 2.0, abs=1e-3)
    assert not r.used_fallback


def test_median_resists_one_outlier_frame():
    frames = [{192: 1.0, 194: 2.0} for _ in range(4)]
    frames.append({192: 99.0, 194: 0.1})   # one pathological frame
    r = solve_from_scores(frames)
    assert r.fov_deg == 192, "median across frames must survive an outlier"


def test_low_confidence_falls_back_to_193():
    frames = [{192: 1.00, 194: 1.01} for _ in range(5)]
    r = solve_from_scores(frames, min_confidence=0.05, fallback_deg=193)
    assert r.fov_deg == 193
    assert r.used_fallback
    assert "192" in r.method, "the rejected argmin must stay visible for Gate 1"


def test_no_frames_falls_back():
    r = solve_from_scores([])
    assert r.fov_deg == 193 and r.used_fallback and r.confidence == 0.0


# -- sampling ----------------------------------------------------------

def test_sample_timestamps_stay_interior():
    ts = sample_timestamps(100.0, 5)
    assert len(ts) == 5 and ts[0] >= 5.0 and ts[-1] <= 95.0


def test_sample_timestamps_zero_duration():
    assert sample_timestamps(0.0, 5) == []


def test_pick_spreads_across_files_before_taking_seconds():
    import pathlib as pl
    cands = []
    for i in range(10):
        f = pl.Path(f"clip{i}.lrv")
        for t in (10.0, 20.0, 30.0):
            cands.append((f, t, 100.0 - i))     # clip0 most textured
    picked = pick_textured_frames(cands, n_frames=20, min_distinct_files=8)
    assert len(picked) == 20
    distinct = {p for p, _ in picked}
    assert len(distinct) >= 8, f"only {len(distinct)} files used"
    # first pass must touch every file once before any file contributes twice
    first_five = [p for p, _ in picked[:5]]
    assert len(set(first_five)) == 5


def test_candidate_fovs_matches_spec_range():
    assert candidate_fovs(188, 208, 2) == [188, 190, 192, 194, 196, 198, 200, 202, 204, 206]
