"""S04.1 -- frame choice and search, without torch."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from nepal.process import embed


def _sharp(h=64, w=64):
    """High-frequency checkerboard: a large Laplacian variance."""
    a = np.indices((h, w)).sum(axis=0) % 2
    return (a * 255).astype(np.uint8)


def _blurred():
    import cv2
    return cv2.GaussianBlur(_sharp(), (9, 9), 0)


def test_the_sharpest_frame_is_the_one_chosen():
    pytest.importorskip("cv2")
    frames = [_blurred(), _sharp(), _blurred()]
    assert embed.sharpest(frames) == 1


def test_a_frame_that_failed_to_decode_is_skipped_not_scored_zero():
    """None must not be treated as a very soft frame -- it is no frame."""
    pytest.importorskip("cv2")
    assert embed.sharpest([None, _blurred()]) == 1
    assert embed.sharpest([None, np.array([]), _sharp()]) == 2


def test_no_readable_frame_reports_no_choice():
    assert embed.sharpest([None, None]) == -1
    assert embed.sharpest([]) == -1


# -- search ------------------------------------------------------------

def _unit(seed, n=8):
    rng = np.random.default_rng(seed)
    e = rng.normal(size=(n, 16)).astype(np.float32)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def test_search_returns_the_nearest_rows_best_first():
    E = _unit(1)
    got = embed.search(E, E[3], top=3)
    assert got[0][0] == 3
    assert got[0][1] == pytest.approx(1.0, abs=1e-5)
    assert [s for _, s in got] == sorted([s for _, s in got], reverse=True)


def test_search_normalises_the_query_so_scale_does_not_matter():
    E = _unit(2)
    scaled = embed.search(E, E[5] * 17.0, top=1)
    plain = embed.search(E, E[5], top=1)
    assert [i for i, _ in scaled] == [i for i, _ in plain]
    assert scaled[0][1] == pytest.approx(plain[0][1], abs=1e-5)


def test_searching_an_empty_index_is_empty_not_an_error():
    assert embed.search(np.zeros((0, 16), np.float32), np.ones(16)) == []


@pytest.mark.slow
def test_clip_embeds_a_real_image_to_a_unit_vector():
    """At the real boundary: the model, not a stand-in. Two different images
    must not embed to the same place."""
    pytest.importorskip("open_clip")
    pytest.importorskip("torch")
    model, preprocess, device = embed.load_model()
    a = np.zeros((224, 224, 3), np.uint8); a[:, :, 2] = 255      # solid red
    b = np.stack([_sharp(224, 224)] * 3, axis=-1)                # checkerboard
    out = embed.encode(model, preprocess, device, [a, b])
    assert out.shape == (2, 768)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-4)
    assert float(out[0] @ out[1]) < 0.95, "different images, different vectors"
