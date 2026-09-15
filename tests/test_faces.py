"""S03.6 -- the projection and the clustering, without a model.

The detector itself is exercised at its real boundary in the slow test at the
bottom, which needs a face to point at.
"""
import os
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from nepal.process import faces


# -- projection --------------------------------------------------------

def test_the_centre_of_a_view_looks_where_the_yaw_says():
    """yaw 0 looks at the middle of the equirect; yaw 90 a quarter turn on.
    Getting this backwards yields views that look plausible and are rotated,
    which nothing downstream would catch."""
    h, w = 512, 1024
    for yaw, want_x in ((0, w / 2), (90, w * 0.75), (180, 0.0), (270, w * 0.25)):
        mx, my = faces.rectilinear_map((h, w), yaw, out=(64, 64))
        cx, cy = mx[32, 32], my[32, 32]
        assert cy == pytest.approx(h / 2, abs=1.0), yaw
        # 180 wraps to either edge of the equirect; both are the same meridian
        assert (cx == pytest.approx(want_x, abs=2.0)
                or cx == pytest.approx(want_x + w, abs=2.0)), (yaw, cx)


def test_the_map_stays_inside_the_frame():
    mx, my = faces.rectilinear_map((512, 1024), 33.0, out=(128, 128))
    assert 0.0 <= float(mx.min()) and float(mx.max()) <= 1024.0
    assert 0.0 <= float(my.min()) and float(my.max()) <= 512.0


def test_a_feature_at_a_known_bearing_lands_in_the_view_that_faces_it():
    """A bright column at 90 degrees must appear in the yaw-90 view and not
    in the yaw-0 one."""
    cv2 = pytest.importorskip("cv2")
    eq = np.zeros((256, 512, 3), np.uint8)
    eq[:, 380:390] = 255                       # 3/4 across = 90 degrees
    views = faces.YawViews((256, 512), out=(128, 128))
    assert views.render(eq, 90).mean() > views.render(eq, 0).mean() * 5


def test_views_are_built_once_and_reused():
    v = faces.YawViews((256, 512), out=(32, 32))
    assert set(v._maps) == {0, 90, 180, 270}
    assert v._maps[0][0].shape == (32, 32)


# -- clustering --------------------------------------------------------

def _person(seed: int, n: int, jitter: float = 0.05) -> list[np.ndarray]:
    """n embeddings of one person: a direction plus noise, normalised."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=512).astype(np.float32)
    base /= np.linalg.norm(base)
    out = []
    for _ in range(n):
        e = base + rng.normal(scale=jitter, size=512).astype(np.float32)
        out.append(e / np.linalg.norm(e))
    return out


def test_two_people_come_back_as_two_clusters():
    embs = _person(1, 5) + _person(2, 4)
    labels = faces.cluster(embs)
    assert len(set(labels)) == 2
    assert len(set(labels[:5])) == 1 and len(set(labels[5:])) == 1
    assert labels[0] != labels[5]


def test_the_same_face_twice_is_one_cluster_not_two():
    e = _person(3, 1)[0]
    assert faces.cluster([e, e.copy()]) == [0, 0]


def test_nothing_in_is_no_clusters_not_a_crash():
    assert faces.cluster([]) == []


def test_the_two_largest_clusters_get_the_names():
    labels = [0, 0, 0, 1, 1, 1, 1, 2]          # cluster 1 is the largest
    named = faces.name_clusters(labels)
    assert named == {1: "keller", 0: "kulikov"}
    assert 2 not in named, "a third person is 'other', not a trekker"


def test_naming_is_stable_when_two_clusters_tie():
    """A tie must not make the label depend on dict ordering: a run that
    relabels the two trekkers between invocations is worse than one that
    picks wrong, because Gate 2 only confirms the labelling once."""
    assert faces.name_clusters([0, 0, 1, 1]) == faces.name_clusters([1, 1, 0, 0])


# -- the real boundary -------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("NEPAL_FACE_IMAGE"),
                    reason="set NEPAL_FACE_IMAGE to a photo with a face in it")
def test_the_detector_finds_a_real_face_and_embeds_it():
    """No synthesiser makes a face the detector accepts, so the positive case
    runs against real material when pointed at some."""
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("insightface")
    img = cv2.imread(os.environ["NEPAL_FACE_IMAGE"])
    assert img is not None
    got = faces.detect(faces.load_model(), img)
    assert got, "expected at least one face"
    emb = got[0]["embedding"]
    assert emb.shape == (512,)
    assert float(np.linalg.norm(emb)) == pytest.approx(1.0, abs=1e-5)
    assert faces.cosine(emb, emb) == pytest.approx(1.0, abs=1e-5)
