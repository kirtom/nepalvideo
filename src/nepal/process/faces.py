"""S03.6 -- who is in the shot.

InsightFace over rectilinear views, then one clustering pass over the whole
corpus so that "the same person" is a corpus-level fact rather than a
per-shot guess. The two largest clusters are the two trekkers; Gate 2
confirms which is which by showing one face from each.

**Faces are measured on rectilinear views, never on the equirect.** Measured
on this corpus: the same face detected both ways agrees at 0.87-0.96 cosine
most of the time, but one pair came back at 0.50 -- far enough apart that the
face would not cluster with itself. Equirect also tears a face in half at the
seam (a detection came back with a negative x). The projection is not free of
cost in principle, but it is here: the detector resizes everything to its own
640x640 input, so four yaw views cost 16.0 s against the equirect frame's
16.5 s. The spec's design is both more correct and no slower.

The remap and the clustering are pure; only ``detect`` touches the model.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

import numpy as np

log = logging.getLogger(__name__)

YAWS: tuple[int, ...] = (0, 90, 180, 270)
VIEW_FOV_DEG = 90.0
VIEW_SIZE = (640, 640)
# Cosine similarity above which two embeddings are the same person.
# ArcFace embeddings put different people near 0.1-0.3 and the same person
# above 0.5 or so; 0.42 is deliberately generous, because a face missed at
# clustering time is a shot that loses its subject, while a cluster that
# merges two strangers is visible at Gate 2 and fixable there.
SAME_PERSON_COS = 0.42
# A detection this weak is usually a rock, a rucksack buckle or a patch of
# lichen. Kept low: the corpus is faces at distance under hoods.
MIN_DET_SCORE = 0.55
# Cosine between two cluster *means* above which they are the same person.
# See merge_clusters: measured on this corpus, fragments of one person sit at
# 0.48-0.78 and different people at 0.00-0.12, so anything in the middle
# works and 0.40 is the middle.
MERGE_COS = 0.40


# -- projection --------------------------------------------------------

def rectilinear_map(eq_shape: tuple[int, int], yaw_deg: float, *,
                    fov_deg: float = VIEW_FOV_DEG,
                    out: tuple[int, int] = VIEW_SIZE
                    ) -> tuple[np.ndarray, np.ndarray]:
    """``(map_x, map_y)`` sampling an equirect frame as a pinhole camera.

    Built once per (shape, yaw) and reused across every frame of the corpus:
    the trigonometry is identical for every frame of the same size, and doing
    it per frame was most of the cost of a first draft.
    """
    h_eq, w_eq = eq_shape[:2]
    w, h = out
    f = (w / 2) / np.tan(np.radians(fov_deg) / 2)
    j, i = np.meshgrid(np.arange(w, dtype=np.float32),
                       np.arange(h, dtype=np.float32))
    x = j - w / 2
    y = i - h / 2
    z = np.full_like(x, f)
    norm = np.sqrt(x * x + y * y + z * z)
    x, y, z = x / norm, y / norm, z / norm
    a = np.radians(yaw_deg)
    xr = x * np.cos(a) + z * np.sin(a)
    zr = -x * np.sin(a) + z * np.cos(a)
    lon = np.arctan2(xr, zr)
    lat = np.arcsin(np.clip(y, -1.0, 1.0))
    map_x = ((lon / (2 * np.pi) + 0.5) * w_eq).astype(np.float32)
    map_y = ((lat / np.pi + 0.5) * h_eq).astype(np.float32)
    return map_x, map_y


class YawViews:
    """Cached remaps for one equirect frame size."""

    def __init__(self, eq_shape: tuple[int, int], yaws: Sequence[int] = YAWS,
                 *, fov_deg: float = VIEW_FOV_DEG,
                 out: tuple[int, int] = VIEW_SIZE) -> None:
        self.yaws = tuple(yaws)
        self._maps = {y: rectilinear_map(eq_shape, y, fov_deg=fov_deg, out=out)
                      for y in self.yaws}

    def render(self, frame: np.ndarray, yaw: int) -> np.ndarray:
        import cv2
        mx, my = self._maps[yaw]
        # BORDER_WRAP so a face on the 180 degree meridian is whole in the
        # view that looks at it, instead of torn the way equirect tears it.
        return cv2.remap(frame, mx, my, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_WRAP)


# -- the model ---------------------------------------------------------

def load_model(name: str = "buffalo_l", *, gpu: bool = False,
               det_size: tuple[int, int] = VIEW_SIZE):
    """InsightFace's bundled detector plus ArcFace recogniser.

    ONNX Runtime, so a GPU is optional. On this corpus the CPU path is what
    ran: the account had no GPU quota and an 8-core box was enough.
    """
    from insightface.app import FaceAnalysis
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu
                 else ["CPUExecutionProvider"])
    app = FaceAnalysis(name=name, providers=providers)
    app.prepare(ctx_id=0 if gpu else -1, det_size=det_size)
    return app


def detect(app, image: np.ndarray, *, min_score: float = MIN_DET_SCORE
           ) -> list[dict[str, Any]]:
    """Faces in one rectilinear view, weakest discarded."""
    out = []
    for f in app.get(image):
        score = float(f.det_score)
        if score < min_score:
            continue
        emb = np.asarray(f.embedding, dtype=np.float32)
        n = float(np.linalg.norm(emb))
        if n <= 0:
            continue
        out.append({"score": score, "embedding": emb / n,
                    "bbox": [int(v) for v in f.bbox]})
    return out


# -- clustering --------------------------------------------------------

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))          # embeddings are stored normalised


def cluster(embeddings: Sequence[np.ndarray], *,
            threshold: float = SAME_PERSON_COS) -> list[int]:
    """Greedy agglomeration against running cluster means.

    Not k-means: the number of people in the corpus is not known, and the
    answer the film needs is "the two largest clusters", which does not
    require the tail to be resolved correctly. Each face joins the nearest
    cluster whose mean it is close enough to, or starts its own.

    Returns one cluster index per embedding, in input order.
    """
    if not len(embeddings):
        return []
    means: list[np.ndarray] = []
    counts: list[int] = []
    labels: list[int] = []
    for emb in embeddings:
        e = np.asarray(emb, dtype=np.float32)
        best, best_sim = -1, threshold
        for i, m in enumerate(means):
            sim = cosine(e, m)
            if sim >= best_sim:
                best, best_sim = i, sim
        if best < 0:
            means.append(e.copy())
            counts.append(1)
            labels.append(len(means) - 1)
        else:
            n = counts[best]
            mean = (means[best] * n + e) / (n + 1)
            norm = float(np.linalg.norm(mean))
            means[best] = mean / norm if norm else mean
            counts[best] = n + 1
            labels.append(best)
    return labels


def merge_clusters(embeddings: Sequence[np.ndarray], labels: Sequence[int], *,
                   merge_cos: float = MERGE_COS) -> list[int]:
    """Merge clusters whose means are the same person, repeatedly.

    :func:`cluster` is a single greedy pass against running means, and that
    fails in one specific way on this corpus: once a cluster's mean has drifted
    toward the frontal views it accumulated first, a later profile of the same
    person no longer matches it and starts a cluster of its own. Measured on
    the real embeddings, the two trekkers had split into five clusters --
    keller across 376 + 65 + 25, kulikov across 200 + 33 -- whose means sat at
    0.48 to 0.78 cosine from one another while genuine strangers sat at 0.00
    to 0.12. That gap is what this pass closes, and it is wide enough that the
    threshold is not delicate: anything from 0.35 to 0.45 gives the same
    answer.

    Lowering the *detection* threshold instead does not work; it was tried.
    The fragments survive because the failure is in the order faces arrive,
    not in how similar they are.

    Average linkage over means, closest pair first, until nothing is close
    enough. Cheap: a few hundred clusters, not a few thousand faces.
    """
    if not len(embeddings):
        return []
    E = np.asarray(embeddings, dtype=np.float32)
    groups: dict[int, list[int]] = {}
    for i, l in enumerate(labels):
        groups.setdefault(int(l), []).append(i)

    def mean_of(idxs: list[int]) -> np.ndarray:
        m = E[idxs].mean(axis=0)
        n = float(np.linalg.norm(m))
        return m / n if n else m

    keys = list(groups)
    M = np.vstack([mean_of(groups[k]) for k in keys])      # one matrix, not a dict
    alive = np.ones(len(keys), dtype=bool)
    # The whole pairwise similarity at once. Looping this in Python was 2M dot
    # products on the real corpus -- fast enough on a server to go unnoticed
    # and slow enough on the operator's machine to be killed mid-run.
    sim = M @ M.T
    np.fill_diagonal(sim, -np.inf)
    while True:
        sim_alive = np.where(alive[:, None] & alive[None, :], sim, -np.inf)
        flat = int(np.argmax(sim_alive))
        i, j = divmod(flat, len(keys))
        if sim_alive[i, j] < merge_cos:
            break
        a, b = (i, j) if len(groups[keys[i]]) >= len(groups[keys[j]]) else (j, i)
        groups[keys[a]].extend(groups[keys[b]])
        del groups[keys[b]]
        alive[b] = False
        M[a] = mean_of(groups[keys[a]])
        row = M @ M[a]
        sim[a, :] = row
        sim[:, a] = row
        sim[a, a] = -np.inf

    out = [0] * len(labels)
    for k, idxs in groups.items():
        for i in idxs:
            out[i] = k
    return out


def name_clusters(labels: Sequence[int], names: Sequence[str] = ("keller", "kulikov")
                  ) -> dict[int, str]:
    """Label the largest clusters, biggest first.

    Which of the two is which is a human judgement -- the spec puts it at
    Gate 2 with one representative face per cluster -- so this only fixes the
    ordering. Everything past the named clusters is "other": a porter, a
    guide, a stranger in a teahouse, all of which the film may still use.
    """
    counts: dict[int, int] = {}
    for c in labels:
        counts[c] = counts.get(c, 0) + 1
    ranked = sorted(counts, key=lambda c: (-counts[c], c))
    return {c: names[i] for i, c in enumerate(ranked[:len(names)])}
