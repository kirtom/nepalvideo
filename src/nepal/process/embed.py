"""S04.1 -- one CLIP embedding per surviving shot.

The spec asks for "the sharpest sampled frame" of each shot, through CLIP
ViT-L, stored as a single ``.npy`` with a parallel shot-id index. It also asks
for this artefact to outlive the film: it is a searchable index of a personal
video library, not a scratch file, which is why the index is written as JSON
beside the array rather than implied by row order in a database.

The sharpest frame is the right representative and it is nearly free: S03.3
already decided which points in a shot to sample, and sharpness is a variance
of a Laplacian, so choosing among the candidates costs one decode each and no
model time. A frame chosen at random would embed a motion blur as readily as
the moment the shot is about.

Model loading and encoding are the only impure parts; frame choice is testable
without torch.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

log = logging.getLogger(__name__)

CLIP_MODEL = "ViT-L-14"
CLIP_PRETRAINED = "openai"


def sharpest(frames: Sequence[np.ndarray]) -> int:
    """Index of the sharpest frame, by variance of the Laplacian.

    The same measure S03.3 uses, so "sharp" means one thing across the
    pipeline. Frames that failed to decode are skipped rather than scored as
    zero, which would make an unreadable frame look merely soft.
    """
    import cv2
    best, best_score = -1, -1.0
    for i, f in enumerate(frames):
        if f is None or not getattr(f, "size", 0):
            continue
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
        score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if score > best_score:
            best, best_score = i, score
    return best


def load_model(name: str = CLIP_MODEL, pretrained: str = CLIP_PRETRAINED,
               *, gpu: bool = False):
    """``(model, preprocess, device)`` ready to encode images.

    CPU is a first-class path here. The account this ran on had no GPU quota
    at all, and ViT-L on an 8-core box embeds this corpus in minutes.
    """
    import open_clip
    import torch
    device = "cuda" if (gpu and torch.cuda.is_available()) else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        name, pretrained=pretrained, device=device)
    model.eval()
    return model, preprocess, device


def encode(model, preprocess, device, images: Sequence[np.ndarray]) -> np.ndarray:
    """Normalised embeddings, one row per image, in input order.

    Normalised on the way out so that similarity is a dot product everywhere
    downstream -- the same convention the face embeddings use.
    """
    import cv2
    import torch
    from PIL import Image
    if not len(images):
        return np.zeros((0, 0), dtype=np.float32)
    batch = torch.stack([
        preprocess(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
        for im in images]).to(device)
    with torch.no_grad():
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype(np.float32)


def search(embeddings: np.ndarray, query: np.ndarray, *, top: int = 10
           ) -> list[tuple[int, float]]:
    """``(row, similarity)`` for the closest rows, best first.

    Here because the artefact is meant to outlive the film: the index is only
    useful if something can query it, and the query is one dot product.
    """
    if not len(embeddings):
        return []
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(q))
    if n:
        q = q / n
    sims = embeddings @ q
    order = np.argsort(-sims)[:top]
    return [(int(i), float(sims[i])) for i in order]
