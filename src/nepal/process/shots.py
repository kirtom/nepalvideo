"""S03.2 -- turn a proxy into shots.

Detection itself is a thin wrapper over PySceneDetect; everything that decides
what a shot *is* stays pure and testable. That split matters here because the
rules are where the judgement lives: how short a shot may be, what happens to
the tail of a recording, and what to do when a recording has no cuts at all.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

log = logging.getLogger(__name__)


def detect_scenes(proxy: "Any", *, threshold: float = 27.0,
                  min_len_s: float = 1.5) -> list[tuple[float, float]]:
    """Scene boundaries as (start_s, end_s), via PySceneDetect ContentDetector.

    Returns a single whole-file scene when nothing is detected, which is the
    right answer rather than an empty list: a locked-off two-minute shot of a
    ridge has no cuts and is still a shot.
    """
    from scenedetect import ContentDetector, SceneManager, open_video

    video = open_video(str(proxy))
    mgr = SceneManager()
    fps = float(video.frame_rate or 30.0)
    mgr.add_detector(ContentDetector(threshold=float(threshold),
                                     min_scene_len=max(1, int(min_len_s * fps))))
    mgr.detect_scenes(video)
    # .seconds on 0.6.3+, .get_seconds() before it
    def secs(t) -> float:
        return float(getattr(t, "seconds", None) if hasattr(t, "seconds")
                     else t.get_seconds())

    scenes = [(secs(s), secs(e)) for s, e in mgr.get_scene_list()]
    if not scenes:
        duration = secs(video.duration) if video.duration else 0.0
        return [(0.0, duration)] if duration > 0 else []
    return scenes


def merge_short_scenes(scenes: Sequence[tuple[float, float]],
                       min_len_s: float = 1.5) -> list[tuple[float, float]]:
    """Fold a scene shorter than ``min_len_s`` into its predecessor.

    PySceneDetect's own minimum is expressed in frames and is applied while
    detecting, so a rounding difference can still emit a 1.4 s scene. A shot
    below the minimum is not usable material -- it cannot be cut to a beat, and
    the quality gate would reject it anyway -- but discarding it would move the
    boundary and leave a hole, so it is absorbed instead.
    """
    out: list[list[float]] = []
    for start, end in scenes:
        if end - start <= 0:
            continue
        if out and (end - start) < min_len_s:
            out[-1][1] = end
        else:
            out.append([start, end])
    # a leading short scene has no predecessor, so it takes from its successor
    if len(out) > 1 and (out[0][1] - out[0][0]) < min_len_s:
        out[1][0] = out[0][0]
        out.pop(0)
    return [(round(a, 3), round(b, 3)) for a, b in out]


def shots_for_recording(scenes: Sequence[tuple[float, float]], recording_id: str,
                        *, min_len_s: float = 1.5) -> list[dict[str, Any]]:
    """Shot rows for one recording, numbered in order.

    The id carries the recording and the index so it is stable across re-runs:
    detection is deterministic, and a shot whose id moved would orphan every
    vote and score attached to it.
    """
    merged = merge_short_scenes(scenes, min_len_s)
    return [{
        "shot_id": f"{recording_id}#{i:04d}",
        "recording_id": recording_id,
        "asset_id": None,
        "media_kind": "video",
        "start_s": start,
        "end_s": end,
        "status": "candidate",
    } for i, (start, end) in enumerate(merged)]
