"""S03.5 -- transcribe the shots that carry speech.

Only shots with ``has_speech = 1`` (spec section S03.5), and only the shot's
own window of the 16 kHz track: the transcript belongs to the shot the
timeline chooses between, so a sentence that crosses a boundary is split
where the boundary is. The VAD's padding already keeps the phonemes at the
edges.

The model wrapper is thin; the joining and the bookkeeping are pure so they
can be tested without a model present.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable

import numpy as np

log = logging.getLogger(__name__)


def join_segments(segments: Iterable[Any], offset_s: float = 0.0) -> dict[str, Any]:
    """One transcript from faster-whisper's segments.

    Segment times are relative to the audio handed in, so ``offset_s`` puts
    them back on the recording's clock -- S06 cuts on the recording's clock.
    """
    parts: list[dict[str, Any]] = []
    words: list[str] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        parts.append({"start_s": round(float(seg.start) + offset_s, 3),
                      "end_s": round(float(seg.end) + offset_s, 3),
                      "text": text})
        words.append(text)
    return {"text": " ".join(words).strip(), "segments": parts}


def load_model(name: str, *, compute_type: str = "int8", cpu_threads: int = 0):
    """faster-whisper's model, on whatever device is there.

    int8 halves the memory of the fp16 weights and is what keeps large-v3
    inside 11 GB on the machine this was first run on.
    """
    from faster_whisper import WhisperModel
    kw: dict[str, Any] = {"device": "auto", "compute_type": compute_type}
    if cpu_threads:
        kw["cpu_threads"] = int(cpu_threads)
    return WhisperModel(name, **kw)


def transcribe_window(model, samples: np.ndarray, *, language: str | None,
                      beam_size: int, offset_s: float) -> dict[str, Any]:
    """Transcribe one shot's samples; times come back on the recording clock."""
    if samples.size == 0:
        return {"text": "", "segments": [], "language": language}
    segments, info = model.transcribe(
        samples, language=language or None, beam_size=int(beam_size),
        vad_filter=False,                 # S03.4 already decided this is speech
        condition_on_previous_text=False)  # a 20 s shot has no useful previous
    out = join_segments(segments, offset_s)
    out["language"] = getattr(info, "language", language)
    return out
