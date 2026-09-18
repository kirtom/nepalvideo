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
from typing import Any, Iterable, Mapping

import numpy as np

log = logging.getLogger(__name__)


# Per-segment attributes worth keeping, from faster-whisper's Segment. The
# first two are what the hallucination filter (Film v2 section 4.1) reads;
# the third is there because a low mean log-probability is the next thing
# one would want to look at when the first two disagree.
CONFIDENCE_ATTRS = ("no_speech_prob", "compression_ratio", "avg_logprob")


def join_segments(segments: Iterable[Any], offset_s: float = 0.0) -> dict[str, Any]:
    """One transcript from faster-whisper's segments.

    Segment times are relative to the audio handed in, so ``offset_s`` puts
    them back on the recording's clock -- S06 cuts on the recording's clock.

    Whisper's confidence and its word times travel with the segment when the
    segment has them. The first version kept only start, end and text, so
    the hallucination filter had nothing to read and a beat could only be
    cut where whisper drew a segment -- which, with no previous text to
    condition on, is usually the whole 20 s shot.
    """
    parts: list[dict[str, Any]] = []
    words: list[str] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        part: dict[str, Any] = {"start_s": round(float(seg.start) + offset_s, 3),
                                "end_s": round(float(seg.end) + offset_s, 3),
                                "text": text}
        for attr in CONFIDENCE_ATTRS:
            v = getattr(seg, attr, None)
            if v is not None:
                part[attr] = round(float(v), 4)
        seg_words = getattr(seg, "words", None)
        if seg_words:
            part["words"] = [{"start_s": round(float(w.start) + offset_s, 3),
                              "end_s": round(float(w.end) + offset_s, 3),
                              "word": (w.word or "").strip(),
                              "probability": round(float(getattr(w, "probability", 0.0)), 3)}
                             for w in seg_words if (w.word or "").strip()]
        parts.append(part)
        words.append(text)
    return {"text": " ".join(words).strip(), "segments": parts}


def synthetic_transcript(shot: Mapping[str, Any]) -> dict[str, Any]:
    """A one-segment transcript for a shot that has only its text.

    The shots transcribed before S03.5 wrote a JSON beside the row have a
    transcript and no segments. One segment spanning the shot is the honest
    description of what is known -- the words were said somewhere in it --
    and it keeps them readable by the filter and the beat sheet until a
    re-transcription replaces it. Marked so nothing mistakes it for timing.
    """
    text = (shot.get("transcript") or "").strip()
    segs = [{"start_s": round(float(shot["start_s"]), 3),
             "end_s": round(float(shot["end_s"]), 3), "text": text}] if text else []
    return {"shot_id": shot["shot_id"], "text": text, "segments": segs, "synthetic": True}


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
                      beam_size: int, offset_s: float,
                      word_timestamps: bool = True) -> dict[str, Any]:
    """Transcribe one shot's samples; times come back on the recording clock."""
    if samples.size == 0:
        return {"text": "", "segments": [], "language": language}
    segments, info = model.transcribe(
        samples, language=language or None, beam_size=int(beam_size),
        vad_filter=False,                 # S03.4 already decided this is speech
        condition_on_previous_text=False,  # a 20 s shot has no useful previous
        word_timestamps=bool(word_timestamps))  # a beat is cut at a word
    out = join_segments(segments, offset_s)
    out["language"] = getattr(info, "language", language)
    return out
