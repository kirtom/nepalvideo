"""S03.4 -- loudness, wind and speech for every video shot.

Three signals, each from the 16 kHz mono track S03.1 already extracted:

- ``audio_lufs``: EBU R128 integrated loudness over the shot window, from
  ffmpeg's ``ebur128`` filter (spec section S03.4). S07 mixes against it.
- ``wind_lf_share``: the share of spectral energy below a cutoff. A camera on
  a pole at five thousand metres records wind as broadband rumble that sits
  under 200 Hz; when most of the energy is there the track is roaring, not
  recording. The share is stored and the flag derived where it is consumed,
  so the threshold can move without a re-measure -- the lesson of jerk_ref.
- ``speech_s``: seconds of speech inside the window, from silero-VAD. The
  model is the ONNX build faster-whisper ships for its own VAD, so no torch.
  Run once per recording, not once per shot: a sentence does not care where
  the shot detector put a boundary, and the context either side is what lets
  the VAD hold through a breath.

Thin wrappers around the two tools; everything that decides is pure.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from nepal.util import proc

log = logging.getLogger(__name__)

WIND_CUTOFF_HZ = 200.0
WIND_SHARE = 0.80
SPEECH_MIN_S = 0.6
SAMPLE_RATE = 16000

_EBUR_I = re.compile(r"^\s*I:\s*(-?[\d.]+|-inf)\s*LUFS", re.MULTILINE)
# ebur128's absolute gate: blocks below this never count, and a stream with
# nothing above it is reported at exactly -70 (older builds print -inf).
EBUR_FLOOR_LUFS = -70.0


# -- loudness ----------------------------------------------------------

def parse_ebur128(stderr: str) -> float | None:
    """Integrated loudness from the filter's end-of-stream summary.

    Digital silence comes back as ``-inf`` or as the -70 LUFS absolute-gate
    floor depending on the ffmpeg build. Either way nothing was measured,
    which is not the same as a very quiet shot, so it is None rather than a
    large negative number that S07 would then try to normalise.
    """
    m = _EBUR_I.search(stderr or "")
    if not m or m.group(1) == "-inf":
        return None
    value = float(m.group(1))
    return None if value <= EBUR_FLOOR_LUFS else value


def lufs_of(wav: str | Path, start_s: float, end_s: float,
            *, timeout: float = 120.0) -> float | None:
    """EBU R128 integrated loudness of one window of a wav file."""
    dur = max(0.0, float(end_s) - float(start_s))
    if dur <= 0:
        return None
    # The wrapper in proc runs ffmpeg at -loglevel error, and ebur128 prints
    # its summary at info, so this is the one ffmpeg call that keeps info on.
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "info",
           "-ss", f"{start_s:.3f}", "-t", f"{dur:.3f}", "-i", str(wav),
           "-af", "ebur128=peak=none", "-f", "null", "-"]
    res = proc.run(cmd, check=False, timeout=timeout)
    return parse_ebur128(res.stderr)


# -- wind --------------------------------------------------------------

def low_frequency_share(samples: np.ndarray, sr: int, *,
                        cutoff_hz: float = WIND_CUTOFF_HZ) -> float:
    """Share of spectral energy below ``cutoff_hz``; 0.0 for silence."""
    x = np.asarray(samples, dtype=np.float64)
    if x.size < 2:
        return 0.0
    x = x - x.mean()                          # DC is not wind
    power = np.abs(np.fft.rfft(x)) ** 2
    total = float(power.sum())
    if total <= 0.0:
        return 0.0
    freqs = np.fft.rfftfreq(x.size, 1.0 / sr)
    return float(power[freqs < cutoff_hz].sum() / total)


def is_wind(share: float | None, *, threshold: float = WIND_SHARE) -> bool:
    return share is not None and share > threshold


# -- speech ------------------------------------------------------------

def segments_to_seconds(raw: Sequence[dict[str, Any]], sr: int) -> list[tuple[float, float]]:
    """silero reports sample indices; the shots table is in seconds."""
    return [(float(s["start"]) / sr, float(s["end"]) / sr) for s in raw]


def speech_segments_of(wav: str | Path, *, sr: int = SAMPLE_RATE,
                       min_silence_ms: int = 500, pad_ms: int = 200
                       ) -> list[tuple[float, float]]:
    """Speech spans over a whole recording, in seconds.

    The wav is read in full: silero wants the whole signal and the longest
    recording here is half an hour at 16 kHz mono, which is 56 MB of float32.
    """
    import soundfile as sf
    from faster_whisper import vad

    x, got = sf.read(str(wav), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if got != sr:
        raise ValueError(f"{wav}: {got} Hz, expected {sr}; S03.1 writes {sr} Hz")
    if x.size == 0:
        return []
    opts = vad.VadOptions(min_silence_duration_ms=min_silence_ms, speech_pad_ms=pad_ms)
    return segments_to_seconds(vad.get_speech_timestamps(x, opts, sampling_rate=sr), sr)


def speech_seconds(segments: Sequence[tuple[float, float]],
                   start_s: float, end_s: float) -> float:
    """Seconds of speech that fall inside ``[start_s, end_s]``."""
    total = 0.0
    for a, b in segments:
        lo, hi = max(a, start_s), min(b, end_s)
        if hi > lo:
            total += hi - lo
    return float(total)


def has_speech(speech_s: float | None, *, min_s: float = SPEECH_MIN_S) -> bool:
    """A shot carries speech when there is enough of it to hear as speech.

    A single syllable clipped at a shot boundary does not earn the gate's
    speech reprieve or a transcription; a sentence does.
    """
    return speech_s is not None and speech_s >= min_s


# -- reading a window ----------------------------------------------------

def read_window(wav: str | Path, start_s: float, end_s: float,
                *, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Samples of one shot window, without decoding the whole file."""
    import soundfile as sf
    with sf.SoundFile(str(wav)) as f:
        got = f.samplerate
        a = max(0, int(round(start_s * got)))
        b = max(a, int(round(end_s * got)))
        f.seek(min(a, f.frames))
        x = f.read(min(b, f.frames) - min(a, f.frames), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, got
