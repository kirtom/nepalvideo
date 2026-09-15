"""S03.4 -- loudness, wind and speech per shot.

The pure parts are tested on arrays; the two tools (ffmpeg's ebur128 and the
silero VAD that faster-whisper ships) are tested at their real boundary under
the slow marker, because every audio metric this pipeline could get wrong is
wrong where Python meets the tool, not in the arithmetic.
"""
import os
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from nepal.process import audio


# -- parsing ffmpeg ----------------------------------------------------

EBUR_SUMMARY = """
[Parsed_ebur128_0 @ 0x5] Summary:

  Integrated loudness:
    I:         -23.4 LUFS
    Threshold: -33.6 LUFS

  Loudness range:
    LRA:         4.1 LU
    Threshold: -43.7 LUFS
    LRA low:   -26.3 LUFS
    LRA high:  -22.2 LUFS
"""


def test_integrated_loudness_is_read_from_the_summary():
    assert audio.parse_ebur128(EBUR_SUMMARY) == pytest.approx(-23.4)


def test_silence_reports_no_loudness_rather_than_a_number():
    # ffmpeg prints -inf or its -70 LUFS floor for digital silence, depending
    # on the build; either is "nothing measured"
    assert audio.parse_ebur128(EBUR_SUMMARY.replace("-23.4", "-inf")) is None
    assert audio.parse_ebur128(EBUR_SUMMARY.replace("-23.4", "-70.0")) is None
    assert audio.parse_ebur128(EBUR_SUMMARY.replace("-23.4", "-69.9")) == pytest.approx(-69.9)
    assert audio.parse_ebur128("") is None


# -- wind --------------------------------------------------------------

def _tone(hz: float, sr: int = 16000, s: float = 2.0) -> np.ndarray:
    t = np.arange(int(sr * s)) / sr
    return np.sin(2 * np.pi * hz * t).astype(np.float32)


def test_low_frequency_share_separates_rumble_from_speech_band():
    sr = 16000
    rumble = _tone(60, sr) + 0.5 * _tone(120, sr)
    voice = _tone(300, sr) + 0.5 * _tone(1200, sr)
    assert audio.low_frequency_share(rumble, sr, cutoff_hz=200) > 0.95
    assert audio.low_frequency_share(voice, sr, cutoff_hz=200) < 0.05


def test_low_frequency_share_of_silence_is_zero_not_nan():
    assert audio.low_frequency_share(np.zeros(16000, dtype=np.float32), 16000,
                                     cutoff_hz=200) == 0.0


def test_wind_is_the_share_against_the_threshold():
    assert audio.is_wind(0.81, threshold=0.80)
    assert not audio.is_wind(0.79, threshold=0.80)
    assert not audio.is_wind(None, threshold=0.80)


# -- speech ------------------------------------------------------------

def test_speech_seconds_is_the_overlap_with_the_shot_window():
    segs = [(0.0, 3.0), (10.0, 12.0), (19.0, 25.0)]
    assert audio.speech_seconds(segs, 2.0, 11.0) == pytest.approx(2.0)   # 2-3, 10-11
    assert audio.speech_seconds(segs, 4.0, 9.0) == 0.0
    assert audio.speech_seconds(segs, 20.0, 22.0) == pytest.approx(2.0)
    assert audio.speech_seconds([], 0.0, 5.0) == 0.0


def test_has_speech_needs_a_minimum_not_a_single_syllable():
    assert audio.has_speech(1.0, min_s=0.6)
    assert not audio.has_speech(0.3, min_s=0.6)
    assert not audio.has_speech(None, min_s=0.6)


def test_vad_segments_are_converted_from_samples_to_seconds():
    raw = [{"start": 0, "end": 16000}, {"start": 32000, "end": 40000}]
    assert audio.segments_to_seconds(raw, 16000) == [(0.0, 1.0), (2.0, 2.5)]


# -- the real boundary -------------------------------------------------

def _wav(tmp_path, name: str, samples: np.ndarray, sr: int = 16000):
    import soundfile as sf
    p = tmp_path / name
    sf.write(p, samples, sr, subtype="PCM_16")
    return p


@pytest.mark.slow
def test_ffmpeg_ebur128_reads_a_known_tone(tmp_path):
    """A 1 kHz sine 10 dB quieter must read about 10 LU quieter. Absolute
    values depend on K-weighting; the difference does not."""
    loud = _wav(tmp_path, "loud.wav", 0.1 * _tone(1000, s=5.0))     # -20 dBFS
    quiet = _wav(tmp_path, "quiet.wav", 0.0316 * _tone(1000, s=5.0))  # -30 dBFS
    a = audio.lufs_of(loud, 0.0, 5.0)
    b = audio.lufs_of(quiet, 0.0, 5.0)
    assert a is not None and b is not None
    assert a - b == pytest.approx(10.0, abs=0.5)
    assert -24.0 < a < -16.0


@pytest.mark.slow
def test_ffmpeg_ebur128_windows_the_shot_not_the_file(tmp_path):
    """The first half is loud, the second is quiet; each window must read
    its own half."""
    x = np.concatenate([0.1 * _tone(1000, s=4.0), 0.01 * _tone(1000, s=4.0)])
    p = _wav(tmp_path, "two.wav", x)
    assert audio.lufs_of(p, 0.0, 4.0) - audio.lufs_of(p, 4.0, 8.0) == pytest.approx(20.0, abs=0.7)


@pytest.mark.slow
def test_silence_and_tone_are_not_speech(tmp_path):
    sr = 16000
    p = _wav(tmp_path, "tone.wav", np.concatenate([np.zeros(sr * 2, np.float32),
                                                   0.3 * _tone(440, sr, 3.0)]))
    segs = audio.speech_segments_of(p)
    assert audio.speech_seconds(segs, 0.0, 5.0) < 0.3


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("NEPAL_SPEECH_WAV"),
                    reason="set NEPAL_SPEECH_WAV to a 16 kHz mono wav with speech")
def test_real_speech_is_detected_as_speech():
    """No synthesiser is available to fabricate speech, so the positive case
    runs against real material when it is pointed at some."""
    p = pathlib.Path(os.environ["NEPAL_SPEECH_WAV"])
    segs = audio.speech_segments_of(p)
    import soundfile as sf
    dur = sf.info(p).duration
    assert audio.speech_seconds(segs, 0.0, dur) > 0.3 * dur
