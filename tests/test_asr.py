"""S03.5 -- the joining and bookkeeping around faster-whisper, without a model."""
import os
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from types import SimpleNamespace

import pytest

from nepal.process import asr


def seg(start, end, text):
    return SimpleNamespace(start=start, end=end, text=text)


def test_segments_are_joined_and_put_on_the_recording_clock():
    out = asr.join_segments([seg(0.0, 1.5, " привет "), seg(1.5, 3.0, "как дела")],
                            offset_s=100.0)
    assert out["text"] == "привет как дела"
    assert out["segments"] == [
        {"start_s": 100.0, "end_s": 101.5, "text": "привет"},
        {"start_s": 101.5, "end_s": 103.0, "text": "как дела"}]


def test_empty_segments_are_dropped_not_joined_as_blanks():
    out = asr.join_segments([seg(0, 1, "  "), seg(1, 2, "да"), seg(2, 3, "")])
    assert out["text"] == "да" and len(out["segments"]) == 1


def test_nothing_in_gives_an_empty_transcript_not_none():
    out = asr.join_segments([])
    assert out == {"text": "", "segments": []}


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("NEPAL_SPEECH_WAV"),
                    reason="set NEPAL_SPEECH_WAV to a 16 kHz mono wav with speech")
def test_real_speech_comes_back_as_words():
    """Runs the configured model on real material; there is no synthesiser
    to fabricate speech with. Asserts words, not particular words."""
    from nepal.config import Config
    from nepal.process import audio
    cfg = Config.load(None)
    model = asr.load_model(cfg.get("spine.whisper_model"))
    x, _ = audio.read_window(os.environ["NEPAL_SPEECH_WAV"], 0.0, 20.0)
    out = asr.transcribe_window(model, x, language=cfg.get("spine.whisper_language"),
                                beam_size=1, offset_s=30.0)
    assert len(out["text"]) > 20 and out["segments"]
    assert out["segments"][0]["start_s"] >= 30.0        # on the recording clock
