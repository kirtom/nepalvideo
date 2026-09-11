import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest
from nepal.process.mix import duck_music_lufs, levels_for_shot


def test_music_sits_at_the_bed_level_without_speech():
    assert duck_music_lufs(base_music_lufs=-14.0, speech_duck_lufs=-22.0,
                           has_speech=False) == -14.0


def test_music_ducks_under_speech():
    assert duck_music_lufs(base_music_lufs=-14.0, speech_duck_lufs=-22.0,
                           has_speech=True) == -22.0


def test_the_duck_depth_is_fixed():
    """An earlier version varied the depth by estimated vocal presence, on sound
    reasoning with an unsound measurement: syllabic modulation cannot separate
    voice from real piano and strings, because note articulation and vibrato sit
    at the same 3-8 Hz. A real Desplat orchestral cue measured 0.42 against 0.75
    for a synthetic vocal. Ducking an instrumental several dB too far is worse
    than not varying the depth at all."""
    import inspect
    params = inspect.signature(duck_music_lufs).parameters
    assert "vocal_score" not in params


# -- per-shot levels ---------------------------------------------------

def test_location_audio_comes_up_under_speech():
    lv = levels_for_shot(has_speech=True)
    assert lv.location_lufs == 0.0
    assert lv.music_lufs == -22.0


def test_location_audio_sits_under_the_bed_without_speech():
    lv = levels_for_shot(has_speech=False)
    assert lv.music_lufs == -14.0
    assert lv.location_lufs == -28.0


def test_the_act_four_silence_window_drops_the_music_entirely():
    """The single most powerful move in the film, per the brief: music out,
    two or three seconds of wind and breathing."""
    lv = levels_for_shot(has_speech=False, in_silence_window=True)
    assert lv.music_lufs <= -60.0
    assert lv.location_lufs == 0.0
    assert "silence" in lv.reason


def test_silence_window_wins_over_speech():
    lv = levels_for_shot(has_speech=True, in_silence_window=True)
    assert lv.music_lufs <= -60.0


def test_levels_are_configurable():
    lv = levels_for_shot(has_speech=True, music_lufs=-12.0, duck_lufs_speech=-25.0)
    assert lv.music_lufs == -25.0


def test_reason_explains_the_choice():
    assert "speech" in levels_for_shot(has_speech=True).reason
    assert "no speech" in levels_for_shot(has_speech=False).reason
