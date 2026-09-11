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


def test_an_instrumental_track_gets_no_extra_duck():
    assert duck_music_lufs(base_music_lufs=-14.0, speech_duck_lufs=-22.0,
                           has_speech=True, vocal_score=0.1) == -22.0


def test_a_vocal_track_ducks_further_under_speech():
    """A sung track masks spoken narration far more than an instrumental at the
    same loudness, because it occupies the same range."""
    instrumental = duck_music_lufs(base_music_lufs=-14.0, speech_duck_lufs=-22.0,
                                   has_speech=True, vocal_score=0.2)
    vocal = duck_music_lufs(base_music_lufs=-14.0, speech_duck_lufs=-22.0,
                            has_speech=True, vocal_score=1.0)
    assert vocal < instrumental
    assert vocal == pytest.approx(-26.0)


def test_the_extra_duck_comes_in_gradually():
    """A step at the threshold would make two similar tracks mix differently."""
    just_over = duck_music_lufs(base_music_lufs=-14.0, speech_duck_lufs=-22.0,
                                has_speech=True, vocal_score=0.61)
    well_over = duck_music_lufs(base_music_lufs=-14.0, speech_duck_lufs=-22.0,
                                has_speech=True, vocal_score=0.9)
    assert -22.0 > just_over > well_over
    assert abs(just_over + 22.0) < 0.5, "just past the threshold is nearly no change"


def test_vocal_score_is_ignored_when_there_is_no_speech():
    assert duck_music_lufs(base_music_lufs=-14.0, speech_duck_lufs=-22.0,
                           has_speech=False, vocal_score=1.0) == -14.0


# -- per-shot levels ---------------------------------------------------

def test_location_audio_comes_up_under_speech():
    lv = levels_for_shot(has_speech=True, vocal_score=0.0)
    assert lv.location_lufs == 0.0
    assert lv.music_lufs == -22.0


def test_location_audio_sits_under_the_bed_without_speech():
    lv = levels_for_shot(has_speech=False, vocal_score=0.0)
    assert lv.music_lufs == -14.0
    assert lv.location_lufs == -28.0


def test_the_act_four_silence_window_drops_the_music_entirely():
    """The single most powerful move in the film, per the brief: music out,
    two or three seconds of wind and breathing."""
    lv = levels_for_shot(has_speech=False, vocal_score=0.9, in_silence_window=True)
    assert lv.music_lufs <= -60.0
    assert lv.location_lufs == 0.0
    assert "silence" in lv.reason


def test_silence_window_wins_over_speech():
    lv = levels_for_shot(has_speech=True, vocal_score=0.0, in_silence_window=True)
    assert lv.music_lufs <= -60.0


def test_reason_explains_the_choice():
    assert "speech" in levels_for_shot(has_speech=True, vocal_score=0.0).reason
    assert "vocal" in levels_for_shot(has_speech=True, vocal_score=1.0).reason
