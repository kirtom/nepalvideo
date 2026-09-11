"""Audio mix levels for S07 and S08.

The spec's S07 mix: the music bed sits at -14 LUFS, the original location audio
is ducked to -28 LUFS beneath it, and where a shot carries speech the music
ducks to -22 LUFS instead so the voice comes through.

An earlier version varied the duck depth by how vocal the music was, on the
reasoning that a sung track masks spoken narration more than an instrumental at
the same loudness. That reasoning is sound; the measurement was not. Estimating
vocal presence from an mp3 by syllabic modulation cannot separate voice from
real piano and strings, because note articulation and vibrato modulate at the
same 3-8 Hz -- an actual Desplat orchestral cue measured 0.42 against 0.75 for a
synthetic vocal and 0.001 for a pure-tone pad. Ducking an instrumental several
dB too far is worse than not varying the depth at all, so the depth is fixed,
as the spec has it.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MixLevels:
    """Target loudnesses for one moment of the timeline, in LUFS."""
    music_lufs: float
    location_lufs: float
    reason: str


def duck_music_lufs(*, base_music_lufs: float, speech_duck_lufs: float,
                    has_speech: bool) -> float:
    """Where the music bed should sit: its own level, or ducked under speech."""
    return speech_duck_lufs if has_speech else base_music_lufs


def levels_for_shot(*, has_speech: bool, music_lufs: float = -14.0,
                    duck_lufs: float = -28.0, duck_lufs_speech: float = -22.0,
                    in_silence_window: bool = False) -> MixLevels:
    """Both levels for one shot.

    ``in_silence_window`` is the Act 4 hard cut: music drops out entirely and
    the location audio comes up to full, which is the wind and breathing the
    brief calls the single most powerful move in the film.
    """
    if in_silence_window:
        return MixLevels(music_lufs=-70.0, location_lufs=0.0,
                         reason="Act 4 silence window: music out, location audio full")

    music = duck_music_lufs(base_music_lufs=music_lufs,
                            speech_duck_lufs=duck_lufs_speech,
                            has_speech=has_speech)
    if has_speech:
        return MixLevels(music_lufs=music, location_lufs=0.0,
                         reason="speech present: music ducked, location audio full")
    return MixLevels(music_lufs=music, location_lufs=duck_lufs,
                     reason="no speech: location audio under the bed")
