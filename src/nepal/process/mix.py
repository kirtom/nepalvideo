"""Audio mix levels for S07 and S08.

The spec's S07 mix: the music bed sits at -14 LUFS, the original location audio
is ducked to -28 LUFS beneath it, and where a shot carries speech the music
ducks to -22 LUFS instead so the voice comes through.

One refinement on top of that. Ducking a *sung* track under spoken narration is
not the same job as ducking an instrumental: a vocal occupies the same
frequency range as the narration, so at equal loudness it masks the voice far
more. The fix is not to throw vocal tracks away -- the narration here is sparse,
and a good track beats a merely instrumental one -- it is to duck them a little
further when they overlap speech. S02.7 already measures vocal presence per
track, so the mix can use it.
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
                    has_speech: bool, vocal_score: float = 0.0,
                    vocal_threshold: float = 0.6,
                    vocal_extra_db: float = 4.0) -> float:
    """Where the music bed should sit.

    Without speech it stays at its bed level. Under speech it drops to
    ``speech_duck_lufs``, and a further ``vocal_extra_db`` when the track is
    itself vocal-heavy, scaled by how far past the threshold it is so the
    correction comes in gradually rather than as a step.
    """
    if not has_speech:
        return base_music_lufs
    target = speech_duck_lufs
    if vocal_score > vocal_threshold:
        headroom = max(0.0, min(1.0, (vocal_score - vocal_threshold)
                                / max(1e-6, 1.0 - vocal_threshold)))
        target -= vocal_extra_db * headroom
    return target


def levels_for_shot(*, has_speech: bool, vocal_score: float,
                    music_lufs: float = -14.0, duck_lufs: float = -28.0,
                    duck_lufs_speech: float = -22.0,
                    vocal_threshold: float = 0.6,
                    vocal_extra_db: float = 4.0,
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
                            has_speech=has_speech, vocal_score=vocal_score,
                            vocal_threshold=vocal_threshold,
                            vocal_extra_db=vocal_extra_db)
    if has_speech:
        reason = "speech present: music ducked"
        if vocal_score > vocal_threshold:
            reason += f", a further {music_lufs - music - (music_lufs - duck_lufs_speech):.1f} dB for a vocal track"
        return MixLevels(music_lufs=music, location_lufs=0.0, reason=reason)

    return MixLevels(music_lufs=music, location_lufs=duck_lufs,
                     reason="no speech: location audio under the bed")
