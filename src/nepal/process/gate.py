"""S03.7 -- the quality gate.

The gate is the pipeline's only cheap reduction. Everything after it costs
money per shot: a caption from a vision model, a transcription, a human at Gate
2 reading a list. The spec expects roughly 70% of camera material to be
rejected here, and the corpus is 1400 candidate shots against a film of 150 to
200 slots -- so the gate is not a formality, it is the step that makes the rest
affordable.

Two things it must not do:

*Judge every source on the same curve.* A Telegram video compressed to 480p is
not competing with a 5.7K camera; it is competing with nothing, because it is
the only footage of the planning months. The curves come from the config, one
per ``quality_curve``, and the telegram curve is deliberately lax.

*Throw away speech.* The brief makes voice the spine of the film (spec 1.4).
A soft, wobbly frame under a sentence that carries the story is worth keeping;
a sharp frame of nothing is not. So a shot with speech is judged against a
lowered sharpness floor and no stability floor at all -- the picture can always
be cut away from while the audio continues, which is exactly what a documentary
does with its best lines.
"""
from __future__ import annotations

from typing import Any, Mapping

SPEECH_SHARPNESS_FACTOR = 0.6
SPEECH_MIN_DURATION_S = 1.0


def verdict(shot: Mapping[str, Any], curve: Mapping[str, float], *,
            speech_sharpness_factor: float = SPEECH_SHARPNESS_FACTOR,
            speech_min_duration_s: float = SPEECH_MIN_DURATION_S) -> str | None:
    """``None`` if the shot survives, else the reason it did not.

    A metric that was never measured does not reject: an unmeasured shot is an
    unknown, and the gate's job is to remove the demonstrably unusable, not to
    punish gaps in the metrics. Photographs, for instance, have no stability by
    design.
    """
    speech = bool(shot.get("has_speech"))
    duration = float(shot.get("end_s") or 0.0) - float(shot.get("start_s") or 0.0)

    min_dur = float(curve.get("min_duration_s", 0.0))
    if speech:
        min_dur = min(min_dur, float(speech_min_duration_s))
    if duration < min_dur:
        return "too short"

    min_sharp = float(curve.get("min_sharpness", 0.0))
    if speech:
        min_sharp *= float(speech_sharpness_factor)
    sharp = shot.get("sharpness")
    if sharp is not None and float(sharp) < min_sharp:
        return "soft"

    pen = shot.get("exposure_pen")
    if pen is not None and float(pen) > float(curve.get("max_exposure_pen", 1.0)):
        return "exposure"

    stability = shot.get("stability")
    if stability is not None and not speech:
        if float(stability) < float(curve.get("min_stability", 0.0)):
            return "shaky"
    return None
