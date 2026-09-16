"""S03.7 -- the quality gate."""
from __future__ import annotations

import pytest

from nepal.process.gate import has_face, verdict

CAMERA = {"min_sharpness": 4.0, "max_exposure_pen": 0.15,
          "min_stability": 0.35, "min_duration_s": 1.5}
TELEGRAM = {"min_sharpness": 2.0, "max_exposure_pen": 0.35,
            "min_stability": 0.15, "min_duration_s": 1.0}


def shot(**kw):
    base = {"start_s": 0.0, "end_s": 4.0, "sharpness": 6.0,
            "exposure_pen": 0.05, "stability": 0.7, "has_speech": 0}
    return {**base, **kw}


def test_a_good_shot_survives():
    assert verdict(shot(), CAMERA) is None


@pytest.mark.parametrize("field,value,reason", [
    ("sharpness", 3.0, "soft"),
    ("exposure_pen", 0.4, "exposure"),
    ("stability", 0.1, "shaky"),
    ("end_s", 1.0, "too short"),
])
def test_each_rule_rejects_and_says_why(field, value, reason):
    assert verdict(shot(**{field: value}), CAMERA) == reason


def test_telegram_is_not_judged_on_the_camera_curve():
    """Act 1 is planning material shot on phones and re-compressed by Telegram.
    Judged on the camera curve it all disappears, and the act loses its core."""
    s = shot(sharpness=2.5, exposure_pen=0.3, stability=0.2, end_s=1.2)
    assert verdict(s, CAMERA) is not None
    assert verdict(s, TELEGRAM) is None


def test_speech_lowers_the_sharpness_floor():
    """Voice is the spine of the film. A soft frame under a good sentence is
    worth more than a sharp frame of nothing."""
    s = shot(sharpness=2.6)
    assert verdict(s, CAMERA) == "soft"
    assert verdict({**s, "has_speech": 1}, CAMERA) is None


def test_speech_waives_stability_entirely():
    """The picture can be cut away from while the audio keeps running."""
    s = shot(stability=0.05)
    assert verdict(s, CAMERA) == "shaky"
    assert verdict({**s, "has_speech": 1}, CAMERA) is None


def test_speech_does_not_waive_exposure():
    """A blown-out frame carries no picture at all; the line can be salvaged in
    S06 as sound over other footage, but the shot itself is not usable."""
    assert verdict(shot(exposure_pen=0.9, has_speech=1), CAMERA) == "exposure"


def test_speech_shortens_but_does_not_abolish_the_minimum_length():
    assert verdict(shot(end_s=1.2, has_speech=1), CAMERA) is None
    assert verdict(shot(end_s=0.4, has_speech=1), CAMERA) == "too short"


def test_an_unmeasured_metric_does_not_reject():
    """A photograph has no stability by design, and a shot whose proxy failed
    has nothing at all. The gate removes the demonstrably unusable, not the
    unknown."""
    assert verdict(shot(sharpness=None, exposure_pen=None, stability=None),
                   CAMERA) is None


def test_a_photograph_is_never_shaky():
    assert verdict({"start_s": 0.0, "end_s": 3.0, "sharpness": 5.0,
                    "exposure_pen": 0.05, "stability": None}, CAMERA) is None


def test_duration_is_checked_even_without_metrics():
    assert verdict({"start_s": 0.0, "end_s": 0.5}, CAMERA) == "too short"


# -- has_face (Film v2 step 1) ------------------------------------------

@pytest.mark.parametrize("score,cluster,want", [
    (0.83, "keller", 1),
    (0.83, None, 1),        # detected, not attributed to a known person
    (0.30, None, 0),        # below the detector floor: a buckle, lichen
    (0.0, None, 0),         # looked at, nothing there
    (None, None, 0),        # never looked at
    (None, "kulikov", 1),   # re-clustered from stored embeddings after a re-detect
])
def test_has_face_is_derived_from_what_was_measured(score, cluster, want):
    assert has_face(score, cluster, min_det_score=0.55) == want
