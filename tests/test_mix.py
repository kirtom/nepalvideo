import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest

from nepal.process.mix import music_envelope, location_envelope, volume_expr, FLOOR_DB


# -- a tiny evaluator for the emitted expression --------------------------
# The expression is deliberately valid Python syntax (function calls and
# infix arithmetic), so evaluating it only needs between()/gte()/pow() in
# the namespace -- this is not a real ffmpeg parser, just enough to check
# the shape this module promises. The real boundary (does ffmpeg itself
# accept and evaluate it) is Task 13's slow test.
def _eval_expr(expr: str, t: float) -> float:
    ns = {
        "t": t,
        "between": lambda x, a, b: 1.0 if a <= x <= b else 0.0,
        "gte": lambda x, y: 1.0 if x >= y else 0.0,
        "pow": pow,
    }
    # eval() here is on this module's own generated output, not on
    # untrusted input, with __builtins__ stripped and only the three
    # functions the expression can call exposed -- a small sandbox, not a
    # real ffmpeg parser (that check is Task 13's slow test).
    return eval(expr, {"__builtins__": {}}, ns)


def _max_paren_depth(expr: str) -> int:
    depth = best = 0
    for ch in expr:
        if ch == "(":
            depth += 1
            best = max(best, depth)
        elif ch == ")":
            depth -= 1
    return best


# -- music_envelope --------------------------------------------------------

def test_music_envelope_speech_and_window_step1():
    """Step 1: one speech span, one window. 0 dB outside, -8 dB inside the
    span with 0.15 s ramps, -70 dB inside the window with 1 s ramps."""
    env = music_envelope(total_s=20.0, speech_spans=[(5.0, 10.0)],
                          windows=[{"t_in": 14.0, "t_out": 16.0}], silence=None)
    expected = [
        (0.0, 0.0), (5.0, 0.0), (5.15, -8.0), (9.85, -8.0), (10.0, 0.0),
        (14.0, 0.0), (15.0, -70.0), (16.0, 0.0),
    ]
    assert env == pytest.approx(expected, abs=1e-9)


def test_music_envelope_breakpoints_are_sorted_and_start_at_zero():
    env = music_envelope(total_s=20.0, speech_spans=[(5.0, 10.0)],
                          windows=[{"t_in": 14.0, "t_out": 16.0}], silence=None)
    times = [t for t, _ in env]
    assert times == sorted(times)
    assert len(set(times)) == len(times)          # no two points share a time
    assert env[0][0] == 0.0


def test_overlapping_speech_spans_merge_into_one_dip():
    """Two overlapping speech spans are one dip, not two."""
    env = music_envelope(total_s=15.0, speech_spans=[(2.0, 6.0), (5.0, 9.0)],
                          windows=[], silence=None)
    expected = [(0.0, 0.0), (2.0, 0.0), (2.15, -8.0), (8.85, -8.0), (9.0, 0.0)]
    assert env == pytest.approx(expected, abs=1e-9)


def test_window_inside_speech_span_the_deeper_level_wins():
    """A window inside a speech span is a window: it dips to the floor and
    returns to the speech level around it, not to the base level."""
    env = music_envelope(total_s=15.0, speech_spans=[(2.0, 12.0)],
                          windows=[{"t_in": 5.0, "t_out": 7.0}], silence=None)
    expected = [
        (0.0, 0.0), (2.0, 0.0), (2.15, -8.0), (5.0, -8.0), (6.0, -70.0),
        (7.0, -8.0), (11.85, -8.0), (12.0, 0.0),
    ]
    assert env == pytest.approx(expected, abs=1e-9)


def test_silence_window_ramps_to_the_floor_too():
    env = music_envelope(total_s=10.0, speech_spans=[], windows=[],
                          silence={"t_start": 4.0, "t_end": 6.0})
    times_values = dict(env)
    assert times_values[5.0] == FLOOR_DB
    assert times_values[4.0] == 0.0
    assert times_values[6.0] == 0.0


def test_music_envelope_with_nothing_going_on_is_flat_at_base():
    env = music_envelope(total_s=10.0, speech_spans=[], windows=[], silence=None)
    assert env == [(0.0, 0.0)]


def test_music_envelope_levels_are_configurable():
    env = music_envelope(total_s=10.0, speech_spans=[(2.0, 4.0)], windows=[],
                          silence=None, base_db=-3.0, under_speech_db=-11.0,
                          cue_fade_s=0.5)
    expected = [(0.0, -3.0), (2.0, -3.0), (2.5, -11.0), (3.5, -11.0), (4.0, -3.0)]
    assert env == pytest.approx(expected, abs=1e-9)


# -- location_envelope ------------------------------------------------------

def test_location_envelope_before_first_cue_is_zero_db():
    cues = [{"t_in": 2.0, "t_out": 6.0, "gain_lufs": -25.0}]
    env = location_envelope(total_s=20.0, cues=cues, full_lufs=-20.0)
    assert env[0] == (0.0, 0.0)


def test_location_envelope_holds_previous_cues_level_between_cues():
    """Ambience continues: between two cues the level doesn't fall back to
    0 dB, it holds whatever the previous cue set."""
    cues = [
        {"t_in": 2.0, "t_out": 6.0, "gain_lufs": -25.0},
        {"t_in": 10.0, "t_out": 14.0, "gain_lufs": -14.0},
    ]
    env = location_envelope(total_s=20.0, cues=cues, full_lufs=-20.0)
    expected = [(0.0, 0.0), (2.0, 0.0), (2.15, -5.0), (10.0, -5.0), (10.15, 6.0)]
    assert env == pytest.approx(expected, abs=1e-9)
    # explicitly: right up to the second cue's own ramp, the level is still
    # the first cue's, not 0 -- the gap between t=6 and t=10 is a hold.
    times_values = dict(env)
    assert times_values[10.0] == pytest.approx(-5.0)


def test_location_envelope_zero_db_means_full_lufs():
    cues = [{"t_in": 1.0, "t_out": 5.0, "gain_lufs": -20.0}]
    env = location_envelope(total_s=10.0, cues=cues, full_lufs=-20.0)
    # the cue's level equals full_lufs, so nothing changes from the 0 dB
    # baseline -- there is no transition to draw.
    assert env == [(0.0, 0.0)]


# -- volume_expr -------------------------------------------------------------

def test_volume_expr_two_point_envelope_evaluates_at_the_midpoint():
    env = [(0.0, -10.0), (10.0, 4.0)]
    expr = volume_expr(env)
    assert "\n" not in expr
    assert " " not in expr
    got = _eval_expr(expr, 5.0)
    want = (pow(10, -10.0 / 20) + pow(10, 4.0 / 20)) / 2.0
    assert got == pytest.approx(want)


def test_volume_expr_one_point_envelope_is_a_bare_gain():
    expr = volume_expr([(0.0, -6.0)])
    assert expr == f"pow(10,{-6.0:.3f}/20)"
    assert _eval_expr(expr, 123.456) == pytest.approx(pow(10, -6.0 / 20))


def test_volume_expr_holds_the_last_value():
    env = [(0.0, -10.0), (10.0, 4.0)]
    expr = volume_expr(env)
    got = _eval_expr(expr, 1000.0)
    assert got == pytest.approx(pow(10, 4.0 / 20))


def test_volume_expr_300_breakpoints_is_flat_and_shallow():
    """A film has hundreds of breakpoints; the expression must stay a flat
    sum (one between()/gte() level, not nested ifs whose depth would track
    the breakpoint count)."""
    env = [(float(i), float(-i % 12)) for i in range(300)]
    expr = volume_expr(env)
    assert "\n" not in expr
    assert " " not in expr
    assert expr.count("between(") == 299
    assert expr.count("gte(") == 1
    assert "if(" not in expr
    assert _max_paren_depth(expr) <= 6          # constant, not O(len(env))

    # spot-check correctness away from the edges
    t0, d0 = env[150]
    t1, d1 = env[151]
    mid = (t0 + t1) / 2.0
    got = _eval_expr(expr, mid)
    want = (pow(10, d0 / 20) + pow(10, d1 / 20)) / 2.0
    assert got == pytest.approx(want)
