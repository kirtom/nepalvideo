"""Exertion from the GPS track: the slowest hour, the steepest gain, the stop."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from datetime import datetime, timedelta, timezone

import pytest

from nepal.spine.effort import (Effort, at, exertion, hardest_windows,
                                light_quality, nepal_hour, profile)
from nepal.spine.gps import GpsPoint

UTC = timezone.utc
T0 = datetime(2024, 5, 6, 0, 0, tzinfo=UTC)          # 05:45 Nepal


def pt(minutes, lat=28.5, lon=84.6, ele=4000.0):
    return GpsPoint(T0 + timedelta(minutes=minutes), lat, lon, ele)


def walk(n=6, m_per_step=90.0, climb_m=0.0):
    """n fixes a minute apart, moving m_per_step metres and climbing climb_m."""
    out = []
    for i in range(n):
        # ~111 km per degree of latitude
        out.append(pt(i, lat=28.5 + (i * m_per_step) / 111_000.0,
                      ele=4000.0 + i * climb_m))
    return out


# -- profile ------------------------------------------------------------

def test_speed_is_measured_between_fixes():
    prof = profile(walk(m_per_step=60.0))          # 60 m per minute = 1 m/s
    assert prof and all(e.speed_ms == pytest.approx(1.0, abs=0.05) for e in prof)


def test_altitude_gain_is_expressed_per_hour():
    prof = profile(walk(climb_m=5.0))              # 5 m per minute = 300 m/h
    assert prof[0].gain_m_per_h == pytest.approx(300.0, abs=5.0)


def test_fixes_too_close_together_are_noise_not_speed():
    """Over a few seconds GPS jitter dominates."""
    track = [pt(0), GpsPoint(T0 + timedelta(seconds=5), 28.5009, 84.6, 4000.0)]
    assert profile(track) == []


def test_a_gap_of_hours_says_nothing_about_any_moment_inside_it():
    track = [pt(0), pt(180)]
    assert profile(track, max_dt_s=3600) == []


def test_a_jeep_to_the_trailhead_is_not_exertion():
    """Above walking pace the party is in a vehicle, however far it travels."""
    fast = [pt(0), pt(1, lat=28.5 + 600 / 111_000.0)]     # 10 m/s
    assert profile(fast) == []


def test_standing_still_accumulates():
    still = [pt(i) for i in range(6)]                      # never moves
    prof = profile(still)
    assert prof[-1].stopped_s == pytest.approx(300.0, abs=1.0)
    assert prof[-1].is_stopped


def test_the_stopped_run_resets_once_moving_again():
    track = [pt(0), pt(1), pt(2), pt(3, lat=28.5 + 90 / 111_000.0)]
    assert profile(track)[-1].stopped_s == 0.0


# -- exertion -----------------------------------------------------------

def test_climbing_steeply_scores_higher_than_flat_walking():
    flat = Effort(T0, speed_ms=1.0, gain_m_per_h=0.0, alt_m=4000.0)
    steep = Effort(T0, speed_ms=1.0, gain_m_per_h=300.0, alt_m=4000.0)
    assert exertion(steep) > exertion(flat)


def test_slow_while_climbing_is_the_hardest_case():
    """The last hour to a pass: still going up, barely moving."""
    steady = Effort(T0, speed_ms=1.2, gain_m_per_h=250.0, alt_m=4900.0)
    crawling = Effort(T0, speed_ms=0.3, gain_m_per_h=250.0, alt_m=4900.0)
    assert exertion(crawling) > exertion(steady)


def test_ambling_downhill_is_not_effort_however_slow():
    slow_down = Effort(T0, speed_ms=0.3, gain_m_per_h=-200.0, alt_m=3000.0)
    assert exertion(slow_down) < 0.2


def test_a_long_stop_counts():
    """Nobody stands still for twenty minutes on a cold trail by choice."""
    brief = Effort(T0, 0.0, 0.0, 4500.0, stopped_s=60.0)
    long = Effort(T0, 0.0, 0.0, 4500.0, stopped_s=1200.0)
    assert exertion(long) > exertion(brief)


def test_exertion_stays_in_range():
    extreme = Effort(T0, 0.0, 5000.0, 5147.0, stopped_s=99999.0)
    assert 0.0 <= exertion(extreme) <= 1.0
    assert 0.0 <= exertion(Effort(T0, 9.0, -9999.0, 0.0)) <= 1.0


# -- light --------------------------------------------------------------

def test_nepal_runs_at_a_quarter_hour_offset():
    assert nepal_hour(datetime(2024, 5, 6, 0, 0, tzinfo=UTC)) == pytest.approx(5.75)


def test_the_alpine_start_scores_highest():
    """A pre-dawn push to a pass is dramatic whatever the light is doing."""
    predawn = datetime(2024, 5, 6, 22, 15, tzinfo=UTC)      # 04:00 Nepal
    midday = datetime(2024, 5, 6, 6, 15, tzinfo=UTC)        # 12:00 Nepal
    assert light_quality(predawn) > light_quality(midday)
    assert light_quality(predawn) == 1.0


def test_flat_midday_sun_is_the_worst_of_it():
    midday = datetime(2024, 5, 6, 6, 15, tzinfo=UTC)        # 12:00 Nepal
    golden = datetime(2024, 5, 6, 12, 15, tzinfo=UTC)       # 18:00 Nepal
    assert light_quality(golden) > light_quality(midday)


def test_light_quality_is_always_a_weight():
    for h in range(24):
        assert 0.0 <= light_quality(datetime(2024, 5, 6, h, tzinfo=UTC)) <= 1.0


# -- lookup and windows -------------------------------------------------

def test_the_nearest_sample_is_returned():
    prof = profile(walk(climb_m=5.0))
    got = at(prof, T0 + timedelta(minutes=3, seconds=10))
    assert got is not None and abs((got.ts - T0).total_seconds() - 180) <= 60


def test_a_moment_with_no_nearby_fix_gets_nothing():
    prof = profile(walk())
    assert at(prof, T0 + timedelta(hours=5)) is None


def test_the_music_drops_out_at_the_hardest_moments():
    easy = [Effort(T0 + timedelta(minutes=i), 1.5, 0.0, 3000.0) for i in range(20)]
    hard = [Effort(T0 + timedelta(minutes=40 + i * 10), 0.2, 320.0, 5000.0)
            for i in range(3)]
    wins = hardest_windows(easy + hard, count=3, window_s=15.0)
    assert len(wins) == 3
    for start, end in wins:
        assert (end - start).total_seconds() == pytest.approx(15.0)
    # every window sits on one of the hard samples
    assert all(any(abs((h.ts - (s + (e - s) / 2)).total_seconds()) < 1
                   for h in hard) for s, e in wins)


def test_windows_do_not_bunch_into_one_minute():
    """The film must not fall silent twice in the same breath."""
    cluster = [Effort(T0 + timedelta(seconds=i * 10), 0.2, 320.0, 5000.0)
               for i in range(30)]
    wins = hardest_windows(cluster, count=4, min_gap_s=120.0)
    mids = [s + (e - s) / 2 for s, e in wins]
    for a, b in zip(mids, mids[1:]):
        assert (b - a).total_seconds() >= 120.0


def test_windows_come_back_in_film_order():
    prof = [Effort(T0 + timedelta(minutes=i * 5), 0.3, 300.0 - i, 5000.0)
            for i in range(6)]
    wins = hardest_windows(prof, count=3)
    assert wins == sorted(wins)


def test_no_track_no_windows():
    assert hardest_windows([], count=4) == []


# -- heart rate (Film v2 section 13.1) ----------------------------------

def test_profile_carries_heart_rate_from_the_track():
    pts = [GpsPoint(T0 + timedelta(minutes=i), 28.5 + i * 90 / 111_000.0, 84.6, 4000.0,
                    "strava", None, 100.0 + i, "A") for i in range(4)]
    prof = profile(pts)
    assert [e.hr_bpm for e in prof] == [101.0, 102.0, 103.0]


def test_heart_rate_raises_exertion_on_the_same_climb():
    gps_only = Effort(T0, 0.5, 250.0, 4500.0)
    hard = Effort(T0, 0.5, 250.0, 4500.0, hr_bpm=160.0)
    easy = Effort(T0, 0.5, 250.0, 4500.0, hr_bpm=80.0)
    assert exertion(hard) > exertion(gps_only) > exertion(easy)


def test_heart_rate_is_clamped_to_the_configured_band():
    e = Effort(T0, 0.5, 0.0, 4500.0, hr_bpm=250.0)
    assert exertion(e, hr_rest=60.0, hr_max=170.0) == pytest.approx(0.5, abs=1e-6)
