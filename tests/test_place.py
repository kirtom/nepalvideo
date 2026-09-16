"""Where and when a moment was: position, altitude, place, trek day."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from datetime import datetime, timedelta, timezone

import pytest

from nepal.spine import place
from nepal.spine.acts import ActBoundary
from nepal.spine.dem import Srtm
from nepal.spine.geocode import Gazetteer, Place
from nepal.spine.gps import GpsPoint

UTC = timezone.utc
T0 = datetime(2024, 5, 1, 2, 0, tzinfo=UTC)          # trek day 1, morning


def _track():
    return [GpsPoint(T0 + timedelta(hours=h), 28.5 + 0.01 * h, 84.6, 2000.0 + 100 * h,
                     "strava", None, 100.0, "A") for h in range(4)]


def _bounds():
    return [ActBoundary(1, T0 - timedelta(days=60), T0 - timedelta(hours=1)),
            ActBoundary(2, T0 - timedelta(hours=1), T0 + timedelta(days=3)),
            ActBoundary(3, T0 + timedelta(days=3), T0 + timedelta(days=6)),
            ActBoundary(4, T0 + timedelta(days=6), T0 + timedelta(days=7)),
            ActBoundary(5, T0 + timedelta(days=7), T0 + timedelta(days=12))]


GAZ = Gazetteer([Place("Namrung", 28.515, 84.6, "P", "PPL", 400)])


def test_trek_span_runs_from_act_2_to_the_last_act():
    span = place.trek_span(_bounds())
    assert span == (T0 - timedelta(hours=1), T0 + timedelta(days=12))


def test_day_of_is_null_outside_the_trek():
    span = place.trek_span(_bounds())
    assert place.day_of(T0 + timedelta(hours=2), span) == 1
    assert place.day_of(T0 + timedelta(days=1, hours=2), span) == 2
    assert place.day_of(T0 - timedelta(days=30), span) is None
    assert place.day_of(T0 + timedelta(days=40), span) is None


def test_place_at_interpolates_and_names_and_dates(tmp_path):
    got = place.place_at(_track(), T0 + timedelta(hours=1, minutes=30),
                         srtm=Srtm(tmp_path), gazetteer=GAZ, max_gap_s=14400.0,
                         span=place.trek_span(_bounds()))
    assert got.lat == pytest.approx(28.515, abs=1e-6)
    assert got.alt_m == pytest.approx(2150.0) and got.alt_source == "gpx"
    assert got.place_name == "Namrung"
    assert got.day_index == 1


def test_place_at_outside_the_track_has_no_position_but_still_a_day(tmp_path):
    got = place.place_at(_track(), T0 + timedelta(days=2),
                         srtm=Srtm(tmp_path), gazetteer=GAZ, max_gap_s=14400.0,
                         span=place.trek_span(_bounds()))
    assert got.lat is None and got.place_name is None and got.alt_m is None
    assert got.day_index == 3


def test_describe_a_photos_own_fix(tmp_path):
    alt, src, name = place.describe(28.515, 84.6, srtm=Srtm(tmp_path), gazetteer=GAZ)
    assert alt is None and src == "none" and name == "Namrung"
