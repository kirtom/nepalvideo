import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from datetime import datetime, timedelta, timezone
import pytest

from nepal.spine.gps import (GpsPoint, haversine_m, parse_gpx, write_gpx, merge_points,
                             drop_outliers, interpolate_at, envelope, coverage, day_index)

UTC = timezone.utc
NEPAL = timezone(timedelta(hours=5, minutes=45))


def t(h, m=0, s=0, day=15):
    return datetime(2023, 10, day, h, m, s, tzinfo=UTC)


def p(hour, lat=27.7, lon=86.7, source="phone_keller", acc=None, day=15, ele=None):
    return GpsPoint(t(hour, day=day), lat, lon, ele, source, acc)


# -- distance ----------------------------------------------------------

def test_haversine_known_distance():
    # Lukla -> Namche, about 12 km line of sight
    d = haversine_m(27.6869, 86.7314, 27.8069, 86.7133)
    assert 12_000 < d < 14_000


def test_haversine_zero():
    assert haversine_m(27.7, 86.7, 27.7, 86.7) == pytest.approx(0.0, abs=1e-6)


# -- gpx ---------------------------------------------------------------

GPX = """<?xml version="1.0"?>
<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">
 <trk><trkseg>
  <trkpt lat="27.6869" lon="86.7314"><ele>2860</ele><time>2023-10-15T04:00:00Z</time></trkpt>
  <trkpt lat="27.8069" lon="86.7133"><ele>3440</ele><time>2023-10-16T04:00:00Z</time></trkpt>
 </trkseg></trk>
</gpx>"""


def test_parse_gpx_with_namespace():
    pts = parse_gpx(GPX)
    assert len(pts) == 2
    assert pts[0].lat == pytest.approx(27.6869)
    assert pts[0].ele == 2860
    assert pts[0].ts == datetime(2023, 10, 15, 4, tzinfo=UTC)


def test_parse_gpx_drops_untimed_points():
    doc = GPX.replace("<time>2023-10-15T04:00:00Z</time>", "")
    assert len(parse_gpx(doc)) == 1


def test_parse_gpx_survives_garbage():
    assert parse_gpx("not xml at all") == []


def test_gpx_roundtrips():
    pts = parse_gpx(GPX)
    again = parse_gpx(write_gpx(pts))
    assert len(again) == 2
    assert again[0].lat == pytest.approx(pts[0].lat, abs=1e-6)
    assert again[1].ts == pts[1].ts


# -- merging -----------------------------------------------------------

def test_merge_sorts_and_dedupes():
    a = [p(6), p(8)]
    b = [p(7, source="phone_kulikov")]
    merged = merge_points([a, b])
    assert [x.ts.hour for x in merged] == [6, 7, 8]


def test_collision_prefers_better_accuracy():
    keller = p(6, lat=27.0, acc=50.0)
    kulikov = p(6, lat=28.0, source="phone_kulikov", acc=5.0)
    assert merge_points([[keller], [kulikov]])[0].lat == 28.0


def test_collision_falls_back_to_keller_when_accuracy_unknown():
    keller = p(6, lat=27.0)
    kulikov = p(6, lat=28.0, source="phone_kulikov")
    assert merge_points([[kulikov], [keller]])[0].lat == 27.0


def test_gpx_outranks_both_phones():
    assert merge_points([[p(6, lat=27.0)],
                         [p(6, lat=29.0, source="gpx")]])[0].lat == 29.0


def test_accuracy_beats_source_rank():
    """A precise kulikov fix should win over a vague gpx point."""
    assert merge_points([[p(6, lat=29.0, source="gpx", acc=200.0)],
                         [p(6, lat=28.0, source="phone_kulikov", acc=4.0)]])[0].lat == 28.0


# -- outliers ----------------------------------------------------------

def test_drops_a_stale_cached_fix():
    """The classic phone EXIF failure: one point still reporting Kathmandu."""
    pts = [p(6, 27.80, 86.71), p(7, 27.72, 86.71), p(8, 27.81, 86.72)]
    bad = GpsPoint(t(7, 30), 27.7172, 85.3240, None, "phone_keller")  # Kathmandu
    got = drop_outliers(sorted(pts + [bad], key=lambda x: x.ts))
    assert bad not in got
    assert len(got) == 3


def test_keeps_plausible_walking_pace():
    pts = [p(6, 27.80, 86.71), p(7, 27.81, 86.72), p(8, 27.82, 86.73)]
    assert len(drop_outliers(pts)) == 3


def test_outlier_filter_noop_on_short_tracks():
    assert len(drop_outliers([p(6), p(7)])) == 2


# -- interpolation -----------------------------------------------------

def test_interpolates_midpoint():
    pts = [p(6, 27.0, 86.0), p(8, 29.0, 88.0)]
    lat, lon = interpolate_at(pts, t(7))
    assert lat == pytest.approx(28.0)
    assert lon == pytest.approx(87.0)


def test_exact_hit_returns_the_point():
    pts = [p(6, 27.0, 86.0), p(8, 29.0, 88.0)]
    assert interpolate_at(pts, t(6)) == (27.0, 86.0)


def test_refuses_to_extrapolate_outside_the_track():
    pts = [p(6, 27.0, 86.0), p(8, 29.0, 88.0)]
    assert interpolate_at(pts, t(5)) is None
    assert interpolate_at(pts, t(9)) is None


def test_refuses_across_a_gap_longer_than_the_limit():
    """A camera clip filmed during an untracked half-day must stay unplaced
    rather than be smeared between two villages."""
    pts = [p(6, 27.0, 86.0, day=15), p(6, 29.0, 88.0, day=16)]   # 24 h apart
    assert interpolate_at(pts, t(18, day=15), max_gap_s=14400) is None


def test_allows_a_gap_within_the_limit():
    pts = [p(6, 27.0, 86.0), p(9, 29.0, 88.0)]                   # 3 h apart
    assert interpolate_at(pts, t(7), max_gap_s=14400) is not None


def test_empty_track_returns_none():
    assert interpolate_at([], t(7)) is None


# -- envelope and coverage ---------------------------------------------

def test_envelope():
    lo, hi = envelope([p(6), p(9)])
    assert (lo.hour, hi.hour) == (6, 9)


def test_coverage_is_the_acceptance_metric():
    pts = [p(6, 27.0, 86.0), p(9, 29.0, 88.0)]
    assert coverage(pts, [t(7), t(8)]) == 1.0
    assert coverage(pts, [t(7), t(20)]) == 0.5
    assert coverage(pts, []) == 0.0


# -- day index ---------------------------------------------------------

def test_day_index_is_one_based():
    start = datetime(2023, 10, 15, 6, tzinfo=NEPAL)
    assert day_index(datetime(2023, 10, 15, 8, tzinfo=NEPAL), start) == 1
    assert day_index(datetime(2023, 10, 17, 8, tzinfo=NEPAL), start) == 3


def test_day_boundary_uses_nepal_local_time():
    """23:40 and 00:20 Nepal time must land on the days a person would name,
    not on whatever the UTC date happens to be."""
    start = datetime(2023, 10, 15, 6, tzinfo=NEPAL)
    late = datetime(2023, 10, 16, 23, 40, tzinfo=NEPAL)
    early = datetime(2023, 10, 17, 0, 20, tzinfo=NEPAL)
    assert day_index(late, start) == 2
    assert day_index(early, start) == 3
    # in UTC both of those fall on the same calendar date, which is the trap
    assert late.astimezone(UTC).date() == early.astimezone(UTC).date()
